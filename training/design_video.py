"""Assigns the weights of the video generator's object experts in closed form.

No training: every object expert is a spectral sine network

    field(x, y) = mean + sum_k a_k * sin(wx_k * x + wy_k * y + phi_k)

whose frequencies, amplitudes and phases are the Fourier coefficients of an
analytic prototype of the object (a signed-distance description rasterised on
a periodic 4 x 2 box). The K largest coefficients are kept (sorted, so each
tier can truncate). A second readout over the same sine neurons gives the
part's shading field (orthogonal projection, also closed form).

Because each sine neuron's phase is explicit, the hypernetwork can translate
an object (sail, rise, fall, drift, flicker) by changing only the neuron's
input bias at run time: the same weights keep generating the object at every
position.

Output: src/render/neural/objects.json   (python3 training/design_video.py)
"""
import json
import os
import numpy as np

BOX_X, BOX_Y = 4.0, 2.0          # periodic box: x in [-2, 2), y in [-0.5, 1.5)
NX, NY = 512, 256
KMAX = 128
BLUR = 0.010                     # occupancy = clip(0.5 - sdf / (2 * BLUR))

x = -BOX_X / 2 + (np.arange(NX) + 0.5) * BOX_X / NX
y = -0.5 + (np.arange(NY) + 0.5) * BOX_Y / NY
X, Y = np.meshgrid(x, y)         # shape (NY, NX)


# ---------------------------------------------------------------- 2-D signed distances (negative inside)
def circle(cx, cy, r):
    return np.hypot(X - cx, Y - cy) - r


def ellipse(cx, cy, a, b, ang=0.0):
    c, s = np.cos(ang), np.sin(ang)
    u, v = (X - cx) * c + (Y - cy) * s, -(X - cx) * s + (Y - cy) * c
    return (np.hypot(u / a, v / b) - 1) * min(a, b)


def box(cx, cy, hw, hh, r=0.0):
    qx, qy = np.abs(X - cx) - hw + r, np.abs(Y - cy) - hh + r
    return np.hypot(np.maximum(qx, 0), np.maximum(qy, 0)) + np.minimum(np.maximum(qx, qy), 0) - r


def seg(ax, ay, bx, by, r, PX=None, PY=None):
    px, py = (X if PX is None else PX) - ax, (Y if PY is None else PY) - ay
    dx, dy = bx - ax, by - ay
    h = np.clip((px * dx + py * dy) / (dx * dx + dy * dy), 0, 1)
    return np.hypot(px - dx * h, py - dy * h) - r


def poly(pts, PX=None, PY=None):
    PX = X if PX is None else PX
    PY = Y if PY is None else PY
    pts = np.asarray(pts, float)
    d = np.full(PX.shape, np.inf)
    sgn = np.ones(PX.shape)
    for i in range(len(pts)):
        a, b = pts[i], pts[i - 1]
        ex, ey = b - a
        wx, wy = PX - a[0], PY - a[1]
        h = np.clip((wx * ex + wy * ey) / (ex * ex + ey * ey), 0, 1)
        d = np.minimum(d, np.hypot(wx - ex * h, wy - ey * h))
        c1 = PY >= a[1]
        c2 = PY < b[1]
        c3 = ex * wy > ey * wx
        flip = (c1 & c2 & c3) | (~c1 & ~c2 & ~c3)
        sgn = np.where(flip, -sgn, sgn)
    return sgn * d


def polyline(pts, r):
    return U(*[seg(*pts[i], *pts[i + 1], r) for i in range(len(pts) - 1)])


def U(*ds):
    return np.minimum.reduce(ds)


def I(a, b):
    return np.maximum(a, b)


def S(a, b):
    return np.maximum(a, -b)


def smin(a, b, k=0.03):
    h = np.clip(0.5 + 0.5 * (b - a) / k, 0, 1)
    return b + (a - b) * h - k * h * (1 - h)


def rep(fn, period, offset=0.0):
    """repeat an object along x with a period dividing the box width"""
    n = BOX_X / period
    assert abs(n - round(n)) < 1e-9, period
    global X
    X0 = X
    X = (X0 - offset + period / 2) % period - period / 2
    try:
        return fn()
    finally:
        X = X0


def rep2(fn, px, py, ox=0.0, oy=0.0):
    for p, B in ((px, BOX_X), (py, BOX_Y)):
        assert abs(B / p - round(B / p)) < 1e-9, p
    global X, Y
    X0, Y0 = X, Y
    X = (X0 - ox + px / 2) % px - px / 2
    Y = (Y0 - oy + py / 2) % py - py / 2
    try:
        return fn()
    finally:
        X, Y = X0, Y0


def heart(cx, cy, s):
    """classic implicit heart (x² + y² - 1)³ - x² y³ <= 0, turned into an approximate distance"""
    u, v = (X - cx) / s, (Y - cy) / s + 0.1
    f = (u * u + v * v - 1) ** 3 - u * u * v ** 3
    gy, gx = np.gradient(f, (Y[1, 0] - Y[0, 0]) / s, (X[0, 1] - X[0, 0]) / s)
    return f / (np.hypot(gx, gy) + 1e-6) * s


def clip01(a):
    return np.clip(a, 0, 1)


# ---------------------------------------------------------------- prototypes
# Each part is rasterised in its own periodic tile `box` centred at `at`
# (absolute screen coordinates are used inside the lambdas). Small objects get a
# small tile, so their spectrum is coarse and a few dozen sine neurons suffice;
# `win` shows a single copy (analytic window neurons at run time), otherwise the
# tile repeats (rows of trees, rain, snow).
P = []


def part(concept, name, depth, sdf, shade=None, c0=(1, 1, 1), c1=None, box=(BOX_X, BOX_Y), at=(0.0, 0.5), win=(False, False),
         vx=0.0, vy=0.0, osc=(0, 0, 0), emissive=False, alpha="concept", kick=0.0, res=128):
    P.append(dict(concept=concept, name=name, depth=depth, sdf=sdf, shade=shade, c0=c0, c1=c1 or c0, box=box, at=at, win=win,
                  vx=vx, vy=vy, osc=osc, emissive=emissive, alpha=alpha, kick=kick, res=res))


H = 0.40  # horizon of landscape prototypes
G = dict(box=(BOX_X, BOX_Y), at=(0.0, 0.5), res=256)   # whole-screen parts
star = lambda cx, cy, s: U(poly([(cx - s, cy), (cx, cy + s * 0.32), (cx + s, cy), (cx, cy - s * 0.32)]),
                           poly([(cx, cy - s), (cx + s * 0.32, cy), (cx, cy + s), (cx - s * 0.32, cy)]), circle(cx, cy, s * 0.3))
cloud = lambda cx, cy, s: I(U(circle(cx, cy, 0.07 * s), circle(cx - 0.08 * s, cy - 0.02 * s, 0.05 * s),
                              circle(cx + 0.085 * s, cy - 0.015 * s, 0.055 * s), circle(cx + 0.03 * s, cy + 0.04 * s, 0.05 * s)),
                            -(Y - (cy - 0.045 * s)))
gull = lambda cx, cy, s: polyline([(cx - 0.05 * s, cy + 0.01 * s), (cx - 0.025 * s, cy + 0.018 * s), (cx, cy),
                                   (cx + 0.025 * s, cy + 0.018 * s), (cx + 0.05 * s, cy + 0.01 * s)], 0.008 * s)
fir = lambda cx, base, h, w: U(poly([(cx - w, base + 0.05 * h), (cx + w, base + 0.05 * h), (cx, base + 0.55 * h)]),
                               poly([(cx - 0.8 * w, base + 0.35 * h), (cx + 0.8 * w, base + 0.35 * h), (cx, base + 0.8 * h)]),
                               poly([(cx - 0.6 * w, base + 0.6 * h), (cx + 0.6 * w, base + 0.6 * h), (cx, base + h)]),
                               box(cx, base, 0.012, 0.05 * h + 0.01))
fish = lambda cx, cy, s: U(ellipse(cx, cy, 0.07 * s, 0.035 * s), poly([(cx - 0.06 * s, cy), (cx - 0.11 * s, cy + 0.035 * s), (cx - 0.11 * s, cy - 0.035 * s)]))
note = lambda cx, cy: U(ellipse(cx, cy, 0.045, 0.032, 0.4), seg(cx + 0.04, cy + 0.008, cx + 0.04, cy + 0.2, 0.01),
                        polyline([(cx + 0.04, cy + 0.2), (cx + 0.085, cy + 0.15), (cx + 0.08, cy + 0.1)], 0.012))
daisy = lambda cx, cy: U(*[circle(cx + 0.028 * np.cos(a), cy + 0.028 * np.sin(a), 0.016) for a in np.arange(8) * np.pi / 4],
                         seg(cx, cy, cx, -0.1, 0.004))
fl = lambda cx, cy, s: smin(circle(cx, cy, 0.07 * s), poly([(cx - 0.065 * s, cy + 0.02 * s), (cx + 0.065 * s, cy + 0.02 * s), (cx + 0.01 * s, cy + 0.31 * s)]), 0.04 * s)

# ---- sky
part("night", "stars", 0, lambda: U(star(-0.3, 0.7, 0.045), star(0.05, 0.86, 0.035), star(0.3, 0.62, 0.03), star(-0.1, 0.58, 0.025)), None, (1, 1, 0.9),
     box=(0.9, 0.4), at=(0, 0.75), win=(False, True), emissive=True, osc=(0.004, 0.004, 0.7))
part("space", "stars", 0, lambda: U(star(-0.3, 0.45, 0.045), star(0.12, 0.75, 0.035), star(0.33, 0.38, 0.03), star(-0.05, 0.6, 0.025)), None, (1, 1, 1),
     box=(0.9, 0.6), at=(0, 0.6), emissive=True, osc=(0.003, 0, 0.5))
part("sunset", "setting sun", 1, lambda: circle(0.12, H + 0.1, 0.15), lambda: clip01((Y - H) / 0.25), (1, 0.45, 0.12), (1, 0.9, 0.45),
     box=(0.5, 0.5), at=(0.12, H + 0.1), win=(True, True), emissive=True, kick=0.15)
part("night", "crescent moon", 1, lambda: S(circle(0.3, 0.8, 0.085), circle(0.345, 0.825, 0.08)), None, (0.97, 0.95, 0.82),
     box=(0.3, 0.3), at=(0.3, 0.8), win=(True, True), emissive=True, kick=0.2)
part("beach", "sun", 1, lambda: circle(-0.45, 0.8, 0.075), None, (1, 0.95, 0.65), box=(0.25, 0.25), at=(-0.45, 0.8), win=(True, True), emissive=True)
part("space", "ringed planet", 1, lambda: U(circle(0.38, 0.66, 0.13), np.abs(ellipse(0.38, 0.66, 0.27, 0.06, -0.25)) - 0.012),
     lambda: clip01(0.5 + (X - 0.38) * 3 + (Y - 0.66) * 2), (0.95, 0.6, 0.35), (0.45, 0.2, 0.25),
     box=(0.75, 0.45), at=(0.38, 0.66), win=(True, True), osc=(0, 0.01, 0.3), kick=0.1)
part("space", "rocket", 2, lambda: U(seg(-0.35, 0.3, -0.35, 0.52, 0.045), poly([(-0.395, 0.52), (-0.305, 0.52), (-0.35, 0.62)]),
                                     poly([(-0.395, 0.34), (-0.44, 0.27), (-0.395, 0.4)]), poly([(-0.305, 0.34), (-0.26, 0.27), (-0.305, 0.4)])),
     lambda: (circle(-0.35, 0.47, 0.02) < 0) * 1.0 + (Y < 0.36) * 0.6, (0.9, 0.9, 0.95), (0.3, 0.6, 1),
     box=(0.3, 0.45), at=(-0.35, 0.45), win=(True, True), vy=0.12, osc=(0.006, 0, 2.0))
part("space", "rocket flame", 2, lambda: smin(circle(-0.35, 0.25, 0.03), poly([(-0.38, 0.26), (-0.32, 0.26), (-0.35, 0.14)])),
     lambda: clip01((Y - 0.16) / 0.1), (1, 0.4, 0.05), (1, 0.95, 0.5), box=(0.3, 0.45), at=(-0.35, 0.45), win=(True, True),
     vy=0.12, osc=(0.006, 0.01, 9.0), emissive=True)
part("clouds", "cumulus", 2, lambda: cloud(0, 0.8, 1.4), lambda: clip01(0.9 - (Y - 0.72) * 6), (1, 1, 1), (0.72, 0.75, 0.82),
     box=(1.0, 0.4), at=(0, 0.8), win=(False, True), vx=0.03)
part("clouds", "cumulus", 2, lambda: cloud(0.35, 0.68, 1.0), lambda: clip01(0.9 - (Y - 0.62) * 6), (1, 1, 1), (0.75, 0.78, 0.85),
     box=(1.3, 0.3), at=(0.35, 0.68), win=(False, True), vx=0.045)
part("storm", "storm clouds", 2, lambda: U(cloud(-0.4, 0.8, 2.2), cloud(0.45, 0.85, 2.4), cloud(1.4, 0.8, 2.0), cloud(-1.3, 0.85, 2.2)),
     lambda: clip01(1 - (Y - 0.62) * 4), (0.42, 0.44, 0.5), (0.2, 0.21, 0.26), vx=0.05, **G)
part("storm", "lightning", 6, lambda: poly([(0.02, 0.78), (-0.06, 0.55), (0.0, 0.56), (-0.07, 0.3), (0.06, 0.6), (0.0, 0.59), (0.08, 0.78)]),
     None, (1, 1, 0.85), box=(0.3, 0.6), at=(0.0, 0.54), win=(True, True), emissive=True, alpha="flash", kick=0.4)
part("birds", "gulls", 3, lambda: U(gull(-0.15, 0.75, 1.7), gull(0.15, 0.83, 1.3)), None, (0.1, 0.1, 0.13),
     box=(0.8, 0.25), at=(0, 0.78), win=(False, True), vx=0.09, osc=(0, 0.012, 3.0))

# ---- far landscape
part("mountain", "peaks", 3, lambda: U(poly([(-2, H - 0.02), (-0.95, H + 0.18), (-0.7, H + 0.35), (-0.45, H + 0.15), (-0.2, H + 0.42), (0.15, H + 0.12),
                                            (0.4, H + 0.33), (0.7, H + 0.1), (0.95, H + 0.28), (2, H - 0.02)]), box(0, H - 0.3, 2, 0.3)),
     lambda: np.maximum(clip01((Y - H - 0.3) / 0.02), 0.45 * clip01((Y - H + 0.05) / 0.35)), (0.22, 0.25, 0.33), (0.95, 0.96, 1), **G)
blds = [(-0.8, 0.07, 0.25), (-0.66, 0.06, 0.38), (-0.52, 0.05, 0.2), (-0.38, 0.07, 0.45), (-0.22, 0.06, 0.3), (-0.08, 0.05, 0.52),
        (0.06, 0.07, 0.27), (0.22, 0.06, 0.42), (0.37, 0.05, 0.33), (0.52, 0.07, 0.5), (0.68, 0.06, 0.24), (0.83, 0.06, 0.36)]
part("city", "skyline", 4, lambda: U(*[box(cx, 0.25 + h / 2 - 0.1, hw, h / 2 + 0.1) for cx, hw, h in blds]),
     lambda: ((np.sin(2 * np.pi * X / (4 / 64)) > 0.2) & (np.sin(2 * np.pi * Y / (2 / 28)) > 0.2)) * 1.0, (0.3, 0.32, 0.4), (1, 0.85, 0.45), **G)
part("desert", "pyramids", 3, lambda: U(poly([(0.05, 0.33), (0.45, 0.33), (0.25, 0.62)]), poly([(0.4, 0.33), (0.7, 0.33), (0.55, 0.52)])),
     lambda: (((X > 0.25) & (X < 0.4)) | (X > 0.55)) * 1.0, (0.93, 0.77, 0.45), (0.7, 0.5, 0.28), box=(0.8, 0.4), at=(0.37, 0.45), win=(True, True), kick=0.1)

# ---- ground / mid
for c, col in (("forest", (0.3, 0.5, 0.2)), ("river", (0.32, 0.52, 0.22)), ("flowers", (0.35, 0.58, 0.25))):
    part(c, "meadow", 4, lambda: box(0, H - 1, 2, 1), lambda: clip01((H - Y) / 0.4), col, tuple(v * 0.5 for v in col), **G)
part("forest", "far firs", 5, lambda: fir(0, H - 0.06, 0.3, 0.06), lambda: clip01((Y - H + 0.06) / 0.3), (0.08, 0.28, 0.16), (0.25, 0.5, 0.3),
     box=(4 / 19, 0.4), at=(0.05, H + 0.1), win=(False, True))
part("forest", "near firs", 5, lambda: fir(0, H - 0.16, 0.38, 0.075), lambda: clip01((Y - H + 0.16) / 0.38), (0.05, 0.2, 0.1), (0.18, 0.42, 0.22),
     box=(4 / 13, 0.5), at=(0.11, H + 0.03), win=(False, True))
part("river", "river", 5, lambda: I(np.abs(X - 0.18 * np.sin(9 * Y) * (H - Y) * 2) - (0.02 + 0.45 * (H - Y)), Y - H),
     lambda: clip01(0.5 + 0.5 * np.sin(2 * np.pi * (Y * 20) + 6 * X)), (0.25, 0.55, 0.8), (0.55, 0.8, 0.95), **G)
part("ocean", "sea", 4, lambda: Y - (H + 0.006 * np.sin(2 * np.pi * X / (4 / 30))),
     lambda: clip01((H - Y) / 0.35) * 0.8 + 0.2 * (np.sin(2 * np.pi * (X / (4 / 24) + Y / (2 / 30))) > 0.6), (0.15, 0.5, 0.72), (0.03, 0.18, 0.38), vx=0.02, **G)
part("ocean", "sailboat", 6, lambda: U(poly([(-0.17, 0.36), (0.17, 0.36), (0.12, 0.3), (-0.12, 0.3)]), seg(0, 0.36, 0, 0.66, 0.006),
                                       poly([(0.01, 0.39), (0.01, 0.65), (0.15, 0.4)]), poly([(-0.01, 0.42), (-0.01, 0.62), (-0.12, 0.42)])),
     lambda: ((Y > 0.37) & (np.abs(X) > 0.008)) * 1.0, (0.45, 0.2, 0.1), (0.97, 0.97, 0.95),
     box=(0.45, 0.45), at=(0, 0.48), win=(True, True), vx=0.04, osc=(0.0, 0.008, 1.3), kick=0.2)
part("desert", "dunes", 4, lambda: Y - (0.33 + 0.04 * np.sin(2 * np.pi * X / 2 + 1) + 0.02 * np.sin(2 * np.pi * X / 0.5)),
     lambda: clip01((0.33 - Y) / 0.3), (0.96, 0.8, 0.5), (0.8, 0.56, 0.3), **G)
part("desert", "cactus", 5, lambda: U(seg(-0.4, 0.2, -0.4, 0.5, 0.03), polyline([(-0.4, 0.3), (-0.48, 0.3), (-0.48, 0.42)], 0.022),
                                      polyline([(-0.4, 0.36), (-0.33, 0.36), (-0.33, 0.46)], 0.022)),
     lambda: clip01(0.5 + (X + 0.4) * 12), (0.15, 0.45, 0.2), (0.3, 0.65, 0.3), box=(0.3, 0.45), at=(-0.4, 0.36), win=(True, True), kick=0.1)
part("beach", "sea", 4, lambda: Y - H, lambda: clip01((H - Y) / 0.2), (0.2, 0.6, 0.78), (0.06, 0.3, 0.55), **G)
part("beach", "sand", 5, lambda: Y - (0.2 + 0.015 * np.sin(2 * np.pi * X / 1.0)), lambda: clip01((Y - 0.12) / 0.08), (0.95, 0.86, 0.62), (1, 1, 1), **G)
part("beach", "palm", 6, lambda: U(polyline([(0.35, 0.1), (0.36, 0.25), (0.39, 0.4), (0.44, 0.52)], 0.016),
                                   *[polyline([(0.44, 0.52), (0.44 + 0.08 * np.cos(a), 0.52 + 0.06 * np.sin(a)),
                                               (0.44 + 0.16 * np.cos(a), 0.52 + 0.06 * np.sin(a) - 0.06)], 0.012) for a in (0.2, 0.9, 1.6, 2.3, 3.0)]),
     lambda: (Y > 0.5) * 1.0, (0.45, 0.3, 0.15), (0.12, 0.45, 0.15), box=(0.5, 0.62), at=(0.42, 0.35), win=(True, True), osc=(0.005, 0, 1.1))
part("beach", "umbrella", 6, lambda: U(I(circle(-0.3, 0.3, 0.13), -(Y - 0.3)), seg(-0.3, 0.12, -0.3, 0.3, 0.007)),
     lambda: (np.sin(np.arctan2(Y - 0.3, X + 0.3) * 8) > 0) * 1.0, (0.9, 0.15, 0.15), (1, 1, 1), box=(0.36, 0.36), at=(-0.3, 0.28), win=(True, True))
part("fields", "crops", 4, lambda: Y - (H - 0.02 + 0.03 * np.sin(2 * np.pi * X / 2)),
     lambda: clip01(0.5 + 0.5 * np.sin(2 * np.pi * (X / (4 / 40) + Y / (2 / 30)))), (0.4, 0.62, 0.22), (0.85, 0.72, 0.3), **G)
part("fields", "farmhouse", 5, lambda: U(box(-0.3, 0.42, 0.11, 0.09), poly([(-0.43, 0.5), (-0.17, 0.5), (-0.3, 0.62)])),
     lambda: ((Y > 0.505) | (box(-0.3, 0.38, 0.025, 0.05) < 0)) * 1.0, (0.9, 0.86, 0.75), (0.7, 0.15, 0.1),
     box=(0.36, 0.36), at=(-0.3, 0.46), win=(True, True), kick=0.15)

# ---- foreground objects
part("snow", "snowfield", 4, lambda: Y - (0.22 + 0.02 * np.sin(2 * np.pi * X / 1.0)), lambda: clip01((0.22 - Y) / 0.2), (0.95, 0.97, 1), (0.8, 0.85, 0.95), **G)
part("snow", "snowman", 6, lambda: U(circle(0.3, 0.2, 0.1), circle(0.3, 0.36, 0.075), circle(0.3, 0.48, 0.055), box(0.3, 0.56, 0.045, 0.035), box(0.3, 0.525, 0.07, 0.008)),
     lambda: ((box(0.3, 0.55, 0.07, 0.045) < 0) | (circle(0.28, 0.5, 0.008) < 0) | (circle(0.32, 0.5, 0.008) < 0) |
              (circle(0.3, 0.38, 0.009) < 0) | (circle(0.3, 0.33, 0.009) < 0)) * 1.0, (0.97, 0.97, 1), (0.1, 0.1, 0.12),
     box=(0.3, 0.56), at=(0.3, 0.36), win=(True, True), kick=0.15)
part("snow", "snowflakes", 7, lambda: U(circle(-0.15, 0.6, 0.014), circle(0.05, 0.42, 0.012), circle(0.17, 0.66, 0.016), circle(-0.02, 0.32, 0.01)), None, (1, 1, 1),
     box=(0.45, 0.45), at=(0, 0.5), vy=-0.12, osc=(0.015, 0, 0.7))
part("rain", "rain", 7, lambda: U(seg(0, 0.44, -0.012, 0.54, 0.0055), seg(0.05, 0.6, 0.038, 0.7, 0.0055)),
     None, (0.75, 0.8, 0.92), box=(0.12, 0.3), at=(0, 0.5), vy=-1.4, vx=-0.15)
part("fire", "flames", 6, lambda: U(fl(0, 0.15, 1.0), fl(-0.08, 0.13, 0.65), fl(0.08, 0.13, 0.7)),
     lambda: clip01(-U(fl(0, 0.15, 1.0), fl(-0.08, 0.13, 0.65), fl(0.08, 0.13, 0.7)) / 0.03), (0.95, 0.25, 0.02), (1, 0.92, 0.45),
     box=(0.4, 0.5), at=(0, 0.25), win=(True, True), emissive=True, osc=(0.006, 0.012, 7.0))
part("fire", "logs", 7, lambda: U(seg(-0.16, 0.06, 0.16, 0.12, 0.025), seg(-0.16, 0.12, 0.16, 0.06, 0.025)), None, (0.35, 0.2, 0.1),
     box=(0.4, 0.2), at=(0, 0.09), win=(True, True))
part("underwater", "seaweed", 5, lambda: U(polyline([(0, 0), (0.02, 0.1), (-0.02, 0.2), (0.015, 0.3)], 0.012), box(0, -0.2, 0.3, 0.25)),
     lambda: (Y < 0.05) * 1.0, (0.1, 0.5, 0.3), (0.85, 0.78, 0.55), box=(0.3, 0.8), at=(0.1, 0.2), win=(False, True), osc=(0.01, 0, 0.8))
part("underwater", "fish", 6, lambda: U(fish(-0.1, 0.62, 1.3), fish(0.2, 0.35, 1.0)),
     lambda: ((circle(-0.1 + 0.065, 0.63, 0.007) < 0) | (circle(0.2 + 0.05, 0.357, 0.006) < 0)) * 1.0, (1, 0.6, 0.1), (0.05, 0.05, 0.08),
     box=(0.8, 0.6), at=(0.05, 0.5), win=(False, True), vx=0.07, osc=(0, 0.01, 1.2))
part("underwater", "bubbles", 7, lambda: U(circle(-0.1, 0.42, 0.018), circle(0.08, 0.6, 0.013), circle(0.12, 0.33, 0.01)), lambda: clip01((Y % 0.1) / 0.1), (0.55, 0.8, 0.95), (0.95, 1, 1),
     box=(0.4, 0.4), at=(0, 0.5), vy=0.1, osc=(0.01, 0, 1.0))
part("technology", "robot", 6, lambda: U(box(0, 0.6, 0.13, 0.1, 0.03), box(0, 0.3, 0.16, 0.15, 0.02), seg(0, 0.7, 0, 0.8, 0.006), circle(0, 0.81, 0.02),
                                         box(-0.2, 0.32, 0.03, 0.1, 0.02), box(0.2, 0.32, 0.03, 0.1, 0.02)),
     lambda: (U(circle(-0.05, 0.62, 0.025), circle(0.05, 0.62, 0.025), box(0, 0.55, 0.06, 0.008), box(0, 0.32, 0.08, 0.05, 0.01), circle(0, 0.81, 0.02)) < 0) * 1.0,
     (0.62, 0.66, 0.74), (0.2, 1, 1), box=(0.5, 0.72), at=(0, 0.5), win=(True, True), osc=(0, 0.01, 1.5), kick=0.1)
part("music", "notes", 6, lambda: U(note(-0.15, 0.35), note(0.12, 0.62)), lambda: clip01((X + 0.25) / 0.5), (0.08, 0.05, 0.2), (0.15, 0.1, 0.4),
     box=(0.6, 0.6), at=(0, 0.5), vy=0.07, osc=(0.02, 0, 1.6))
part("love", "hearts", 6, lambda: U(heart(0, 0.45, 0.13), heart(0.3, 0.75, 0.06)), lambda: clip01(0.5 + (Y - 0.45) * 3), (0.75, 0.02, 0.15), (1, 0.3, 0.45),
     box=(0.75, 0.7), at=(0.1, 0.55), vy=0.05, osc=(0.01, 0, 1.1))
part("flowers", "daisies", 6, lambda: U(daisy(-0.1, 0.22), daisy(0.12, 0.14)),
     lambda: (U(circle(-0.1, 0.22, 0.014), circle(0.12, 0.14, 0.014)) < 0) * 1.0, (1, 0.55, 0.75), (1, 0.85, 0.15),
     box=(0.44, 0.34), at=(0, 0.17), win=(False, True), osc=(0.006, 0, 1.4))


# ---------------------------------------------------------------- spectral projection
def grid(p):
    """set the global X, Y to the part's tile (absolute coordinates)"""
    global X, Y, x, y, NX, NY, BX, BY
    BX, BY = p["box"]
    n = p["res"]
    NX = n * 2 if BX / BY >= 1.5 else n
    NY = n if BX / BY >= 1.5 else (n * 2 if BY / BX >= 1.5 else n)
    x = p["at"][0] - BX / 2 + (np.arange(NX) + 0.5) * BX / NX
    y = p["at"][1] - BY / 2 + (np.arange(NY) + 0.5) * BY / NY
    X, Y = np.meshgrid(x, y)


def spectral(occ, shade, kmax):
    f = occ - occ.mean()
    F = np.fft.fft2(f)
    G = np.fft.fft2(shade - shade.mean())
    ky = np.fft.fftfreq(NY, BY / NY)  # cycles / unit
    kx = np.fft.fftfreq(NX, BX / NX)
    KX, KY = np.meshgrid(kx, ky)
    half = (KX > 0) | ((KX == 0) & (KY > 0))
    half &= (np.abs(KX) < NX / BX / 2 - 1e-9) & (np.abs(KY) < NY / BY / 2 - 1e-9)
    score = np.where(half, np.abs(F) + 0.5 * np.abs(G), 0)
    idx = np.argsort(score.ravel())[::-1][:kmax]
    N = NX * NY
    modes = []
    for i in idx:
        r, c = divmod(i, NX)
        wx, wy = 2 * np.pi * KX[r, c], 2 * np.pi * KY[r, c]
        # sample (i, j) sits at local (x[i] - at_x, y[j] - at_y): phases refer to the tile centre
        shift = wx * (x[0] - AT[0]) + wy * (y[0] - AT[1])
        cf, cg = F[r, c], G[r, c]
        # one neuron cos(theta + psi) serves both readouts: psi maximises the jointly
        # explained energy |F|² cos²(argF - psi) + λ|G|² cos²(argG - psi)  =>  psi = ½ arg(F² + λG²)
        psi = 0.5 * np.angle(cf * cf + 0.5 * cg * cg)
        a = 2 * np.real(cf * np.exp(-1j * psi)) / N
        b = 2 * np.real(cg * np.exp(-1j * psi)) / N
        phi = psi + np.pi / 2 - shift
        modes.append([wx, wy, (phi + np.pi) % (2 * np.pi) - np.pi, a, b])
    return modes


def reconstruct(modes, mean, K):
    v = np.full(X.shape, mean)
    for wx, wy, phi, a, _ in modes[:K]:
        v += a * np.sin(wx * (X - AT[0]) + wy * (Y - AT[1]) + phi)
    return v


def main():
    out = dict(box=[BOX_X, BOX_Y], kmax=KMAX, blur=BLUR, parts=[])
    report = []
    global AT
    for p in sorted(P, key=lambda q: q["depth"]):
        grid(p)
        AT = p["at"]
        occ = clip01(0.5 - p["sdf"]() / (2 * BLUR))
        sh = np.zeros_like(X) if p["shade"] is None else clip01(np.asarray(p["shade"](), float) * np.ones_like(X))
        modes = spectral(occ, sh, KMAX)
        errs = []
        for K in (40, 64, 80, 128):
            rec = reconstruct(modes, occ.mean(), K) > 0.5
            tgt = occ > 0.5
            errs.append(float((rec != tgt).sum()) / max(1, tgt.sum()))
        report.append(f"{p['concept']:>11} {p['name']:<15} area {occ.mean():.3f}  mask error K=40/64/80/128: " + " ".join(f"{e:.2f}" for e in errs))
        out["parts"].append(dict(
            concept=p["concept"], name=p["name"], depth=p["depth"], c0=list(p["c0"]), c1=list(p["c1"]),
            box=list(p["box"]), at=list(p["at"]), win=list(p["win"]), vx=p["vx"], vy=p["vy"], osc=list(p["osc"]), emissive=p["emissive"], alpha=p["alpha"], kick=p["kick"],
            mean=round(float(occ.mean()), 5), shadeMean=round(float(sh.mean()), 5),
            modes=[[float(f"{v:.4g}") for v in m] for m in modes],
        ))
    path = os.path.join(os.path.dirname(__file__), "..", "src", "render", "neural", "objects.json")
    with open(path, "w") as f:
        json.dump(out, f, separators=(",", ":"))
    print("\n".join(report))
    print(f"{len(P)} object experts -> {os.path.relpath(path)} ({os.path.getsize(path) / 1024:.0f} KB)")


if __name__ == "__main__":
    main()
