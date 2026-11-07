// Neural video generator: weight assignment (no training, no predefined shapes).
//
// The pixel network is a compositional pattern-producing network (CPPN):
//
//   [x, y, |x|, r]  ─►  Fourier layer (N1 sines)  ─►  tanh layer (N2)  ─►  tanh layer (N3)
//                                                   ─►  K sigmoid "paint" channels  ─►  layered colour
//
// It contains no object, sprite or shape: there is no "sun neuron" or "tree
// neuron". Forms emerge from the composition of Fourier features through
// random-orthogonal tanh layers. What it hears does not select pictures; it
// sets the *dynamics* of the network through a hypernetwork:
//
//   * gain of the tanh layers (mean-field theory of deep random networks:
//     gain < 1 → ordered, smooth forms; gain ≫ 1 → chaotic, intricate forms),
//   * spatial scale and anisotropy of the Fourier basis (stratified "landscape"
//     bands vs isotropic blobs), bilateral symmetry (|x| input), radial focus,
//   * flow velocity and turbulence (phases integrated over time, domain warp),
//   * biases of every hidden neuron from an 8-d "form" latent, so each concept
//     and each heard word pushes the network to a different, reproducible
//     attractor (say the same thing → same family of images),
//   * a generated colour palette (opponent-colour hue, lightness, chroma,
//     harmonic spacing) and the sharpness of the paint channels.
//
// The only knowledge linking meaning to appearance is a small "synesthetic"
// table of 12 low-level visual statistics per concept (hue tendency,
// lightness, complexity, scale, flow, turbulence, stratification, symmetry,
// sharpness). Shapes and colours themselves are produced by the network.
import { FrameNet, PixelNet, Param, type PL } from "./graph";
import { CONCEPTS } from "../../nn/text";
import OBJECTS from "./objects.json";

const TAU = Math.PI * 2;
export const FORM_DIM = 8;

function rng(seed: number) {
  let s = seed >>> 0;
  return () => {
    s = (s + 0x6d2b79f5) >>> 0;
    let t = s;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}
const gauss = (r: () => number) => Math.sqrt(-2 * Math.log(r() + 1e-12)) * Math.cos(TAU * r());

/** deterministic matrix with orthonormal rows (Gram–Schmidt on a seeded Gaussian) */
function orthoMatrix(rows: number, cols: number, seed: number): number[][] {
  const r = rng(seed);
  const m: number[][] = [];
  for (let i = 0; i < rows; i++) {
    let v = Array.from({ length: cols }, () => gauss(r));
    for (const u of m.slice(Math.max(0, i - cols + 1))) {
      const dd = v.reduce((s, x, k) => s + x * u[k], 0);
      v = v.map((x, k) => x - dd * u[k]);
    }
    const n = Math.hypot(...v) || 1;
    m.push(v.map((x) => x / n));
  }
  return m;
}

// Synesthetic prior: low-level visual statistics of each concept.
//   a: red(+)/green(-)  b: yellow(+)/blue(-)  L: lightness  ch: chroma
//   cx: complexity (layer gain)  sc: spatial scale  fx, fy: flow  tu: turbulence
//   st: stratification (horizontal bands)  sy: bilateral symmetry  sh: sharpness
const DIMS = ["a", "b", "L", "ch", "cx", "sc", "fx", "fy", "tu", "st", "sy", "sh"] as const;
type Desc = Record<(typeof DIMS)[number], number>;
const d = (...v: number[]): Desc => Object.fromEntries(DIMS.map((k, i) => [k, v[i]])) as Desc;
export const SYNESTHESIA: Record<(typeof CONCEPTS)[number], Desc> = {
  ocean: d(-0.3, -0.75, 0, 0.7, 0.35, 0.35, 0.35, 0, 0.3, 0.8, 0, 0.25),
  beach: d(0.1, 0.55, 0.45, 0.55, 0.3, 0.3, 0.15, 0, 0.15, 0.85, 0, 0.3),
  forest: d(-0.7, 0.25, -0.15, 0.7, 0.6, 0.6, 0, 0.05, 0.25, 0.5, 0.2, 0.55),
  mountain: d(-0.05, -0.25, 0.05, 0.3, 0.55, 0.25, 0.05, 0, 0.2, 0.7, 0.15, 0.7),
  desert: d(0.35, 0.7, 0.35, 0.6, 0.25, 0.2, 0.2, 0, 0.2, 0.9, 0, 0.3),
  city: d(0, -0.15, -0.15, 0.25, 0.75, 0.8, 0.1, 0, 0.1, 0.4, 0.3, 0.9),
  space: d(0.1, -0.6, -0.7, 0.6, 0.55, 0.7, 0.05, 0.03, 0.45, 0, 0.2, 0.4),
  night: d(0, -0.5, -0.8, 0.4, 0.4, 0.5, 0.05, 0, 0.2, 0.3, 0, 0.4),
  rain: d(-0.1, -0.4, -0.3, 0.25, 0.6, 0.8, 0.05, -0.9, 0.3, 0.1, 0, 0.5),
  snow: d(0, -0.2, 0.75, 0.15, 0.45, 0.7, 0.05, -0.35, 0.2, 0.3, 0.2, 0.4),
  fire: d(0.8, 0.65, 0.25, 1, 0.8, 0.55, 0, 0.8, 0.9, 0.2, 0.3, 0.6),
  sunset: d(0.7, 0.55, 0.1, 0.9, 0.3, 0.25, 0.05, 0, 0.2, 0.85, 0, 0.2),
  river: d(-0.35, -0.55, 0.05, 0.6, 0.4, 0.45, 0, -0.45, 0.45, 0.5, 0.5, 0.35),
  flowers: d(0.7, 0.3, 0.3, 0.9, 0.65, 0.75, 0.05, 0.05, 0.3, 0.3, 0.6, 0.7),
  storm: d(-0.05, -0.35, -0.55, 0.3, 0.85, 0.55, 0.5, -0.3, 0.9, 0.3, 0, 0.6),
  underwater: d(-0.45, -0.6, -0.25, 0.7, 0.45, 0.4, 0.15, 0.25, 0.7, 0.2, 0.2, 0.3),
  technology: d(-0.25, -0.75, -0.35, 0.9, 0.9, 0.85, 0, -0.3, 0.1, 0.2, 0.7, 1),
  music: d(0.55, -0.4, 0.05, 0.95, 0.7, 0.5, 0.3, 0, 0.5, 0.2, 0.8, 0.7),
  love: d(0.85, 0.15, 0.3, 0.85, 0.35, 0.35, 0, 0.2, 0.2, 0.2, 0.95, 0.5),
  clouds: d(0, -0.15, 0.55, 0.2, 0.25, 0.35, 0.3, 0, 0.6, 0.5, 0, 0.15),
  birds: d(0, -0.3, 0.35, 0.35, 0.5, 0.65, 0.6, 0.1, 0.3, 0.4, 0.4, 0.7),
  fields: d(-0.3, 0.6, 0.25, 0.7, 0.45, 0.55, 0.2, 0, 0.2, 0.8, 0, 0.45),
};
/** what the network drifts to when nothing meaningful has been heard */
const NEUTRAL = d(0, 0.05, 0.05, 0.45, 0.4, 0.4, 0.1, 0.02, 0.35, 0.3, 0.1, 0.35);

/** animation tempo multiplier from arousal (used for the director's clock) */
export const tempo = (arousal: number) => 0.6 + 0.9 * (arousal * 0.5 + 0.5);

/** Per-word identity: a deterministic unit vector derived from the word's hash. */
export function wordKick(hash: number): Float32Array {
  const r = rng(hash ^ 0x9e3779b9);
  const v = Float32Array.from({ length: FORM_DIM }, () => gauss(r));
  const n = Math.hypot(...v) || 1;
  return v.map((x) => x / n);
}

/** Layer widths per tier: [Fourier, hidden 1, hidden 2, paint channels, sine neurons per object expert] */
export const WIDTHS = [[14, 12, 10, 4, 40], [20, 16, 12, 4, 64], [24, 20, 16, 5, 80], [32, 24, 20, 6, 128]] as const;

interface ObjectPart {
  concept: string; name: string; depth: number; c0: number[]; c1: number[];
  box: number[]; at: number[]; win: boolean[]; vx: number; vy: number; osc: number[];
  emissive: boolean; alpha: string; kick: number; mean: number; shadeMean: number; modes: number[][];
}

export function design(detail = 2): PixelNet {
  const F = new FrameNet();
  const N = new PixelNet(F);
  const { X, Y, X2, Y2 } = N;
  const [N1, N2, N3, K, KB] = WIDTHS[Math.max(0, Math.min(3, detail))];

  // ------------------------------------------------ hypernetwork inputs
  const C = CONCEPTS.map((c) => F.input(c));
  const val = F.input("val"), aro = F.input("aro");
  const rms = F.input("rms"), mid = F.input("mid"), flux = F.input("flux");
  const flash = F.input("flash"), iasp = F.input("iasp"), zoom = F.input("zoom");
  const words = Array.from({ length: FORM_DIM }, (_, k) => F.input("w" + k));
  const add = (...p: PL[]) => F.add(...p), sub = (a: PL, b: PL) => F.sub(a, b);
  const mul = (a: PL, b: PL) => F.mul(a, b), sc = (a: PL, k: number) => F.scale(a, k);
  const clip = (a: PL) => F.clip(a);
  /** materialise an expression as one hypernetwork unit (keeps products from expanding pairwise) */
  const unit = (a: PL) => F.act(a, "id");

  // ------------------------------------------------ semantic code → visual latent (weighted mean of the prior)
  const prior = 0.15;
  const norm = F.act(add(prior, ...C), "inv");
  const lat = {} as Record<(typeof DIMS)[number], Param>;
  for (const k of DIMS) {
    lat[k] = unit(mul(add(prior * NEUTRAL[k], ...C.map((c, i) => sc(c, SYNESTHESIA[CONCEPTS[i]][k]))), norm));
  }
  // affect and voice modulate the same latent
  const L = clip(add(0.5, sc(lat.L, 0.5), sc(val, 0.1), sc(rms, 0.1)));
  const hueA = unit(add(lat.a, sc(val, 0.15))), hueB = unit(add(lat.b, sc(val, 0.2)));
  const chroma = clip(add(lat.ch, sc(val, 0.15)));
  const cx = clip(add(lat.cx, sc(aro, 0.2), sc(rms, 0.25)));
  const turb = clip(add(lat.tu, sc(aro, 0.2), sc(flux, 0.4)));
  const sharp = clip(add(lat.sh, sc(aro, 0.15)));
  const speed = unit(add(0.6, sc(aro, 0.35), sc(rms, 0.4)));
  const strat = clip(lat.st), sym = clip(sc(lat.sy, 1.3)), asym = unit(sub(1, sym));
  // form latent: concept identities (fixed random directions) + heard-word kicks
  const ident = CONCEPTS.map((_, i) => { const r = rng(1000 + i); return Array.from({ length: FORM_DIM }, () => gauss(r) * 0.8); });
  const form = words.map((w, k) => unit(add(w, mul(add(...C.map((c, i) => sc(c, ident[i][k]))), norm))));

  // ------------------------------------------------ input encoding: x, |x| (symmetry), y, r (radial focus)
  const AX = N.n("abs", [[X, 1]]);
  const R = N.n("sqrt", [[X2, 1], [Y2, 1], [Y, -1]], 0.25, "radius"); // distance to the centre (0, 0.5)

  // slow warp field: turbulence bends the Fourier basis (domain warping)
  const warp: number[] = [];
  for (let k = 0; k < 2; k++) {
    const ph = F.integrate(add(0.13 + 0.07 * k, sc(turb, 0.5), sc(flux, 2)));
    const wx = k ? 1.7 : 2.3;
    warp.push(N.n("sin", [[X, sc(asym, wx)], [AX, sc(sym, wx)], [Y, k ? -2.9 : 1.9]], add(ph, k * 2)));
  }

  // ------------------------------------------------ layer 1: Fourier features
  // orientations on the golden angle, radii log-spaced over 4 octaves: an even
  // multi-scale basis (assigned, not sampled). Scale, anisotropy, symmetry,
  // flow and turbulence are generated per frame.
  const scale = unit(add(1.2, sc(lat.sc, 5)));
  const xs = unit(mul(scale, sub(1, sc(strat, 0.85)))); // stratified scenes vary mostly along y
  const h1: number[] = [];
  for (let i = 0; i < N1; i++) {
    const th = i * 2.399963, rad = Math.pow(2, (4 * i) / (N1 - 1));
    const wx = sc(xs, rad * Math.cos(th)), wy = sc(scale, rad * Math.sin(th));
    // phase velocity = −k·v (pattern flows with v) + intrinsic shimmer + onset kicks
    const vel = unit(add(sc(mul(wx, lat.fx), -1.2), sc(mul(wy, lat.fy), -1.2)));
    const ph = F.integrate(add(mul(vel, speed), sc(add(0.15, turb), 0.6 * Math.sin(i * 1.7)), sc(flux, 3 * Math.cos(i))));
    h1.push(N.n("sin", [
      [X, mul(wx, asym)], [AX, mul(wx, sym)], [Y, wy],
      [warp[0], sc(turb, 1.4 * Math.cos(i))], [warp[1], sc(turb, 1.4 * Math.sin(i))],
    ], add(ph, (i * 0.618 * TAU) % TAU)));
  }

  // ------------------------------------------------ layers 2, 3: orthogonal tanh, gain = complexity
  const r2 = rng(7);
  const W2 = orthoMatrix(N2, N1, 21), W3 = orthoMatrix(N3, N2, 22), Wo = orthoMatrix(K, N3, 23);
  const B2 = orthoMatrix(N2, FORM_DIM, 24), B3 = orthoMatrix(N3, FORM_DIM, 25), Bo = orthoMatrix(K, FORM_DIM, 26);
  const fbias = (row: number[], k: number) => add(...row.map((w, j) => sc(form[j], w * k)));
  const g2 = add(1, sc(cx, 3)), g3 = add(1, sc(cx, 2.2));
  const h2: number[] = [];
  for (let j = 0; j < N2; j++) {
    h2.push(N.n("tanh", [
      ...h1.map((s, i) => [s, W2[j][i]] as [number, PL]),
      [Y, sc(strat, 3 * gauss(r2))], [R, 1.5 * gauss(r2)],
    ], fbias(B2[j], 0.9), undefined, g2));
  }
  const h3: number[] = [];
  for (let j = 0; j < N3; j++) {
    h3.push(N.n("tanh", h2.map((s, i) => [s, W3[j][i]] as [number, PL]), fbias(B3[j], 0.7), undefined, g3));
  }

  // ------------------------------------------------ paint channels + generated palette
  const gOut = add(2, sc(sharp, 12));
  // opponent colour space → RGB
  const rgb = (Lp: PL, a: PL, b: PL): Param[] => [
    add(Lp, sc(a, 0.55), sc(b, 0.35)), add(Lp, sc(a, -0.45), sc(b, 0.15)), add(Lp, sc(a, -0.1), sc(b, -0.6)),
  ];
  const ca = unit(sc(mul(hueA, chroma), 0.75)), cb = unit(sc(mul(hueB, chroma), 0.75));
  N.layer({ name: "ground tone", c0: rgb(sc(L, 0.55), sc(ca, 0.8), sc(cb, 0.8)), c1: rgb(sc(L, 0.9), ca, cb), shade: Y, alpha: 1 });
  // harmonic palette: hue rotated by k·Δ, Δ grows with turbulence (analogous → more contrasting)
  const delta = add(0.12, sc(turb, 0.4));
  for (let k = 0; k < K; k++) {
    const u = k / (K - 1) - 0.5; // same total palette span on every tier
    const rot = sc(delta, 2 * u);
    const cs = F.cos(rot), sn = F.sin(rot);
    const a = sub(mul(ca, cs), mul(cb, sn)), b = add(mul(ca, sn), mul(cb, cs));
    const Lk = clip(add(L, 0.4 * u, sc(chroma, 0.05)));
    const paint = N.n("sig", h3.map((s, i) => [s, Wo[k][i] * Math.sqrt(N3) * 0.6] as [number, PL]),
      add(-0.4 - 0.25 * k, fbias(Bo[k], 0.6)), `paint ${k}`, gOut);
    N.layer({ name: `paint ${k}`, gate: paint, shade: h3[k % N3], c0: rgb(sc(Lk, 0.8), a, b), c1: rgb(Lk, sc(a, 1.2), sc(b, 1.2)), alpha: 0.92 });
  }
  // ------------------------------------------------ object experts (spectral sine networks, closed-form weights)
  // Weights = Fourier coefficients of object prototypes (training/design_video.py).
  // The hypernetwork moves each object by shifting its local coordinates, so the
  // same sine neurons generate it at every position (sailing, rising, falling,
  // drifting, flickering); heard words nudge single objects sideways.
  {
    const parts = (OBJECTS as { blur: number; parts: ObjectPart[] }).parts;
    const blur = (OBJECTS as { blur: number }).blur;
    // portrait screens see a zoomed-out scene so objects stay in view
    const Xs = N.n("lin", [[X, zoom]], 0, "scene x");
    const Ys = N.n("lin", [[Y, zoom]], sc(sub(1, zoom), 0.5), "scene y");
    const dim = sub(1, sc(clip(add(C[CONCEPTS.indexOf("night")], sc(C[CONCEPTS.indexOf("space")], 0.9), sc(C[CONCEPTS.indexOf("storm")], 0.5))), 0.6));
    const total = clip(add(...C));
    // the abstract atmosphere recedes behind recognisable objects
    for (const l of N.layers) if (l.name.startsWith("paint")) l.alpha = add(0.92, sc(total, -0.6));
    parts.forEach((p, idx) => {
      const ci = CONCEPTS.indexOf(p.concept as never);
      const wrapX = p.win[0] ? 3.6 : p.box[0], wrapY = p.win[1] ? 2.2 : p.box[1];
      const disp = (v: number, wrap: number) => (v ? sc(F.integrate(TAU * v / wrap), wrap / TAU) : add(0));
      const osc = p.osc[2] ? F.sin(F.integrate(p.osc[2])) : add(0);
      const dx = add(disp(p.vx, wrapX), sc(osc, p.osc[0]), sc(words[idx % FORM_DIM], p.kick));
      const dy = add(disp(p.vy, wrapY), sc(F.cos(F.integrate(p.osc[2] || 1)), p.osc[1]));
      const u = N.n("lin", [[Xs, 1]], sub(-p.at[0], dx), `${p.name} u`);
      const v = N.n("lin", [[Ys, 1]], sub(-p.at[1], dy), `${p.name} v`);
      const [m, sh] = N.bank(p.modes, u, v, p.name);
      let gate = N.n("edge", [[m, 2 * blur]], (p.mean - 0.5) * 2 * blur, `${p.name} mask`);
      // windows: show one copy of a single object (the spectral field is periodic in its tile)
      if (p.win[0]) gate = N.prod(gate, N.n("edge", [[N.n("abs", [[u, 1]]), -1]], 0.48 * p.box[0]));
      if (p.win[1]) gate = N.prod(gate, N.n("edge", [[N.n("abs", [[v, 1]]), -1]], 0.48 * p.box[1]));
      // a soft threshold removes the spectral ringing of two-tone shading
      const shade = N.n("sig", [[sh, 7]], 7 * (p.shadeMean - 0.5));
      const lit = p.emissive ? 1 : dim;
      // opacity floor: objects of topics that are fading out are skipped for the whole frame
      // (conditional computation) instead of costing a full sine bank at ~1% opacity
      const on = clip(sc(sub(C[ci], 0.1), 1 / 0.9));
      const alpha = p.alpha === "flash" ? mul(flash, on) : on;
      N.layer({ name: `${p.concept}: ${p.name}`, gate, shade, c0: p.c0.map((c) => sc(lit, c)), c1: p.c1.map((c) => sc(lit, c)), alpha });
    });
    N.bankK = KB;
  }

  // voice: glow from the centre with loudness / mid band
  N.layer({ name: "voice glow", gate: N.n("gauss", [[R, 2.2]]), c0: rgb(L, ca, cb), alpha: add(sc(rms, 0.25), sc(mid, 0.1)), add: true });

  // ------------------------------------------------ grading: arousal contrast, onset pulse, lightning flash, vignette
  const k = add(1, sc(aro, 0.25));
  const pulse = add(1, sc(flux, 0.25));
  const gain = mul(k, pulse);
  const fl = mul(flash, C[CONCEPTS.indexOf("storm")]);
  N.grade = {
    M: [gain, 0, 0, 0, gain, 0, 0, 0, gain].map((v) => add(v)),
    o: [0.8, 0.85, 1].map((c) => add(mul(sc(sub(1, k), 0.5), pulse), sc(fl, c))),
  };
  const vr = N.n("sqrt", [[X2, mul(iasp, iasp)], [Y2, 1], [Y, -1]], 0.25);
  N.vignette = N.n("lin", [[vr, -0.35]], 1, "vignette");
  return N;
}

export interface FrameState {
  concepts: ArrayLike<number>; valence: number; arousal: number;
  audio: { rms: number; low: number; mid: number; flux: number };
  /** heard-word form latent (FORM_DIM) */
  form?: ArrayLike<number>;
  dt: number; flash: number; aspect: number;
}

/** Hypernetwork inputs for one frame. */
export function frameInputs(o: FrameState): Record<string, number> {
  const r: Record<string, number> = {
    val: o.valence, aro: o.arousal, rms: o.audio.rms, low: o.audio.low, mid: o.audio.mid, flux: o.audio.flux,
    dt: o.dt, flash: o.flash, asp: o.aspect, iasp: 1 / o.aspect,
    // portrait / narrow screens: zoom out so the scene's objects stay visible
    zoom: Math.max(1, Math.sqrt(1.6 / o.aspect)),
  };
  CONCEPTS.forEach((c, i) => (r[c] = o.concepts[i] ?? 0));
  for (let k = 0; k < FORM_DIM; k++) r["w" + k] = o.form?.[k] ?? 0;
  return r;
}
