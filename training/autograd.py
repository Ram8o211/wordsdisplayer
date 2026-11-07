"""Minimal reverse-mode autodiff on NumPy (float32), just enough to train the
two networks of this project without external deep-learning frameworks."""
import numpy as np

F = np.float32


def _unbroadcast(g, shape):
    while g.ndim > len(shape):
        g = g.sum(0)
    for i, s in enumerate(shape):
        if s == 1 and g.shape[i] != 1:
            g = g.sum(i, keepdims=True)
    return g


class T:
    __slots__ = ("data", "grad", "_parents", "_back", "requires_grad")

    def __init__(self, data, parents=(), back=None, requires_grad=False):
        self.data = data if isinstance(data, np.ndarray) else np.asarray(data, F)
        self.grad = None
        self._parents = parents
        self._back = back
        self.requires_grad = requires_grad or any(p.requires_grad for p in parents)

    @property
    def shape(self):
        return self.data.shape

    # ---- graph
    def backward(self, grad=None):
        order, seen = [], set()
        stack = [(self, False)]
        while stack:
            node, done = stack.pop()
            if done:
                order.append(node)
                continue
            if id(node) in seen or not node.requires_grad:
                continue
            seen.add(id(node))
            stack.append((node, True))
            for p in node._parents:
                stack.append((p, False))
        self.grad = np.ones_like(self.data) if grad is None else grad
        for node in reversed(order):
            if node._back is not None and node.grad is not None:
                node._back(node.grad)
                node.grad = None  # free intermediate memory (leaves have no _back)

    def _acc(self, g):
        if not self.requires_grad:
            return
        self.grad = g if self.grad is None else self.grad + g

    # ---- arithmetic
    def __add__(self, o):
        o = o if isinstance(o, T) else T(np.asarray(o, F))
        out = T(self.data + o.data, (self, o))
        out._back = lambda g: (self._acc(_unbroadcast(g, self.shape)), o._acc(_unbroadcast(g, o.shape)))
        return out

    __radd__ = __add__

    def __sub__(self, o):
        return self + (o * -1.0 if isinstance(o, T) else -o)

    def __mul__(self, o):
        o = o if isinstance(o, T) else T(np.asarray(o, F))
        out = T(self.data * o.data, (self, o))
        out._back = lambda g: (self._acc(_unbroadcast(g * o.data, self.shape)),
                               o._acc(_unbroadcast(g * self.data, o.shape)))
        return out

    __rmul__ = __mul__

    def __matmul__(self, o):
        out = T(self.data @ o.data, (self, o))

        def back(g):
            if self.requires_grad:
                self._acc(_unbroadcast(g @ np.swapaxes(o.data, -1, -2), self.shape))
            if o.requires_grad:
                o._acc(_unbroadcast(np.swapaxes(self.data, -1, -2) @ g, o.shape))
        out._back = back
        return out

    def reshape(self, *shape):
        out = T(self.data.reshape(*shape), (self,))
        out._back = lambda g: self._acc(g.reshape(self.shape))
        return out

    def transpose(self, *axes):
        out = T(self.data.transpose(*axes), (self,))
        inv = np.argsort(axes)
        out._back = lambda g: self._acc(g.transpose(*inv))
        return out

    def sum(self, axis=None, keepdims=False):
        out = T(self.data.sum(axis=axis, keepdims=keepdims), (self,))

        def back(g):
            if axis is not None and not keepdims:
                g = np.expand_dims(g, axis)
            self._acc(np.broadcast_to(g, self.shape).astype(F))
        out._back = back
        return out

    def __getitem__(self, idx):
        out = T(self.data[idx], (self,))

        def back(g):
            z = np.zeros_like(self.data)
            np.add.at(z, idx, g)
            self._acc(z)
        out._back = back
        return out


def _is_leaf(n):
    return not n._parents


def param(shape, scale=None, rng=None, init=None):
    rng = rng or np.random.default_rng(0)
    if init is not None:
        d = np.asarray(init, F)
    else:
        scale = scale if scale is not None else 1.0 / np.sqrt(shape[0])
        d = (rng.standard_normal(shape) * scale).astype(F)
    return T(d, requires_grad=True)


def const(x):
    return T(np.asarray(x, F))


# ---- functions
def embed(table: T, ids: np.ndarray):
    out = T(table.data[ids], (table,))

    def back(g):
        z = np.zeros_like(table.data)
        np.add.at(z, ids.reshape(-1), g.reshape(-1, table.shape[1]))
        table._acc(z)
    out._back = back
    return out


def gelu(x: T):
    d = x.data
    c = np.sqrt(2 / np.pi).astype(F)
    u = c * (d + 0.044715 * d ** 3)
    th = np.tanh(u)
    out = T(0.5 * d * (1 + th), (x,))

    def back(g):
        du = c * (1 + 3 * 0.044715 * d ** 2)
        x._acc(g * (0.5 * (1 + th) + 0.5 * d * (1 - th ** 2) * du))
    out._back = back
    return out


def tanh(x: T):
    y = np.tanh(x.data)
    out = T(y, (x,))
    out._back = lambda g: x._acc(g * (1 - y * y))
    return out


def softmax(x: T, mask_add=None):
    d = x.data if mask_add is None else x.data + mask_add
    d = d - d.max(-1, keepdims=True)
    e = np.exp(d)
    y = e / e.sum(-1, keepdims=True)
    out = T(y, (x,))
    out._back = lambda g: x._acc(y * (g - (g * y).sum(-1, keepdims=True)))
    return out


def layernorm(x: T, gamma: T, beta: T, eps=1e-5):
    mu = x.data.mean(-1, keepdims=True)
    xc = x.data - mu
    var = (xc ** 2).mean(-1, keepdims=True)
    inv = 1.0 / np.sqrt(var + eps)
    xh = xc * inv
    out = T(xh * gamma.data + beta.data, (x, gamma, beta))

    def back(g):
        n = x.shape[-1]
        gamma._acc(_unbroadcast(g * xh, gamma.shape))
        beta._acc(_unbroadcast(g, beta.shape))
        gx = g * gamma.data
        x._acc(inv / n * (n * gx - gx.sum(-1, keepdims=True) - xh * (gx * xh).sum(-1, keepdims=True)))
    out._back = back
    return out


def spike(v: T, gamma=0.3, slope=5.0):
    """Heaviside forward, fast-sigmoid surrogate gradient backward."""
    z = (v.data > 0).astype(F)
    out = T(z, (v,))
    out._back = lambda g: v._acc(g * gamma / (1 + slope * np.abs(v.data)) ** 2)
    return out


def detach(x: T):
    return T(x.data)


def fake_quant(w: T, bits, group=32, axis=0):
    """Symmetric fake-quantisation with straight-through estimator.
    Groups run along `axis` of `w` (the input dimension of a layer)."""
    if bits is None or bits >= 16:
        return w
    out = T(quantize_dequantize(w.data, bits, group, axis), (w,))
    out._back = lambda g: w._acc(g)
    return out


def quantize_dequantize(d, bits, group=32, axis=0):
    q, s = quantize(np.moveaxis(d, axis, -1), bits, group)
    return np.moveaxis(dequantize(q, s, group), -1, axis).astype(F)


def quantize(rows: np.ndarray, bits: int, group: int):
    """Quantise a 2-D [R, C] array along C. int8: one scale per row,
    int4: one scale per `group` columns. Returns (q int8 array, scales f32)."""
    R, C = rows.shape
    qmax = 2 ** (bits - 1) - 1
    g = C if bits == 8 else group
    ng = (C + g - 1) // g
    pad = ng * g - C
    r = np.pad(rows, ((0, 0), (0, pad))).reshape(R, ng, g)
    amax = np.abs(r).max(-1)
    s = np.where(amax > 0, amax / qmax, 1.0).astype(F)
    q = np.clip(np.round(r / s[..., None]), -qmax, qmax).astype(np.int8)
    return q.reshape(R, ng * g)[:, :C], s


def dequantize(q, s, group):
    R, C = q.shape
    g = C if s.shape[1] == 1 else group
    ng = s.shape[1]
    qp = np.pad(q.astype(F), ((0, 0), (0, ng * g - C))).reshape(R, ng, g)
    return (qp * s[..., None]).reshape(R, ng * g)[:, :C]


def bce_logits(logits: T, target: np.ndarray, weight=None):
    """Mean binary cross-entropy; optional per-row weights [B, 1]."""
    x = logits.data
    p = 1 / (1 + np.exp(-x))
    w = np.ones((x.shape[0], 1), F) if weight is None else weight
    loss = ((np.maximum(x, 0) - x * target + np.log1p(np.exp(-np.abs(x)))) * w).mean()
    out = T(np.asarray(loss, F), (logits,))
    out._back = lambda g: logits._acc(g * (p - target) * w / x.size)
    return out


def mse(pred: T, target: np.ndarray, weight=None):
    diff = pred.data - target
    w = np.ones((diff.shape[0], 1), F) if weight is None else weight
    out = T(np.asarray((diff ** 2 * w).mean(), F), (pred,))
    out._back = lambda g: pred._acc(g * 2 * diff * w / diff.size)
    return out


class Adam:
    def __init__(self, params, lr=1e-3, b1=0.9, b2=0.999, eps=1e-8, wd=0.0, clip=1.0):
        self.p = params
        self.lr, self.b1, self.b2, self.eps, self.wd, self.clip = lr, b1, b2, eps, wd, clip
        self.m = [np.zeros_like(p.data) for p in params]
        self.v = [np.zeros_like(p.data) for p in params]
        self.t = 0

    def step(self):
        self.t += 1
        gn = np.sqrt(sum(float((p.grad ** 2).sum()) for p in self.p if p.grad is not None))
        k = min(1.0, self.clip / (gn + 1e-6))
        for p, m, v in zip(self.p, self.m, self.v):
            if p.grad is None:
                continue
            g = p.grad * k
            m *= self.b1
            m += (1 - self.b1) * g
            v *= self.b2
            v += (1 - self.b2) * g * g
            mh = m / (1 - self.b1 ** self.t)
            vh = v / (1 - self.b2 ** self.t)
            p.data -= (self.lr * (mh / (np.sqrt(vh) + self.eps) + self.wd * p.data)).astype(F)
            p.grad = None
        return gn
