"""The two networks.

ANN  ("traditional"): hashed uni+bigram embeddings -> 2-layer pre-LN
     Transformer encoder (d=64, 4 heads) with recency positions ->
     mean+last pooling -> MLP head -> 22 concept logits + valence/arousal.

SNN  ("neuromorphic"): sparse binary word codes (4096 afferents) ->
     recurrent adaptive-LIF layer (256) -> LIF layer (128) ->
     leaky-integrator readout (22 + 2). Dynamics are defined in "word time"
     and discretised with T steps per word, so the same weights run with
     T=2 (phone) ... T=8 (workstation); more steps = finer integration.
"""
import numpy as np

from autograd import (F, T, param, embed, gelu, tanh, softmax, layernorm, spike, detach,
                      fake_quant, const)
from common import (ANN_BUCKETS, ANN_MAX_LEN, NUM_CONCEPTS, NUM_AFFECT, SNN_AFFERENTS,
                    tokenize, ann_ids, snn_afferents)

NOUT = NUM_CONCEPTS + NUM_AFFECT

# --------------------------------------------------------------------- ANN
ANN_CFG = dict(d=64, heads=4, ff=192, layers=2)


def ann_init(seed=0):
    r = np.random.default_rng(seed)
    d, ff = ANN_CFG["d"], ANN_CFG["ff"]
    p = {"emb": param((ANN_BUCKETS, d), 0.1, r), "pos": param((ANN_MAX_LEN, d), 0.02, r)}
    for i in range(ANN_CFG["layers"]):
        p.update({
            f"b{i}.ln1.g": param((d,), init=np.ones(d)), f"b{i}.ln1.b": param((d,), init=np.zeros(d)),
            f"b{i}.qkv": param((d, 3 * d), None, r), f"b{i}.qkv.b": param((3 * d,), init=np.zeros(3 * d)),
            f"b{i}.o": param((d, d), 0.5 / np.sqrt(d), r), f"b{i}.o.b": param((d,), init=np.zeros(d)),
            f"b{i}.ln2.g": param((d,), init=np.ones(d)), f"b{i}.ln2.b": param((d,), init=np.zeros(d)),
            f"b{i}.f1": param((d, ff), None, r), f"b{i}.f1.b": param((ff,), init=np.zeros(ff)),
            f"b{i}.f2": param((ff, d), 0.5 / np.sqrt(ff), r), f"b{i}.f2.b": param((d,), init=np.zeros(d)),
        })
    p.update({
        "lnf.g": param((d,), init=np.ones(d)), "lnf.b": param((d,), init=np.zeros(d)),
        "h1m": param((d, d), None, r), "h1l": param((d, d), None, r), "h1.b": param((d,), init=np.zeros(d)),
        "h2": param((d, NOUT), None, r), "h2.b": param((NOUT,), init=np.zeros(NOUT)),
    })
    return p


# tensors that get quantised (the rest - norms, biases, positions - stay fp32)
ANN_QUANT = ["emb"] + [f"b{i}.{n}" for i in range(ANN_CFG["layers"]) for n in ("qkv", "o", "f1", "f2")] + \
            ["h1m", "h1l", "h2"]


def _w(p, name, bits):
    b = bits.get(name) if bits else None
    # Linear weights are [in, out]; quantisation groups run along `in`
    # (exported transposed as [out, in] rows). Embedding rows: along d.
    return fake_quant(p[name], b, axis=1 if name == "emb" else 0)


def ann_encode(texts, window):
    """Left-padded id matrices for a batch of texts, keeping the last `window` tokens."""
    U = np.zeros((len(texts), window), np.int64)
    Bg = np.zeros_like(U)
    M = np.zeros((len(texts), window), F)
    for i, t in enumerate(texts):
        u, b = ann_ids(tokenize(t))
        u, b = u[-window:], b[-window:]
        n = len(u)
        if n:
            U[i, window - n:], Bg[i, window - n:], M[i, window - n:] = u, b, 1
    return U, Bg, M


def ann_forward(p, U, Bg, M, bits=None):
    Bn, L = U.shape
    d, h = ANN_CFG["d"], ANN_CFG["heads"]
    dh = d // h
    emb = _w(p, "emb", bits)
    pos_idx = np.arange(L - 1, -1, -1)  # position 0 = most recent token
    x = embed(emb, U) + embed(emb, Bg) + embed(p["pos"], pos_idx)
    key_mask = ((1 - M) * -1e9)[:, None, None, :].astype(F)
    for i in range(ANN_CFG["layers"]):
        y = layernorm(x, p[f"b{i}.ln1.g"], p[f"b{i}.ln1.b"])
        qkv = (y @ _w(p, f"b{i}.qkv", bits) + p[f"b{i}.qkv.b"]).reshape(Bn, L, 3, h, dh)
        qkv = qkv.transpose(2, 0, 3, 1, 4)  # 3, B, h, L, dh
        q, k, v = qkv[0], qkv[1], qkv[2]
        att = softmax((q @ k.transpose(0, 1, 3, 2)) * (1.0 / np.sqrt(dh)), key_mask)
        o = (att @ v).transpose(0, 2, 1, 3).reshape(Bn, L, d)
        x = x + (o @ _w(p, f"b{i}.o", bits) + p[f"b{i}.o.b"])
        y = layernorm(x, p[f"b{i}.ln2.g"], p[f"b{i}.ln2.b"])
        y = gelu(y @ _w(p, f"b{i}.f1", bits) + p[f"b{i}.f1.b"])
        x = x + (y @ _w(p, f"b{i}.f2", bits) + p[f"b{i}.f2.b"])
    x = layernorm(x, p["lnf.g"], p["lnf.b"])
    cnt = np.maximum(M.sum(1, keepdims=True), 1)
    mean = (x * const((M / cnt)[..., None])).sum(1)
    last = x[:, L - 1]
    hdn = gelu(mean @ _w(p, "h1m", bits) + last @ _w(p, "h1l", bits) + p["h1.b"])
    return hdn @ _w(p, "h2", bits) + p["h2.b"]


# --------------------------------------------------------------------- SNN
SNN_CFG = dict(n1=256, n2=128)


def snn_init(seed=1):
    r = np.random.default_rng(seed)
    n1, n2 = SNN_CFG["n1"], SNN_CFG["n2"]
    # heterogeneous, fixed time constants (in words) -> multi-timescale memory
    tau_m1 = np.exp(r.uniform(np.log(0.3), np.log(3.0), n1))
    tau_a1 = np.exp(r.uniform(np.log(2.0), np.log(20.0), n1))
    tau_m2 = np.exp(r.uniform(np.log(0.5), np.log(8.0), n2))
    return {
        "w_in": param((SNN_AFFERENTS, n1), 0.45, r), "b1": param((n1,), init=np.zeros(n1)),
        "w_rec": param((n1, n1), 0.4 / np.sqrt(n1), r),
        "w_12": param((n1, n2), 1.2 / np.sqrt(n1), r), "b2": param((n2,), init=np.zeros(n2)),
        "w_out": param((n2, NOUT), 1.0 / np.sqrt(n2), r), "b_out": param((NOUT,), init=np.zeros(NOUT)),
        # skip readout from the recurrent layer (shorter credit-assignment path)
        "w_out1": param((n1, NOUT), 1.0 / np.sqrt(n1), r),
        # non-trainable neuron constants (kept as params for export convenience)
        "tau_m1": const(tau_m1), "tau_a1": const(tau_a1), "tau_m2": const(tau_m2),
        "beta1": const(np.full(n1, 0.2)),  # threshold adaptation strength
        "tau_out": const(np.full(NOUT, 3.0)),
    }


# input current scaling per step; (1 - a) makes the steady state independent of T
INPUT_GAIN = lambda a, steps: 1 - a
SURR = dict(gamma=0.5, slope=2.0)  # surrogate gradient shape

SNN_QUANT = ["w_in", "w_rec", "w_12", "w_out", "w_out1"]
SNN_TRAIN = ["w_in", "b1", "w_rec", "w_12", "b2", "w_out", "w_out1", "b_out"]


def snn_encode(texts, max_words):
    X = np.zeros((len(texts), max_words, SNN_AFFERENTS), F)
    for i, t in enumerate(texts):
        toks = tokenize(t)[-max_words:]
        off = max_words - len(toks)
        prev = "<s>"
        for j, tok in enumerate(toks):
            X[i, off + j, snn_afferents(prev, tok)] = 1
            prev = tok
    return X


def snn_forward(p, X, steps, bits=None, return_rates=False, word_loss=None):
    """X: [B, W, afferents] binary word codes.

    The readout of a word is the mean of the output integrators over that
    word's T steps. Returns the readout of the last word (concept logits,
    pre-tanh affect); with `word_loss(w, readout)` the per-word losses
    (used for teacher distillation at every word) are summed and returned too.
    """
    Bn, W, _ = X.shape
    n1, n2 = SNN_CFG["n1"], SNN_CFG["n2"]
    q = (lambda n: fake_quant(p[n], bits.get(n) if bits else None, axis=1))
    w_in, w_rec, w_12, w_out, w_out1 = q("w_in"), q("w_rec"), q("w_12"), q("w_out"), q("w_out1")
    a1 = np.exp(-1.0 / (p["tau_m1"].data * steps)).astype(F)
    r1 = np.exp(-1.0 / (p["tau_a1"].data * steps)).astype(F)
    a2 = np.exp(-1.0 / (p["tau_m2"].data * steps)).astype(F)
    ko = np.exp(-1.0 / (p["tau_out"].data * steps)).astype(F)
    beta = p["beta1"].data
    # word-level input currents, constant during the T steps of each word
    I_in = (const(X) @ w_in + p["b1"]) * const(INPUT_GAIN(a1, steps))
    b2 = p["b2"] * const(1 - a2)
    v1 = const(np.zeros((Bn, n1))); z1 = const(np.zeros((Bn, n1))); ad = np.zeros((Bn, n1), F)
    v2 = const(np.zeros((Bn, n2))); z2 = const(np.zeros((Bn, n2)))
    y = const(np.zeros((Bn, NOUT)))
    s1 = s2 = 0.0
    wl = None
    out = None
    for w in range(W):
        Iw = I_in[:, w]
        acc = None
        for _ in range(steps):
            ad = r1 * ad + z1.data
            thr = 1.0 + beta * ad
            v1 = v1 * const(a1) + Iw + z1 @ w_rec - detach(z1) * const(thr)
            z1 = spike(v1 - const(thr), **SURR)
            v2 = v2 * const(a2) + b2 + z1 @ w_12 - detach(z2)
            z2 = spike(v2 - 1.0, **SURR)
            y = y * const(ko) + z2 @ w_out + z1 @ w_out1
            acc = y if acc is None else acc + y
            s1 = s1 + z1.sum()
            s2 = s2 + z2.sum()
        if word_loss is not None or w == W - 1:
            out = acc * (1.0 / steps) + p["b_out"]
            if word_loss is not None:
                l = word_loss(w, out)
                if l is not None:
                    wl = l if wl is None else wl + l
    res = [out]
    if return_rates:
        nsteps = W * steps * Bn
        res += [s1 * (1.0 / (nsteps * n1)), s2 * (1.0 / (nsteps * n2))]
    if word_loss is not None:
        res.append(wl)
    return res[0] if len(res) == 1 else tuple(res)
