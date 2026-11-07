"""Constructs the spiking network's weights directly instead of training them
by back-propagation through time (Neural-Engineering-Framework style).

Only the linear readout is fitted (a convex problem). The rest is assigned:

  * Layer 1 (256 adaptive LIF neurons) is split into populations: 10 neurons
    per visual concept (22 x 10 = 220) and 4 x 9 affect neurons
    (valence+, valence-, arousal+, arousal-).
  * w_in (4096 afferents -> 256), factorised as w_in_a (4096 -> 26
    populations) @ w_in_e (26 -> 256 heterogeneous encoders). w_in_a is the
    minimum-norm ridge solution that maps
    the sparse spike code of every vocabulary word onto the input current its
    populations should receive: a keyword drives its concept population,
    function words and small talk drive nothing.
  * w_rec: each population excites itself (a short attractor memory that
    keeps the topic alive for a few words, ended by threshold adaptation) and
    concept populations inhibit each other (winner-take-most topic switching).
  * w_12: every layer-1 population projects onto its own slower layer-2
    population (5 neurons per concept, 4 per affect pool): longer memory.
  * Readout (w_out, w_out1, b_out): logistic regression (concepts) and least
    squares (affect) on the simulated spike traces, folded into the weights.

The whole construction takes about a minute on a laptop CPU, instead of an
hour of BPTT, and is fully reproducible. The result is saved as
checkpoints/snn.npz in exactly the format train.py writes, so export.py
quantises it like any trained model.
"""
import random
import time

import numpy as np

from common import CONCEPTS, NUM_CONCEPTS, SNN_AFFERENTS, SNN_WORD_SPIKES, tokenize, fnv1a
from dataset import LEXICON, MOODS, make_sample
from models import SNN_CFG, NOUT, snn_init, snn_encode

F = np.float32
POOL = 10                 # layer-1 neurons per concept
AFF0 = NUM_CONCEPTS * POOL  # first affect neuron (220)
APOOL = 9                 # neurons per affect pool
POOL2 = 5                 # layer-2 neurons per concept
AFF2 = NUM_CONCEPTS * POOL2  # 110
APOOL2 = 4
WORDS = 20


def l1_pool(c):
    return slice(c * POOL, (c + 1) * POOL)


def aff_pool(k):  # 0 v+, 1 v-, 2 a+, 3 a-
    return slice(AFF0 + k * APOOL, AFF0 + (k + 1) * APOOL)


def l2_pool(c):
    return slice(c * POOL2, (c + 1) * POOL2)


def aff2_pool(k):
    return slice(AFF2 + k * APOOL2, AFF2 + (k + 1) * APOOL2)


def vocabulary(n=20000, seed=3):
    rng = random.Random(seed)
    vocab = set()
    for _ in range(n):
        vocab.update(tokenize(make_sample(rng)[0]))
    return sorted(vocab)


NPOP = NUM_CONCEPTS + 4  # 22 concept populations + v+, v-, a+, a-


def pop_slice(k):
    return l1_pool(k) if k < NUM_CONCEPTS else aff_pool(k - NUM_CONCEPTS)


def desired_currents(vocab, npop):
    """Target input current of every population for every word."""
    word_concepts = {}
    for c, (kws, _, _) in LEXICON.items():
        for w in tokenize(kws):
            word_concepts.setdefault(w, set()).add(CONCEPTS.index(c))
    Y = np.zeros((len(vocab), npop), F)
    K = NUM_CONCEPTS
    for i, w in enumerate(vocab):
        for c in word_concepts.get(w, ()):
            Y[i, c] = 1.0
        if w in MOODS:
            v, a = MOODS[w]
            Y[i, K] = max(v, 0); Y[i, K + 1] = max(-v, 0)
            Y[i, K + 2] = max(a, 0); Y[i, K + 3] = max(-a, 0)
    return Y


def word_codes(vocab):
    X = np.zeros((len(vocab), SNN_AFFERENTS), F)
    for i, w in enumerate(vocab):
        # unigram afferents only: the bigram spikes vary with context
        for j in range(SNN_WORD_SPIKES):
            X[i, fnv1a("s:" + w + ":" + str(j)) % SNN_AFFERENTS] = 1
    return X


def design(gain_in=4.0, self_exc=0.6, inhib=0.15, gain12=1.2, bias1=-0.2, seed=1):
    p = {k: v.data.copy() for k, v in snn_init(seed).items()}
    n1, n2 = SNN_CFG["n1"], SNN_CFG["n2"]
    vocab = vocabulary()
    X, Y = word_codes(vocab), desired_currents(vocab, NPOP)
    # ridge, dual form: A = X^T (X X^T + lam I)^-1 Y   (afferents -> populations)
    lam = 0.5
    A = X.T @ np.linalg.solve(X @ X.T + lam * np.eye(len(vocab), dtype=F), Y)
    # factorised input weights w_in = w_in_a @ w_in_e: the 4096 x 256 matrix
    # has rank 26 (one column per population), so it is stored as
    # 4096 x 26 "decoders" + 26 x 256 heterogeneous "encoders" (10x smaller,
    # 10x fewer synaptic operations per input spike).
    r = np.random.default_rng(seed + 7)
    E = np.zeros((NPOP, n1), F)
    for k in range(NPOP):
        E[k, pop_slice(k)] = r.uniform(0.8, 1.2, pop_slice(k).stop - pop_slice(k).start)
    p.pop("w_in", None)
    p["w_in_a"] = (gain_in * A).astype(F)
    p["w_in_e"] = E
    p["b1"] = np.full(n1, bias1, F)
    rec = np.zeros((n1, n1), F)
    for c in range(NUM_CONCEPTS):
        s = l1_pool(c)
        rec[s, :AFF0] = -inhib / POOL        # inhibit the other concepts...
        rec[s, s] = self_exc / POOL          # ...excite the own population
    for k in range(4):
        rec[aff_pool(k), aff_pool(k)] = self_exc / APOOL
    p["w_rec"] = rec
    w12 = np.zeros((n1, n2), F)
    for c in range(NUM_CONCEPTS):
        w12[l1_pool(c), l2_pool(c)] = gain12 / POOL * 2
    for k in range(4):
        w12[aff_pool(k), aff2_pool(k)] = gain12 / APOOL * 2
    p["w_12"] = w12
    p["b2"] = np.full(n2, -0.1, F)
    return p


def simulate(p, X, steps):
    """NumPy forward of models.snn_forward without autograd. Returns the mean
    spike traces of both layers over the last word's steps + spike rates."""
    B, W, _ = X.shape
    n1, n2 = p["w_rec"].shape[0], p["w_12"].shape[1]
    a1 = np.exp(-1 / (p["tau_m1"] * steps)); r1 = np.exp(-1 / (p["tau_a1"] * steps))
    a2 = np.exp(-1 / (p["tau_m2"] * steps)); ko = float(np.exp(-1 / (p["tau_out"][0] * steps)))
    I = ((X @ p["w_in_a"]) @ p["w_in_e"] + p["b1"]) * (1 - a1)
    b2 = p["b2"] * (1 - a2)
    v1 = np.zeros((B, n1), F); z1 = np.zeros_like(v1); ad = np.zeros_like(v1)
    v2 = np.zeros((B, n2), F); z2 = np.zeros_like(v2)
    t1 = np.zeros_like(v1); t2 = np.zeros_like(v2)
    f1 = f2 = None
    s1 = s2 = 0.0
    for w in range(W):
        f1 = np.zeros_like(v1); f2 = np.zeros_like(v2)
        for _ in range(steps):
            ad = r1 * ad + z1
            thr = 1 + p["beta1"] * ad
            v1 = v1 * a1 + I[:, w] + z1 @ p["w_rec"] - z1 * thr
            z1 = (v1 - thr > 0).astype(F)
            v2 = v2 * a2 + b2 + z1 @ p["w_12"] - z2
            z2 = (v2 - 1 > 0).astype(F)
            t1 = t1 * ko + z1; t2 = t2 * ko + z2
            f1 += t1; f2 += t2
            s1 += z1.mean(); s2 += z2.mean()
    n = W * steps
    return np.concatenate([f2, f1], 1) / steps, s1 / n, s2 / n


def make_data(n, seed):
    rng = random.Random(seed)
    tx, tg, va = [], [], []
    for _ in range(n):
        t, c, v, a = make_sample(rng)
        tx.append(t); tg.append(c); va.append([v, a])
    return tx, np.asarray(tg, F), np.asarray(va, F)


def features(p, texts, steps_list=(2, 4, 8), chunk=500):
    X = snn_encode(texts, WORDS)
    feats = []
    for st in steps_list:  # readout must work for every T -> fit on all of them
        fs = [simulate(p, X[i:i + chunk], st)[0] for i in range(0, len(texts), chunk)]
        feats.append(np.concatenate(fs))
    return feats


def fit_readout(p, feats, tg, va, iters=600, l2=1e-4):
    """Logistic regression for concepts + ridge for affect, on standardised
    features; standardisation is folded back into raw-trace weights."""
    Fm = np.concatenate(feats)
    T = np.concatenate([tg] * len(feats))
    A = np.concatenate([va] * len(feats))
    mu, sd = Fm.mean(0), Fm.std(0) + 1e-3
    Z = (Fm - mu) / sd
    d = Z.shape[1]
    Wc = np.zeros((d, NUM_CONCEPTS), F); bc = np.zeros(NUM_CONCEPTS, F)
    m = [np.zeros_like(Wc), np.zeros_like(bc)]; v = [np.zeros_like(Wc), np.zeros_like(bc)]
    for it in range(1, iters + 1):  # full-batch Adam
        P = 1 / (1 + np.exp(-(Z @ Wc + bc)))
        G = (P - T) / len(Z)
        grads = [Z.T @ G + l2 * Wc, G.sum(0)]
        for k, (prm, g) in enumerate(zip((Wc, bc), grads)):
            m[k] = 0.9 * m[k] + 0.1 * g; v[k] = 0.999 * v[k] + 0.001 * g * g
            prm -= 0.05 * (m[k] / (1 - 0.9 ** it)) / (np.sqrt(v[k] / (1 - 0.999 ** it)) + 1e-8)
    At = np.arctanh(np.clip(A, -0.95, 0.95))
    Zb = np.concatenate([Z, np.ones((len(Z), 1), F)], 1)
    Wa = np.linalg.solve(Zb.T @ Zb + 1.0 * np.eye(d + 1), Zb.T @ At)
    Wall = np.concatenate([Wc, Wa[:-1]], 1) / sd[:, None]
    ball = np.concatenate([bc, Wa[-1]]) - mu @ (np.concatenate([Wc, Wa[:-1]], 1) / sd[:, None])
    n2 = p["w_12"].shape[1]
    p["w_out"] = Wall[:n2].astype(F)
    p["w_out1"] = Wall[n2:].astype(F)
    p["b_out"] = ball.astype(F)
    return p


def evaluate(p, data, steps=4):
    tx, tg, va = data
    f, r1, r2 = simulate(p, snn_encode(tx, WORDS), steps)
    out = f @ np.concatenate([p["w_out"], p["w_out1"]]) + p["b_out"]
    pr = 1 / (1 + np.exp(-out[:, :NUM_CONCEPTS]))
    gold, pred = tg >= 0.5, pr >= 0.5
    tp = (gold & pred).sum(); fp = (~gold & pred).sum(); fn = (gold & ~pred).sum()
    has = gold.any(1)
    top1 = gold[np.arange(len(pr)), pr.argmax(1)][has].mean()
    mae = np.abs(np.tanh(out[:, NUM_CONCEPTS:]) - va).mean()
    return dict(f1=round(float(2 * tp / (2 * tp + fp + fn)), 4), top1=round(float(top1), 4),
                affect_mae=round(float(mae), 4), rates=(round(float(r1), 4), round(float(r2), 4)))


def build(cfg, train_n=3000, quiet=False):
    p = design(**cfg)
    tr = make_data(train_n, 11)
    p = fit_readout(p, features(p, tr[0]), tr[1], tr[2])
    return p


if __name__ == "__main__":
    import argparse
    import os
    ap = argparse.ArgumentParser()
    ap.add_argument("--search", action="store_true", help="grid-search the dynamics constants")
    a = ap.parse_args()
    val = make_data(800, 99)
    best = dict(gain_in=4.0, self_exc=0.6, inhib=0.15, gain12=1.2, bias1=-0.2)
    if a.search:
        best_f1 = -1
        for gi in (1.5, 2.5, 4.0):
            for se in (0.0, 0.6, 1.2):
                for ih in (0.0, 0.15, 0.4):
                    cfg = dict(best, gain_in=gi, self_exc=se, inhib=ih)
                    t0 = time.time()
                    r = evaluate(build(cfg, 1500), val)
                    print(cfg, r, f"{time.time() - t0:.0f}s", flush=True)
                    if r["f1"] > best_f1:
                        best_f1, best_cfg = r["f1"], cfg
        best = best_cfg
        print("best", best)
    p = build(best)
    for st in (2, 4, 6, 8):
        print("T", st, evaluate(p, val, st))
    os.makedirs("checkpoints", exist_ok=True)
    np.savez("checkpoints/snn.npz", **p)
