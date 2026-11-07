"""Train both networks with any-precision quantisation-aware training.

    python3 train.py --steps-ann 3000 --steps-snn 2500 [--extra data.jsonl]

Phase 1 trains in fp32. Phase 2 (last 35% of steps) samples a random weight
precision per step from {4, 8, fp32} and fake-quantises every quantisable
tensor with a straight-through estimator, so a single set of master weights
works well at every precision tier selected at runtime.
"""
import argparse
import os
import random
import time

import numpy as np

from autograd import Adam, bce_logits, mse, tanh
from dataset import make_sample, load_extra
from models import (ann_init, ann_encode, ann_forward, ANN_QUANT, snn_init, snn_encode,
                    snn_forward, SNN_QUANT, SNN_TRAIN, NUM_CONCEPTS)

HERE = os.path.dirname(os.path.abspath(__file__))
CKPT = os.path.join(HERE, "checkpoints")
ANN_WINDOWS = [16, 24, 32, 48]
SNN_STEPS = [2, 4, 6, 8]
SNN_WORDS = 20
SNN_LR = 1.5e-3


def batch(rng, n, extra):
    texts, tg, va = [], [], []
    for _ in range(n):
        t, c, v, a = make_sample(rng, extra)
        texts.append(t); tg.append(c); va.append([v, a])
    return texts, np.asarray(tg, np.float32), np.asarray(va, np.float32)


def loss_fn(out, tg, va):
    return bce_logits(out[:, :NUM_CONCEPTS], tg) + mse(tanh(out[:, NUM_CONCEPTS:]), va) * 0.5


def random_bits(rng, names, qat):
    if not qat:
        return None
    b = rng.choice([4, 8, None])
    return {n: b for n in names}


def lr_at(step, total, base):
    warm = 100
    if step < warm:
        return base * (step + 1) / warm
    return base * 0.5 * (1 + np.cos(np.pi * (step - warm) / max(1, total - warm))) + base * 0.02


def train_ann(steps, extra, seed=0, bs=64):
    rng = random.Random(seed)
    p = ann_init(seed)
    opt = Adam(list(p.values()), lr=2e-3, wd=1e-5)
    t0 = time.time()
    for s in range(steps):
        texts, tg, va = batch(rng, bs, extra)
        U, B, M = ann_encode(texts, rng.choice(ANN_WINDOWS))
        bits = random_bits(rng, ANN_QUANT, s > 0.65 * steps)
        loss = loss_fn(ann_forward(p, U, B, M, bits), tg, va)
        loss.backward()
        opt.lr = lr_at(s, steps, 2e-3)
        opt.step()
        if s % 100 == 0:
            print(f"[ann] step {s:5d} loss {float(loss.data):.4f}  {time.time() - t0:.0f}s", flush=True)
    return p


def teacher_targets(teacher, texts, words, rng, per_sample=6):
    """Run the trained Transformer on prefixes of every sample (the last word
    plus `per_sample` random positions): soft concept probabilities + affect
    used as per-word distillation targets."""
    from common import tokenize
    prefixes, index = [], []
    for i, t in enumerate(texts):
        toks = tokenize(t)
        vis = toks[-words:]
        off = words - len(vis)
        start = len(toks) - len(vis)
        pos = set(rng.sample(range(len(vis)), min(per_sample, len(vis)))) | {len(vis) - 1}
        for j in sorted(pos):
            prefixes.append(" ".join(toks[: start + j + 1]))
            index.append((i, off + j))
    U, B, M = ann_encode(prefixes, 32)
    out = ann_forward(teacher, U, B, M).data
    P = np.zeros((len(texts), words, NUM_CONCEPTS), np.float32)
    A = np.zeros((len(texts), words, 2), np.float32)
    mask = np.zeros((len(texts), words), np.float32)
    for k, (i, w) in enumerate(index):
        P[i, w] = 1 / (1 + np.exp(-out[k, :NUM_CONCEPTS]))
        A[i, w] = np.tanh(out[k, NUM_CONCEPTS:])
        mask[i, w] = 1
    return P, A, mask


def train_snn(steps, extra, seed=1, bs=48, teacher=None, distill=1.0):
    """Surrogate-gradient BPTT. With a teacher (the trained Transformer) the
    SNN also gets a soft target at *every word*, not only at the end of the
    sample: dense supervision that makes sparse spiking nets train far better
    and teaches the streaming behaviour the app relies on."""
    rng = random.Random(seed + 100)
    p = snn_init(seed)
    opt = Adam([p[n] for n in SNN_TRAIN], lr=SNN_LR)
    t0 = time.time()
    for s in range(steps):
        texts, tg, va = batch(rng, bs, extra)
        X = snn_encode(texts, SNN_WORDS)
        T = rng.choice(SNN_STEPS)
        bits = random_bits(rng, SNN_QUANT, s > 0.65 * steps)
        word_loss = None
        if teacher is not None:
            P, A, mask = teacher_targets(teacher, texts, SNN_WORDS, rng)
            n_words = max(1.0, float(mask.sum()) / bs)

            def word_loss(w, out):
                m = mask[:, w:w + 1]
                if not m.any():
                    return None
                l = bce_logits(out[:, :NUM_CONCEPTS], P[:, w], m) + mse(tanh(out[:, NUM_CONCEPTS:]), A[:, w], m) * 0.5
                return l * (distill / n_words)
        res = snn_forward(p, X, T, bits, return_rates=True, word_loss=word_loss)
        out, r1, r2 = res[:3]
        task = loss_fn(out, tg, va)
        # task loss + firing-rate regulariser (sparsity = energy efficiency)
        loss = task + (r1 * r1 + r2 * r2) * 2.0
        if teacher is not None and res[3] is not None:
            loss = loss + res[3]
        loss.backward()
        opt.lr = lr_at(s, steps, SNN_LR)
        opt.step()
        if s % 50 == 0:
            print(f"[snn] step {s:5d} T={T} task {float(task.data):.4f} total {float(loss.data):.4f} "
                  f"rates {float(r1.data):.3f}/{float(r2.data):.3f}  {time.time() - t0:.0f}s", flush=True)
    return p


def save(p, name):
    os.makedirs(CKPT, exist_ok=True)
    np.savez(os.path.join(CKPT, name + ".npz"), **{k: v.data for k, v in p.items()})


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps-ann", type=int, default=3000)
    ap.add_argument("--steps-snn", type=int, default=2500)
    ap.add_argument("--extra", default=None)
    ap.add_argument("--only", choices=["ann", "snn"], default=None)
    a = ap.parse_args()
    extra = load_extra(a.extra) if a.extra else None
    if a.only in (None, "ann"):
        save(train_ann(a.steps_ann, extra), "ann")
    if a.only in (None, "snn"):
        z = np.load(os.path.join(CKPT, "ann.npz"))
        from autograd import T as Tensor
        teacher = {k: Tensor(z[k]) for k in z.files}
        save(train_snn(a.steps_snn, extra, teacher=teacher), "snn")
