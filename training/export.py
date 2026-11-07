"""Sensitivity-driven mixed-precision quantisation + export for every hardware tier.

For each network:
  1. measure the validation-loss increase caused by quantising each tensor
     alone to 4 / 8 / 16 bits (layer sensitivity);
  2. for each tier, solve a knapsack-like allocation greedily: start from the
     cheapest allowed precision and repeatedly upgrade the tensor with the best
     (loss reduction / extra bytes) ratio until the tier's byte budget is used;
  3. pack the weights (int4 group-wise, int8 per-row, fp16, fp32) into a
     binary blob + JSON manifest, evaluate the *packed* weights and write a
     report and golden vectors for the TypeScript parity test.
"""
import json
import os
import random

import numpy as np

from autograd import F, T, quantize, dequantize, bce_logits, mse, tanh
from common import (tokenize, ann_ids, snn_afferents, CONCEPTS, NUM_CONCEPTS, ANN_BUCKETS, ANN_MAX_LEN, SNN_AFFERENTS,
                    SNN_WORD_SPIKES, SNN_BIGRAM_SPIKES)
from dataset import make_sample
from models import (ANN_CFG, SNN_CFG, ANN_QUANT, SNN_QUANT, ann_encode, ann_forward,
                    snn_encode, snn_forward)

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "..", "public", "models")
GROUP = 32
SMALL = 32768  # elements

TIERS = {
    # allowed precisions, byte budget (relative), runtime knobs
    "nano":    dict(bits=[4, 8], budget=("int8", 0.55), ann_window=16, snn_steps=2),
    "mobile":  dict(bits=[4, 8, 16], budget=("int8", 1.0), ann_window=24, snn_steps=4),
    "desktop": dict(bits=[8, 16], budget=("fp16", 1.0), ann_window=32, snn_steps=6),
    "ultra":   dict(bits=[32], budget=("fp32", 1.0), ann_window=48, snn_steps=8),
}
SNN_WORDS = 20


def load(name):
    z = np.load(os.path.join(HERE, "checkpoints", name + ".npz"))
    return {k: T(z[k].astype(F)) for k in z.files}


def nbytes(shape, bits):
    R, C = shape
    if bits == 32:
        return R * C * 4
    if bits == 16:
        return R * C * 2
    if bits == 8:
        return R * C + R * 4
    return (R * C + 1) // 2 + R * ((C + GROUP - 1) // GROUP) * 4


def rows_of(name, w, model):
    """Tensor in export layout: rows x cols, quantisation groups along cols."""
    if model == "ann" and name in ANN_QUANT and name != "emb":
        return w.T  # Linear [in,out] -> [out,in]
    return w  # embeddings / position table: rows are looked up


def from_rows(name, r, model):
    return r.T if (model == "ann" and name in ANN_QUANT and name != "emb") else r


def qdq(name, w, bits, model):
    if bits == 32:
        return w
    r = rows_of(name, w, model)
    if bits == 16:
        r = r.astype(np.float16).astype(F)
    else:
        q, s = quantize(r, bits, GROUP)
        r = dequantize(q, s, GROUP)
    return np.ascontiguousarray(from_rows(name, r, model)).astype(F)


def with_bits(p, alloc, model):
    q = dict(p)
    for n, b in alloc.items():
        q[n] = T(qdq(n, p[n].data, b, model))
    return q


# ------------------------------------------------------------------ evaluation
def eval_set(n=800, seed=12345):
    rng = random.Random(seed)
    texts, tg, va = [], [], []
    for _ in range(n):
        t, c, v, a = make_sample(rng)
        texts.append(t); tg.append(c); va.append([v, a])
    return texts, np.asarray(tg, F), np.asarray(va, F)


def predict(model, p, texts, knobs, chunk=200):
    outs = []
    for i in range(0, len(texts), chunk):
        tx = texts[i:i + chunk]
        if model == "ann":
            U, B, M = ann_encode(tx, knobs["ann_window"])
            outs.append(ann_forward(p, U, B, M).data)
        else:
            outs.append(snn_forward(p, snn_encode(tx, SNN_WORDS), knobs["snn_steps"]).data)
    return np.concatenate(outs)


def loss_of(out, tg, va):
    return float((bce_logits(T(out[:, :NUM_CONCEPTS]), tg) + mse(tanh(T(out[:, NUM_CONCEPTS:])), va) * 0.5).data)


def probs(out):
    return 1 / (1 + np.exp(-out[:, :NUM_CONCEPTS])), np.tanh(out[:, NUM_CONCEPTS:])


def fuse(pa, pb):
    """Confidence-weighted fusion (same formula as src/nn/fusion.ts)."""
    ca = np.abs(2 * pa - 1).mean(1, keepdims=True)
    cb = np.abs(2 * pb - 1).mean(1, keepdims=True)
    wa = ca / (ca + cb + 1e-6)
    return wa * pa + (1 - wa) * pb


def metrics(p, aff, tg, va):
    gold = tg >= 0.5
    pred = p >= 0.5
    tp = (gold & pred).sum(); fp = (~gold & pred).sum(); fn = (gold & ~pred).sum()
    f1 = 2 * tp / max(1, 2 * tp + fp + fn)
    has = gold.any(1)
    top1 = gold[np.arange(len(p)), p.argmax(1)][has].mean()
    return dict(micro_f1=round(float(f1), 4), top1=round(float(top1), 4),
                affect_mae=round(float(np.abs(aff - va).mean()), 4))


# ------------------------------------------------------------------ allocation
def sensitivity(model, p, names, data, knobs):
    texts, tg, va = data
    base = loss_of(predict(model, p, texts, knobs), tg, va)
    sens = {}
    for n in names:
        sens[n] = {32: 0.0}
        for b in (4, 8, 16):
            q = with_bits(p, {n: b}, model)
            sens[n][b] = max(0.0, loss_of(predict(model, q, texts, knobs), tg, va) - base)
        print(f"  sensitivity {model}.{n}: " + ", ".join(f"{b}b={sens[n][b]:.5f}" for b in (4, 8, 16)), flush=True)
    return sens


def allocate(model, p, names, sens, tier):
    allowed = sorted(tier["bits"])
    # small tensors (output heads, encoders) cost a few KB: never below int8,
    # they carry the calibrated outputs and are very sensitive to int4
    floor = {n: 8 if p[n].data.size < SMALL and 8 in allowed else allowed[0] for n in names}
    ref, frac = tier["budget"]
    refbits = {"int8": 8, "fp16": 16, "fp32": 32}[ref]
    shape = {n: rows_of(n, p[n].data, model).shape for n in names}
    budget = frac * sum(nbytes(shape[n], refbits) for n in names)
    alloc = dict(floor)
    used = sum(nbytes(shape[n], alloc[n]) for n in names)
    while True:
        best = None
        for n in names:
            higher = [b for b in allowed if b > alloc[n]]
            if not higher:
                continue
            b2 = higher[0]
            cost = nbytes(shape[n], b2) - nbytes(shape[n], alloc[n])
            gain = sens[n][alloc[n]] - sens[n][b2]
            if used + cost <= budget and (best is None or gain / cost > best[0]):
                best = (gain / cost, n, b2, cost)
        if best is None or best[0] <= 0:
            break
        _, n, b2, cost = best
        alloc[n] = b2
        used += cost
    return alloc, used


# ------------------------------------------------------------------ packing
def pack(model, p, alloc, tier_name, knobs):
    blob = bytearray()
    tensors = {}

    def put(b):
        while len(blob) % 4:
            blob.append(0)
        off = len(blob)
        blob.extend(b)
        return off, len(b)

    for name, t in p.items():
        w = t.data
        bits = alloc.get(name, 32)
        if w.ndim == 1:
            off, ln = put(w.astype("<f4").tobytes())
            tensors[name] = dict(shape=[w.shape[0]], dtype="f32", offset=off, length=ln)
            continue
        r = np.ascontiguousarray(rows_of(name, w, model))
        R, C = r.shape
        entry = dict(shape=[R, C])
        if bits == 32:
            entry.update(dtype="f32"); entry["offset"], entry["length"] = put(r.astype("<f4").tobytes())
        elif bits == 16:
            entry.update(dtype="f16"); entry["offset"], entry["length"] = put(r.astype("<f2").tobytes())
        else:
            q, s = quantize(r, bits, GROUP)
            if bits == 8:
                data = q.astype(np.int8).tobytes()
                entry.update(dtype="q8", group=C)
            else:
                flat = (q.astype(np.int16).reshape(-1) + 8).astype(np.uint8)
                if flat.size % 2:
                    flat = np.append(flat, np.uint8(8))
                data = (flat[0::2] | (flat[1::2] << 4)).astype(np.uint8).tobytes()
                entry.update(dtype="q4", group=GROUP)
            entry["offset"], entry["length"] = put(data)
            entry["scaleOffset"], entry["scaleLength"] = put(s.astype("<f4").tobytes())
        tensors[name] = entry
    cfg = dict(ANN_CFG) if model == "ann" else dict(SNN_CFG)
    manifest = dict(
        format="wdnn-1", model=model, tier=tier_name, config=cfg, knobs=knobs,
        constants=dict(concepts=CONCEPTS, annBuckets=ANN_BUCKETS, annMaxLen=ANN_MAX_LEN,
                       snnAfferents=SNN_AFFERENTS, snnWordSpikes=SNN_WORD_SPIKES,
                       snnBigramSpikes=SNN_BIGRAM_SPIKES, snnWords=SNN_WORDS),
        tensors=tensors, bytes=len(blob), file=f"{model}-{tier_name}.bin")
    with open(os.path.join(OUT, f"{model}-{tier_name}.bin"), "wb") as f:
        f.write(bytes(blob))
    with open(os.path.join(OUT, f"{model}-{tier_name}.json"), "w") as f:
        json.dump(manifest, f, indent=1)
    return len(blob)


GOLDEN_TEXTS = [
    "we went to the beach and the waves were beautiful",
    "a terrible storm with thunder and lightning over the city",
    "i dream of space and stars and planets tonight",
    "ricordo il bosco e la neve in montagna",
    "please send me the report before the meeting",
]


def main():
    os.makedirs(OUT, exist_ok=True)
    ann, snn = load("ann"), load("snn")
    nets = {"ann": (ann, ANN_QUANT), "snn": (snn, [n for n in SNN_QUANT if n in snn])}
    sens_data = eval_set(400, seed=777)
    test = eval_set(1000, seed=4242)
    texts, tg, va = test
    report = {"tiers": {}, "allocation": {}, "sensitivity": {}}
    golden = {"texts": GOLDEN_TEXTS, "outputs": {}, "hashes": {
        "ann": [list(ann_ids(tokenize(t))) for t in GOLDEN_TEXTS],
        "snn": [[snn_afferents(a, b) for a, b in zip(["<s>"] + tokenize(t), tokenize(t))] for t in GOLDEN_TEXTS]}}
    for model, (p, names) in nets.items():
        sens = sensitivity(model, p, names, sens_data, TIERS["mobile"])
        report["sensitivity"][model] = {n: {str(b): round(v, 6) for b, v in s.items()} for n, s in sens.items()}
    preds = {}
    for tname, tier in TIERS.items():
        knobs = dict(ann_window=tier["ann_window"], snn_steps=tier["snn_steps"])
        report["tiers"][tname] = {"knobs": knobs}
        for model, (p, names) in nets.items():
            sens = {n: {int(b): v for b, v in s.items()} for n, s in report["sensitivity"][model].items()}
            alloc, _ = allocate(model, p, names, sens, tier)
            size = pack(model, p, alloc, tname, knobs)
            q = with_bits(p, alloc, model)
            out = predict(model, q, texts, knobs)
            pr, aff = probs(out)
            preds[(tname, model)] = (pr, aff)
            report["allocation"].setdefault(tname, {})[model] = alloc
            report["tiers"][tname][model] = dict(bytes=size, **metrics(pr, aff, tg, va))
            g = predict(model, q, GOLDEN_TEXTS, knobs)
            golden["outputs"][f"{model}-{tname}"] = np.round(g, 5).tolist()
            print(tname, model, alloc, report["tiers"][tname][model], flush=True)
        pa, aa = preds[(tname, "ann")]
        ps, as_ = preds[(tname, "snn")]
        report["tiers"][tname]["hybrid"] = metrics(fuse(pa, ps), (aa + as_) / 2, tg, va)
        print(tname, "hybrid", report["tiers"][tname]["hybrid"], flush=True)
    with open(os.path.join(OUT, "report.json"), "w") as f:
        json.dump(report, f, indent=1)
    with open(os.path.join(HERE, "..", "tests", "golden.json"), "w") as f:
        json.dump(golden, f)


if __name__ == "__main__":
    main()
