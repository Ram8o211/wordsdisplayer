// Traditional network: 2-layer pre-LN Transformer encoder over the most
// recent `window` words (port of training/models.py::ann_forward).
import { annIds, tokenize, NUM_OUT, ANN_MAX_LEN } from "./text";
import type { Manifest, Weights } from "./qtensor";

export interface NetOutput {
  logits: Float32Array; // NUM_OUT values: concept logits + (valence, arousal) pre-tanh
}

function layerNorm(x: Float32Array, off: number, d: number, g: Float32Array, b: Float32Array, out: Float32Array, oOff: number) {
  let mu = 0;
  for (let i = 0; i < d; i++) mu += x[off + i];
  mu /= d;
  let v = 0;
  for (let i = 0; i < d; i++) v += (x[off + i] - mu) ** 2;
  const inv = 1 / Math.sqrt(v / d + 1e-5);
  for (let i = 0; i < d; i++) out[oOff + i] = (x[off + i] - mu) * inv * g[i] + b[i];
}

const C_GELU = Math.sqrt(2 / Math.PI);
const gelu = (x: number) => 0.5 * x * (1 + Math.tanh(C_GELU * (x + 0.044715 * x * x * x)));

export class TransformerNet {
  readonly d: number;
  readonly heads: number;
  readonly ff: number;
  readonly layers: number;
  window: number;
  // work buffers sized for the longest window, reused on every call
  private bx: Float32Array; private by: Float32Array; private bqkv: Float32Array; private bo: Float32Array;
  private batt: Float32Array; private btmp: Float32Array; private bhid: Float32Array;
  private bmean: Float32Array; private bh1: Float32Array; private bh1l: Float32Array;

  constructor(private w: Weights, m: Manifest) {
    this.d = m.config.d;
    this.heads = m.config.heads;
    this.ff = m.config.ff;
    this.layers = m.config.layers;
    this.window = m.knobs.ann_window;
    const L = ANN_MAX_LEN, d = this.d;
    this.bx = new Float32Array(L * d); this.by = new Float32Array(L * d); this.bo = new Float32Array(L * d);
    this.bqkv = new Float32Array(L * 3 * d); this.batt = new Float32Array(L);
    this.btmp = new Float32Array(d); this.bhid = new Float32Array(this.ff);
    this.bmean = new Float32Array(d); this.bh1 = new Float32Array(d); this.bh1l = new Float32Array(d);
  }

  /** Run on raw text (only the last `window` tokens are used). */
  runText(text: string): NetOutput {
    return this.runTokens(tokenize(text));
  }

  runTokens(allTokens: string[]): NetOutput {
    const tokens = allTokens.slice(-Math.min(this.window, ANN_MAX_LEN));
    const L = tokens.length;
    const { d, heads, ff } = this;
    const dh = d / heads;
    const W = this.w;
    const logits = new Float32Array(NUM_OUT);
    if (L === 0) {
      // empty context -> just the head biases are meaningless; return "nothing"
      logits.fill(-6);
      logits[NUM_OUT - 2] = 0;
      logits[NUM_OUT - 1] = 0;
      return { logits };
    }
    // the bigram of the first visible token must see its real predecessor
    const ids = annIds(allTokens.slice(-(L + 1)));
    const off = ids.uni.length - L;
    const x = this.bx; x.fill(0, 0, L * d);
    const pos = W["pos"];
    for (let j = 0; j < L; j++) {
      W["emb"].addRow(ids.uni[off + j], x, 1, j * d);
      W["emb"].addRow(ids.bi[off + j], x, 1, j * d);
      pos.addRow(L - 1 - j, x, 1, j * d);
    }
    const y = this.by, qkv = this.bqkv, att = this.batt, o = this.bo, tmp = this.btmp, hid = this.bhid;
    const scale = 1 / Math.sqrt(dh);
    for (let i = 0; i < this.layers; i++) {
      const p = `b${i}.`;
      const g1 = W[p + "ln1.g"].vector(), b1 = W[p + "ln1.b"].vector();
      for (let j = 0; j < L; j++) layerNorm(x, j * d, d, g1, b1, y, j * d);
      const wqkv = W[p + "qkv"], bqkv = W[p + "qkv.b"].vector();
      for (let j = 0; j < L; j++) wqkv.matvec(y, qkv, bqkv, j * d, j * 3 * d);
      // multi-head attention; qkv row layout: [3][heads][dh]
      for (let h = 0; h < heads; h++) {
        for (let qi = 0; qi < L; qi++) {
          let mx = -Infinity;
          for (let ki = 0; ki < L; ki++) {
            let s = 0;
            const qo = qi * 3 * d + h * dh, ko = ki * 3 * d + d + h * dh;
            for (let c = 0; c < dh; c++) s += qkv[qo + c] * qkv[ko + c];
            att[ki] = s * scale;
            if (att[ki] > mx) mx = att[ki];
          }
          let sum = 0;
          for (let ki = 0; ki < L; ki++) sum += (att[ki] = Math.exp(att[ki] - mx));
          for (let c = 0; c < dh; c++) {
            let acc = 0;
            for (let ki = 0; ki < L; ki++) acc += att[ki] * qkv[ki * 3 * d + 2 * d + h * dh + c];
            o[qi * d + h * dh + c] = acc / sum;
          }
        }
      }
      const wo = W[p + "o"], bo = W[p + "o.b"].vector();
      for (let j = 0; j < L; j++) {
        wo.matvec(o, tmp, bo, j * d, 0);
        for (let c = 0; c < d; c++) x[j * d + c] += tmp[c];
      }
      const g2 = W[p + "ln2.g"].vector(), b2 = W[p + "ln2.b"].vector();
      const f1 = W[p + "f1"], f1b = W[p + "f1.b"].vector(), f2 = W[p + "f2"], f2b = W[p + "f2.b"].vector();
      for (let j = 0; j < L; j++) {
        layerNorm(x, j * d, d, g2, b2, y, j * d);
        f1.matvec(y, hid, f1b, j * d, 0);
        for (let c = 0; c < ff; c++) hid[c] = gelu(hid[c]);
        f2.matvec(hid, tmp, f2b, 0, 0);
        for (let c = 0; c < d; c++) x[j * d + c] += tmp[c];
      }
    }
    const gf = W["lnf.g"].vector(), bf = W["lnf.b"].vector();
    for (let j = 0; j < L; j++) layerNorm(x, j * d, d, gf, bf, y, j * d);
    const mean = this.bmean; mean.fill(0);
    for (let j = 0; j < L; j++) for (let c = 0; c < d; c++) mean[c] += y[j * d + c] / L;
    const h1 = this.bh1, h1l = this.bh1l;
    W["h1m"].matvec(mean, h1, W["h1.b"].vector());
    W["h1l"].matvec(y, h1l, undefined, (L - 1) * d, 0);
    for (let c = 0; c < d; c++) h1[c] = gelu(h1[c] + h1l[c]);
    W["h2"].matvec(h1, logits, W["h2.b"].vector());
    return { logits };
  }

  /** multiply-accumulate count for a window of L tokens (for the stats panel) */
  macs(L: number): number {
    const { d, ff, layers } = this;
    return layers * (L * (3 * d * d + d * d + 2 * d * ff) + 2 * L * L * d) + 2 * d * d + d * NUM_OUT;
  }
}
