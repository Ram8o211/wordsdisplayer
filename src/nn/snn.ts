// Neuromorphic network: event-driven recurrent adaptive-LIF SNN
// (port of training/models.py::snn_forward).
//
// Inference is event-driven like on neuromorphic hardware (Loihi, SpiNNaker,
// Akida): synaptic weights are only touched when a presynaptic neuron spikes,
// so the work per step is proportional to the number of spikes (synaptic
// operations, "SOPs"), not to the number of synapses. State persists across
// words, so the network streams naturally over an endless speech input.
import { snnAfferents, tokenize, NUM_OUT } from "./text";
import type { Manifest, Weights } from "./qtensor";
import type { NetOutput } from "./ann";

export class SpikingNet {
  readonly n1: number;
  readonly n2: number;
  steps: number;
  private v1!: Float32Array; private z1!: Uint8Array; private ad!: Float32Array;
  private v2!: Float32Array; private z2!: Uint8Array; private y!: Float32Array;
  private a1!: Float32Array; private r1!: Float32Array; private a2!: Float32Array; private ko!: Float32Array;
  private iw: Float32Array;
  private readout = new Float32Array(NUM_OUT);
  // preallocated work buffers (no garbage in the per-word hot loop)
  private rec: Float32Array; private in2: Float32Array; private thr: Float32Array;
  private acc = new Float32Array(NUM_OUT);
  private act1: Int32Array; private act2: Int32Array; private nAct1 = 0; private nAct2 = 0;
  private b1: Float32Array; private beta: Float32Array;
  private cache = new Map<string, Float32Array>();
  private b2s!: Float32Array;
  private prev = "<s>";
  /** running statistics */
  sops = 0;
  spikes = 0;
  neuronSteps = 0;

  constructor(private w: Weights, m: Manifest) {
    this.n1 = m.config.n1;
    this.n2 = m.config.n2;
    this.steps = m.knobs.snn_steps;
    this.iw = new Float32Array(this.n1);
    this.rec = new Float32Array(this.n1); this.thr = new Float32Array(this.n1); this.in2 = new Float32Array(this.n2);
    this.act1 = new Int32Array(this.n1); this.act2 = new Int32Array(this.n2);
    this.b1 = w["b1"].vector(); this.beta = w["beta1"].vector();
    this.setSteps(this.steps);
    this.reset();
  }

  /** Time constants are in word units; recompute per-step decays for T steps/word. */
  setSteps(T: number) {
    this.steps = T;
    const dec = (tau: Float32Array) => tau.map((t) => Math.exp(-1 / (t * T)));
    this.a1 = dec(this.w["tau_m1"].vector());
    this.r1 = dec(this.w["tau_a1"].vector());
    this.a2 = dec(this.w["tau_m2"].vector());
    this.ko = dec(this.w["tau_out"].vector());
    const b2 = this.w["b2"].vector();
    this.b2s = b2.map((b, i) => b * (1 - this.a2[i]));
  }

  reset() {
    this.v1 = new Float32Array(this.n1); this.z1 = new Uint8Array(this.n1); this.ad = new Float32Array(this.n1);
    this.v2 = new Float32Array(this.n2); this.z2 = new Uint8Array(this.n2); this.y = new Float32Array(NUM_OUT);
    this.readout = new Float32Array(NUM_OUT);
    this.nAct1 = this.nAct2 = 0;
    this.prev = "<s>";
  }

  /** Input current of a word (before the per-neuron (1 - a) gain), cached. */
  private wordCurrent(prev: string, tok: string): Float32Array {
    const key = prev + "\u0001" + tok;
    const hit = this.cache.get(key);
    if (hit) {
      this.cache.delete(key); // LRU: move to the back
      this.cache.set(key, hit);
      return hit;
    }
    const { n1, w } = this;
    const cur = new Float32Array(n1);
    const aff = snnAfferents(prev, tok);
    if (w["w_in_a"]) {
      // factorised synapses: afferent spikes -> populations -> neurons
      const A = w["w_in_a"], E = w["w_in_e"];
      const u = new Float32Array(A.cols);
      for (const a of aff) { A.addRow(a, u); this.sops += A.cols; }
      for (let k = 0; k < u.length; k++) if (u[k] !== 0) { E.addRow(k, cur, u[k]); this.sops += n1; }
    } else {
      for (const a of aff) { w["w_in"].addRow(a, cur); this.sops += n1; }
    }
    this.cache.set(key, cur);
    if (this.cache.size > 512) this.cache.delete(this.cache.keys().next().value!);
    return cur;
  }

  /** Feed one word (or `null` = a silent word slot). Allocation-free hot loop. */
  feedWord(tok: string | null) {
    const { n1, n2, w } = this;
    const iw = this.iw;
    iw.set(this.b1);
    if (tok !== null) {
      const cur = this.wordCurrent(this.prev, tok);
      for (let i = 0; i < n1; i++) iw[i] += cur[i];
      this.prev = tok;
    }
    for (let i = 0; i < n1; i++) iw[i] *= 1 - this.a1[i];
    const beta = this.beta;
    const wrec = w["w_rec"], w12 = w["w_12"], wout = w["w_out"], wout1 = w["w_out1"];
    const { rec, in2, thr, acc, act1, act2 } = this;
    let nAct1 = this.nAct1, nAct2 = this.nAct2; // spiking neurons of the previous step
    acc.fill(0);
    for (let t = 0; t < this.steps; t++) {
      // layer 1: recurrent adaptive LIF (reset by subtraction); events only
      rec.fill(0);
      for (let k = 0; k < nAct1; k++) wrec.addRow(act1[k], rec);
      this.sops += nAct1 * n1;
      for (let i = 0; i < n1; i++) {
        const z = this.z1[i];
        this.ad[i] = this.r1[i] * this.ad[i] + z;
        thr[i] = 1 + beta[i] * this.ad[i];
        this.v1[i] = this.v1[i] * this.a1[i] + iw[i] + rec[i] - z * thr[i];
      }
      nAct1 = 0;
      for (let i = 0; i < n1; i++) {
        const z = this.v1[i] - thr[i] > 0 ? 1 : 0;
        this.z1[i] = z;
        if (z) act1[nAct1++] = i;
      }
      // layer 2: LIF
      in2.fill(0);
      for (let k = 0; k < nAct1; k++) w12.addRow(act1[k], in2);
      this.sops += nAct1 * n2;
      nAct2 = 0;
      for (let i = 0; i < n2; i++) {
        this.v2[i] = this.v2[i] * this.a2[i] + this.b2s[i] + in2[i] - this.z2[i];
        const z = this.v2[i] - 1 > 0 ? 1 : 0;
        this.z2[i] = z;
        if (z) act2[nAct2++] = i;
      }
      // readout: leaky integrators of layer-2 spikes (+ skip connection from layer 1)
      for (let k = 0; k < NUM_OUT; k++) this.y[k] *= this.ko[k];
      for (let k = 0; k < nAct2; k++) wout.addRow(act2[k], this.y);
      for (let k = 0; k < nAct1; k++) wout1.addRow(act1[k], this.y);
      this.sops += (nAct1 + nAct2) * NUM_OUT;
      for (let k = 0; k < NUM_OUT; k++) acc[k] += this.y[k];
      this.spikes += nAct1 + nAct2;
      this.neuronSteps += n1 + n2;
    }
    this.nAct1 = nAct1;
    this.nAct2 = nAct2;
    // readout of a word = mean of the output integrators over its T steps
    for (let k = 0; k < NUM_OUT; k++) this.readout[k] = acc[k] / this.steps;
  }

  output(): NetOutput {
    const logits = new Float32Array(NUM_OUT);
    const b = this.w["b_out"].vector();
    for (let k = 0; k < NUM_OUT; k++) logits[k] = this.readout[k] + b[k];
    return { logits };
  }

  /** Same protocol as training/export: fresh state, left-pad to `words` slots. */
  runText(text: string, words = 20): NetOutput {
    const toks = tokenize(text).slice(-words);
    this.reset();
    for (let i = toks.length; i < words; i++) this.feedWord(null);
    let prev = "<s>";
    for (const t of toks) { this.prev = prev; this.feedWord(t); prev = t; }
    return this.output();
  }

  get spikeRate(): number {
    return this.neuronSteps ? this.spikes / this.neuronSteps : 0;
  }
}
