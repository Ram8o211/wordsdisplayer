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
    this.prev = "<s>";
  }

  /** Feed one word (or `null` = a silent word slot). */
  feedWord(tok: string | null) {
    const { n1, n2, w } = this;
    const iw = this.iw;
    iw.set(w["b1"].vector());
    if (tok !== null) {
      for (const a of snnAfferents(this.prev, tok)) { w["w_in"].addRow(a, iw); this.sops += n1; }
      this.prev = tok;
    }
    for (let i = 0; i < n1; i++) iw[i] *= 1 - this.a1[i];
    const beta = w["beta1"].vector();
    const wrec = w["w_rec"], w12 = w["w_12"], wout = w["w_out"], wout1 = w["w_out1"];
    const rec = new Float32Array(n1);
    const in2 = new Float32Array(n2);
    const thr = new Float32Array(n1);
    const acc = new Float32Array(NUM_OUT);
    for (let t = 0; t < this.steps; t++) {
      // layer 1: recurrent adaptive LIF (reset by subtraction)
      rec.fill(0);
      for (let j = 0; j < n1; j++) if (this.z1[j]) { wrec.addRow(j, rec); this.sops += n1; }
      for (let i = 0; i < n1; i++) {
        this.ad[i] = this.r1[i] * this.ad[i] + this.z1[i];
        thr[i] = 1 + beta[i] * this.ad[i];
        this.v1[i] = this.v1[i] * this.a1[i] + iw[i] + rec[i] - this.z1[i] * thr[i];
      }
      let s1 = 0;
      for (let i = 0; i < n1; i++) { const z = this.v1[i] - thr[i] > 0 ? 1 : 0; this.z1[i] = z; s1 += z; }
      // layer 2: LIF
      in2.fill(0);
      for (let j = 0; j < n1; j++) if (this.z1[j]) { w12.addRow(j, in2); this.sops += n2; }
      let s2 = 0;
      for (let i = 0; i < n2; i++) {
        this.v2[i] = this.v2[i] * this.a2[i] + this.b2s[i] + in2[i] - this.z2[i];
        const z = this.v2[i] - 1 > 0 ? 1 : 0;
        this.z2[i] = z; s2 += z;
      }
      // readout: leaky integrators of layer-2 spikes (+ skip connection from layer 1)
      for (let k = 0; k < NUM_OUT; k++) this.y[k] *= this.ko[k];
      for (let j = 0; j < n2; j++) if (this.z2[j]) { wout.addRow(j, this.y); this.sops += NUM_OUT; }
      for (let j = 0; j < n1; j++) if (this.z1[j]) { wout1.addRow(j, this.y); this.sops += NUM_OUT; }
      for (let k = 0; k < NUM_OUT; k++) acc[k] += this.y[k];
      this.spikes += s1 + s2;
      this.neuronSteps += n1 + n2;
    }
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
