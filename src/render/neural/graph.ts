// Neural video generator: graph definition.
//
// The video is produced pixel by pixel by a sparse feed-forward network whose
// weights are ASSIGNED (constructed analytically in design.ts), never trained.
// It is made of two networks:
//
//  * FrameNet (the hypernetwork) runs once per frame on the CPU. Its inputs are
//    the semantic code produced by the speech networks (22 concept weights,
//    valence, arousal), audio features and time. Its units compute scene
//    quantities (darkness, horizon height, sun position, colour palette,
//    animation phases...). Every weight of the pixel network is a linear
//    function of these units, so the hypernetwork *generates the weights* of
//    the pixel network for the current frame.
//
//  * PixelNet runs once per pixel on the GPU (or on the CPU as a fallback).
//    Input: the pixel coordinate encoded as [x, y, x², y², xy]. Hidden
//    neurons have sparse incoming connections and one activation each
//    (sine = Fourier/SIREN feature, logistic = soft half-space / edge,
//    abs, reciprocal for perspective, sqrt for radial distance, and
//    sigma-pi product units). The output stage is a gated recurrence over
//    depth (GRU-like update c ← c + g·(colour − c)) followed by a per-frame
//    linear colour-grading layer.
//
// "Layers" in the output stage are experts of a mixture: each has an opacity
// produced by the hypernetwork; experts with zero opacity are skipped for the
// whole frame (uniform branch), so per-pixel cost scales with what is on screen.

export type FAct = "id" | "clip" | "relu" | "sin" | "cos" | "sig" | "abs" | "inv" | "tanh";
export type Act = "input" | "lin" | "sin" | "sig" | "edge" | "abs" | "rcp" | "relu" | "clip" | "sqrt" | "prod" | "tanh" | "gauss" | "bank" | "bank2";

/** A weight of the pixel network: c + Σ coef · frameUnit. */
export class Param {
  constructor(public c: number, public t: Map<number, number> = new Map()) {}
  get isConst() { return this.t.size === 0; }
  key(): string {
    return this.c.toPrecision(9) + "|" + [...this.t].sort((a, b) => a[0] - b[0]).map(([u, k]) => `${u}:${k.toPrecision(9)}`).join(",");
  }
}
export type PL = Param | number;
const asP = (p: PL) => (typeof p === "number" ? new Param(p) : p);

type FUnit =
  | { kind: "in"; name: string }
  | { kind: "lin"; p: Param; act: FAct }
  | { kind: "mul"; a: number; b: number }
  /** integrator neuron: state += input · dt (phase accumulators; wrapped to [-π, π)) */
  | { kind: "int"; p: Param; state: number };

export class FrameNet {
  units: FUnit[] = [];
  private inputs = new Map<string, number>();
  private mulCache = new Map<string, number>();

  input(name: string): Param {
    let i = this.inputs.get(name);
    if (i === undefined) {
      i = this.units.push({ kind: "in", name }) - 1;
      this.inputs.set(name, i);
    }
    return new Param(0, new Map([[i, 1]]));
  }
  inputNames() { return [...this.inputs.keys()]; }

  add(...ps: PL[]): Param {
    const r = new Param(0);
    for (const q of ps.map(asP)) {
      r.c += q.c;
      for (const [u, k] of q.t) r.t.set(u, (r.t.get(u) ?? 0) + k);
    }
    for (const [u, k] of r.t) if (k === 0) r.t.delete(u);
    return r;
  }
  scale(p: PL, k: number): Param {
    const q = asP(p);
    return new Param(q.c * k, new Map([...q.t].map(([u, v]) => [u, v * k])));
  }
  sub(a: PL, b: PL) { return this.add(a, this.scale(b, -1)); }
  /** product of two parameters: cross terms become multiplicative (sigma-pi) units */
  mul(a: PL, b: PL): Param {
    const p = asP(a), q = asP(b);
    const r = this.add(this.scale(q, p.c), this.scale(p, q.c), new Param(-p.c * q.c));
    r.c = p.c * q.c;
    for (const [u, ku] of p.t) for (const [v, kv] of q.t) {
      const key = u < v ? `${u},${v}` : `${v},${u}`;
      let m = this.mulCache.get(key);
      if (m === undefined) {
        m = this.units.push({ kind: "mul", a: u, b: v }) - 1;
        this.mulCache.set(key, m);
      }
      r.t.set(m, (r.t.get(m) ?? 0) + ku * kv);
    }
    return r;
  }
  mix(a: PL, b: PL, t: PL) { return this.add(a, this.mul(t, this.sub(b, a))); }
  act(p: PL, act: FAct): Param {
    const q = asP(p);
    if (q.isConst) {
      const v = evalFAct(act, q.c);
      return new Param(v);
    }
    const i = this.units.push({ kind: "lin", p: q, act }) - 1;
    return new Param(0, new Map([[i, 1]]));
  }
  /** integrate a rate over time (inputs.dt): smooth phases even when the rate changes */
  integrate(rate: PL): Param {
    const i = this.units.push({ kind: "int", p: asP(rate), state: 0 }) - 1;
    return new Param(0, new Map([[i, 1]]));
  }
  clip(p: PL) { return this.act(p, "clip"); }
  relu(p: PL) { return this.act(p, "relu"); }
  sin(p: PL) { return this.act(p, "sin"); }
  cos(p: PL) { return this.act(p, "cos"); }
  max(a: PL, b: PL) { return this.add(a, this.relu(this.sub(b, a))); }

  /** Run the hypernetwork. */
  run(inputs: Record<string, number>, out = new Float64Array(this.units.length)): Float64Array {
    const u = this.units;
    for (let i = 0; i < u.length; i++) {
      const x = u[i];
      if (x.kind === "in") out[i] = inputs[x.name] ?? 0;
      else if (x.kind === "mul") out[i] = out[x.a] * out[x.b];
      else if (x.kind === "int") {
        let s = x.p.c;
        for (const [k, w] of x.p.t) s += w * out[k];
        let v = x.state + s * (inputs.dt ?? 0);
        v -= 2 * Math.PI * Math.floor(v / (2 * Math.PI) + 0.5);
        out[i] = x.state = v;
      }
      else {
        let s = x.p.c;
        for (const [k, w] of x.p.t) s += w * out[k];
        out[i] = evalFAct(x.act, s);
      }
    }
    return out;
  }
}

function evalFAct(a: FAct, s: number): number {
  switch (a) {
    case "id": return s;
    case "clip": return Math.min(1, Math.max(0, s));
    case "relu": return Math.max(0, s);
    case "sin": return Math.sin(s);
    case "cos": return Math.cos(s);
    case "sig": return 1 / (1 + Math.exp(-s));
    case "abs": return Math.abs(s);
    case "inv": return 1 / s;
    case "tanh": return Math.tanh(s);
  }
}

export interface Neuron {
  act: Act;
  in: { src: number; w: Param }[];
  b: Param;
  /** optional gain applied to the weighted sum before the bias (FiLM-style modulation) */
  g?: Param;
  /** spectral bank: row in the bank table (inputs: [u, v] local coordinates) */
  bankRow?: number;
  /** product units (and bank2: the bank neuron whose second readout this is) */
  pa?: number;
  pb?: number;
  name?: string;
}

export interface Layer {
  name: string;
  /** neuron giving the per-pixel mask (-1 = whole frame) */
  gate: number;
  /** neuron in [0,1] choosing between the two colours (-1 = c0) */
  shade: number;
  c0: Param[];
  c1: Param[];
  alpha: Param;
  /** additive light instead of over-compositing */
  add: boolean;
}

export const INPUTS = ["x", "y", "x2", "y2", "xy"] as const;

export class PixelNet {
  neurons: Neuron[] = INPUTS.map((name) => ({ act: "input" as Act, in: [], b: new Param(0), name }));
  layers: Layer[] = [];
  /** 3x3 colour matrix (row-major) + offset, generated per frame */
  grade: { M: Param[]; o: Param[] } = { M: [1, 0, 0, 0, 1, 0, 0, 0, 1].map(asP), o: [0, 0, 0].map(asP) };
  /**
   * Spectral sine banks: row r holds K modes [wx, wy, phi, a, b]; a bank neuron
   * outputs Σ a·sin(wx·u + wy·v + phi) and its bank2 twin Σ b·sin(...). Weights
   * that large are kept in a float texture instead of being emitted as code.
   */
  banks: number[][][] = [];
  bankK = 0;
  /** multiplicative output neuron (vignette), -1 = none */
  vignette = -1;
  readonly X = 0; readonly Y = 1; readonly X2 = 2; readonly Y2 = 3; readonly XY = 4;

  constructor(public F: FrameNet) {}

  n(act: Exclude<Act, "input" | "prod">, terms: [number, PL][], bias: PL = 0, name?: string, gain?: PL): number {
    const t = terms.map(([src, w]) => ({ src, w: asP(w) })).filter((x) => !(x.w.isConst && x.w.c === 0));
    return this.neurons.push({ act, in: t, b: asP(bias), name, g: gain === undefined ? undefined : asP(gain) }) - 1;
  }
  /** spectral sine bank over local coordinates (u, v): returns [readout a, readout b] */
  bank(modes: number[][], u: number, v: number, name?: string): [number, number] {
    const row = this.banks.push(modes) - 1;
    const i = this.neurons.push({ act: "bank", in: [{ src: u, w: new Param(1) }, { src: v, w: new Param(1) }], b: new Param(0), bankRow: row, name }) - 1;
    const j = this.neurons.push({ act: "bank2", in: [], b: new Param(0), pa: i, pb: i, name: name && name + " shade" }) - 1;
    return [i, j];
  }
  prod(a: number, b: number, name?: string): number {
    return this.neurons.push({ act: "prod", in: [], b: new Param(0), pa: a, pb: b, name }) - 1;
  }
  layer(l: Omit<Layer, "c0" | "c1" | "alpha" | "gate" | "shade" | "add"> & { gate?: number; shade?: number; c0: PL[]; c1?: PL[]; alpha: PL; add?: boolean }) {
    this.layers.push({
      name: l.name, gate: l.gate ?? -1, shade: l.shade ?? -1, add: l.add ?? false,
      c0: l.c0.map(asP), c1: (l.c1 ?? l.c0).map(asP), alpha: asP(l.alpha),
    });
  }
}
