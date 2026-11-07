import { NUM_CONCEPTS } from "./text";

export interface Interpretation {
  probs: Float32Array; // concept probabilities
  valence: number;     // [-1, 1]
  arousal: number;     // [-1, 1]
}

const sigmoid = (x: number) => 1 / (1 + Math.exp(-x));

export function interpret(logits: Float32Array): Interpretation {
  const probs = new Float32Array(NUM_CONCEPTS);
  for (let i = 0; i < NUM_CONCEPTS; i++) probs[i] = sigmoid(logits[i]);
  return { probs, valence: Math.tanh(logits[NUM_CONCEPTS]), arousal: Math.tanh(logits[NUM_CONCEPTS + 1]) };
}

/** Mean decisiveness |2p-1| of a prediction: a calibration-free confidence. */
export function confidence(p: Float32Array): number {
  let c = 0;
  for (const v of p) c += Math.abs(2 * v - 1);
  return c / p.length;
}

/**
 * Hybrid mode: confidence-weighted late fusion of the two networks
 * (identical to training/export.py::fuse, which is what the report measures).
 */
export function fuse(a: Interpretation, b: Interpretation): Interpretation & { weightA: number } {
  const ca = confidence(a.probs), cb = confidence(b.probs);
  const wa = ca / (ca + cb + 1e-6);
  const probs = new Float32Array(NUM_CONCEPTS);
  for (let i = 0; i < NUM_CONCEPTS; i++) probs[i] = wa * a.probs[i] + (1 - wa) * b.probs[i];
  return {
    probs,
    valence: wa * a.valence + (1 - wa) * b.valence,
    arousal: wa * a.arousal + (1 - wa) * b.arousal,
    weightA: wa,
  };
}

export function l1(a: Float32Array, b: Float32Array): number {
  let s = 0;
  for (let i = 0; i < a.length; i++) s += Math.abs(a[i] - b[i]);
  return s;
}
