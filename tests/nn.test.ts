// Parity tests: the TypeScript inference must reproduce the Python/NumPy
// reference (tests/golden.json is written by training/export.py) for every
// precision tier, and the tokenizer/hashes must be bit-exact.
import { describe, expect, it } from "vitest";
import { readFileSync } from "node:fs";
import { join } from "node:path";
import { TransformerNet } from "../src/nn/ann";
import { SpikingNet } from "../src/nn/snn";
import { parseWeights, QTensor, type Manifest } from "../src/nn/qtensor";
import { annIds, fnv1a, snnAfferents, tokenize, NUM_CONCEPTS } from "../src/nn/text";
import { Engine } from "../src/nn/engine";
import { TIERS } from "../src/nn/device";

const ROOT = join(__dirname, "..");
const MODELS = join(ROOT, "public", "models");
const golden = JSON.parse(readFileSync(join(__dirname, "golden.json"), "utf8"));

function load(model: "ann" | "snn", tier: string) {
  const m: Manifest = JSON.parse(readFileSync(join(MODELS, `${model}-${tier}.json`), "utf8"));
  const b = readFileSync(join(MODELS, m.file));
  const buf = b.buffer.slice(b.byteOffset, b.byteOffset + b.byteLength) as ArrayBuffer;
  return { m, w: parseWeights(m, buf) };
}

const sig = (x: number) => 1 / (1 + Math.exp(-x));

describe("text front-end", () => {
  it("normalises, tokenises and hashes like Python", () => {
    expect(tokenize("Città, Perché?! ÉTÉ 2024")).toEqual(["citta", "perche", "ete", "2024"]);
    expect(fnv1a("")).toBe(0x811c9dc5);
    expect(fnv1a("a")).toBe(0xe40c292c);
    expect(golden.hashes.ann).toEqual(golden.texts.map((t: string) => { const r = annIds(tokenize(t)); return [r.uni, r.bi]; }));
    expect(golden.hashes.snn).toEqual(golden.texts.map((t: string) => {
      let prev = "<s>";
      return tokenize(t).map((w) => { const a = snnAfferents(prev, w); prev = w; return a; });
    }));
  });
});

describe("quantised kernels", () => {
  it("q4 / q8 / f16 rowDot and addRow agree with dequantised math", () => {
    // 2 x 40 matrix, int4 with group 32 (two groups per row)
    const vals = Array.from({ length: 80 }, (_, i) => ((i * 7) % 15) - 7);
    const packed = new Uint8Array(40);
    vals.forEach((v, i) => { packed[i >> 1] |= (v + 8) << (i & 1 ? 4 : 0); });
    const scales = new Float32Array([0.5, 2, 0.25, 1]);
    const t = new QTensor("q4", [2, 40], packed.buffer, 0, 40, scales, 32);
    const x = new Float32Array(40).map((_, i) => Math.sin(i));
    for (let r = 0; r < 2; r++) {
      let ref = 0;
      for (let c = 0; c < 40; c++) ref += vals[r * 40 + c] * scales[r * 2 + (c >= 32 ? 1 : 0)] * x[c];
      expect(t.rowDot(r, x)).toBeCloseTo(ref, 4);
      const out = new Float32Array(40);
      t.addRow(r, out, 2);
      expect(out[35]).toBeCloseTo(2 * vals[r * 40 + 35] * scales[r * 2 + 1], 5);
    }
    const h = new Uint16Array([0x3c00, 0xc000, 0x3800]); // 1, -2, 0.5
    const f = new QTensor("f16", [1, 3], h.buffer, 0, 6);
    expect(f.rowDot(0, new Float32Array([1, 1, 1]))).toBeCloseTo(-0.5, 6);
  });
});

for (const tier of TIERS) {
  describe(`tier ${tier}`, () => {
    it("Transformer matches the NumPy reference", () => {
      const { m, w } = load("ann", tier);
      const net = new TransformerNet(w, m);
      const ref: number[][] = golden.outputs[`ann-${tier}`];
      golden.texts.forEach((t: string, i: number) => {
        const out = net.runText(t).logits;
        for (let k = 0; k < out.length; k++) expect(Math.abs(out[k] - ref[i][k])).toBeLessThan(2e-3);
      });
    });

    it("spiking network matches the NumPy reference", () => {
      const { m, w } = load("snn", tier);
      const net = new SpikingNet(w, m);
      const ref: number[][] = golden.outputs[`snn-${tier}`];
      golden.texts.forEach((t: string, i: number) => {
        const out = net.runText(t, m.constants.snnWords).logits;
        // spikes are thresholded: a float-rounding flip can move a logit a
        // little, so compare probabilities with a tolerance
        let maxDiff = 0;
        for (let k = 0; k < NUM_CONCEPTS; k++) maxDiff = Math.max(maxDiff, Math.abs(sig(out[k]) - sig(ref[i][k])));
        expect(maxDiff).toBeLessThan(0.05);
      });
    });
  });
}

describe("engine", () => {
  it("streams words and produces coherent concepts in all three modes", async () => {
    const fetcher = (async (url: string) => {
      const b = readFileSync(join(MODELS, url.replace("mem://", "")));
      return {
        json: async () => JSON.parse(b.toString("utf8")),
        arrayBuffer: async () => b.buffer.slice(b.byteOffset, b.byteOffset + b.byteLength),
      };
    }) as unknown as typeof fetch;
    const e = new Engine();
    await e.load("mem://", "mobile", fetcher);
    const idx = (c: string) => ["ocean", "beach", "forest", "mountain", "desert", "city", "space", "night", "rain", "snow", "fire", "sunset", "river", "flowers", "storm", "underwater", "technology", "music", "love", "clouds", "birds", "fields"].indexOf(c);
    for (const mode of ["ann", "snn", "hybrid"] as const) {
      e.mode = mode;
      e.reset();
      let r = null;
      for (const w of "we sailed on the ocean and watched the waves".split(" ")) r = e.push(w) ?? r;
      const p = r!.final.probs;
      const top = [...p.keys()].sort((a, b) => p[b] - p[a])[0];
      expect(top).toBe(idx("ocean"));
    }
  });
});
