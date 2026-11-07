// Per-word latency of each network and tier (run: npx vitest run tests/bench.test.ts)
import { it } from "vitest";
import { readFileSync } from "node:fs";
import { join } from "node:path";
import { TransformerNet } from "../src/nn/ann";
import { SpikingNet } from "../src/nn/snn";
import { parseWeights, type Manifest } from "../src/nn/qtensor";
import { TIERS } from "../src/nn/device";
import { tokenize } from "../src/nn/text";

const M = join(__dirname, "..", "public", "models");
const load = (model: string, tier: string) => {
  const m: Manifest = JSON.parse(readFileSync(join(M, `${model}-${tier}.json`), "utf8"));
  const b = readFileSync(join(M, m.file));
  return { m, w: parseWeights(m, b.buffer.slice(b.byteOffset, b.byteOffset + b.byteLength) as ArrayBuffer) };
};
const TEXT = tokenize("we sailed across the ocean and then a terrible storm arrived over the city at night with rain and thunder ".repeat(4));

it("latency per word", () => {
  const rows: string[] = [];
  for (const tier of TIERS) {
    const a = load("ann", tier), s = load("snn", tier);
    const ann = new TransformerNet(a.w, a.m), snn = new SpikingNet(s.w, s.m);
    const time = (f: (i: number) => void, n: number) => { for (let i = 0; i < 20; i++) f(i); const t = performance.now(); for (let i = 0; i < n; i++) f(i); return (performance.now() - t) / n; };
    const ta = time((i) => ann.runTokens(TEXT.slice(0, 48 + (i % 10))), 200);
    snn.sops = 0;
    const ts = time((i) => snn.feedWord(TEXT[i % TEXT.length]), 2000);
    rows.push(`${tier.padEnd(8)} ANN ${ta.toFixed(3)} ms/word (window ${ann.window})  SNN ${ts.toFixed(3)} ms/word (T=${snn.steps}, spike rate ${(snn.spikeRate * 100).toFixed(2)}%, ${(snn.sops / 2020 / 1000).toFixed(1)}k SOPs/word)  MACs ANN ${(ann.macs(ann.window) / 1e6).toFixed(2)}M`);
  }
  console.log(rows.join("\n"));
}, 120_000);

// Neural video generator: size and per-pixel cost per tier
import { design, frameInputs } from "../src/render/neural/design";
import { compile } from "../src/render/neural/compile";
import { CONCEPTS } from "../src/nn/text";

it("video generator cost per tier", () => {
  const scenes = ["", "ocean,sunset", "fire", "love"];
  const res = { nano: [360, 640, 0.45], mobile: [720, 1280, 0.6], desktop: [1080, 1920, 0.85], ultra: [1080, 1920, 1] } as const;
  const rows: string[] = [];
  TIERS.forEach((tier, detail) => {
    const net = compile(design(detail));
    let macs = 0, tr = 0, maxM = 0, hyper = 0;
    for (const s of scenes) {
      const c = new Float32Array(22);
      for (const n of s.split(",")) { const i = CONCEPTS.indexOf(n as never); if (i >= 0) c[i] = 1; }
      const inp = frameInputs({ concepts: c, valence: 0.3, arousal: 0, audio: { rms: 0.2, low: 0.2, mid: 0.2, flux: 0 }, dt: 0.016, flash: 0, aspect: 16 / 9 });
      const t0 = performance.now();
      for (let k = 0; k < 50; k++) net.update(inp);
      hyper += (performance.now() - t0) / 50;
      const cst = net.cost();
      macs += cst.macs; tr += cst.transcendental; maxM = Math.max(maxM, cst.macs);
    }
    macs /= scenes.length; tr /= scenes.length; hyper /= scenes.length;
    const W = 160, H = 90, buf = new Uint8ClampedArray(W * H * 4);
    net.renderCPU(buf, W, H);
    const t0 = performance.now();
    for (let k = 0; k < 5; k++) net.renderCPU(buf, W, H);
    const cpuMs = (performance.now() - t0) / 5;
    const [h, w, s] = res[tier];
    const px = h * w * s * s;
    rows.push(`${tier.padEnd(8)} ${net.size.neurons} neurons, ${net.size.synapses} synapses, ${net.size.layers} paint layers, ${net.size.dynamicParams} generated weights, ${net.size.frameUnits} hypernet units | ` +
      `mean ${macs.toFixed(0)} MAC + ${tr.toFixed(0)} sin/exp per px (max ${maxM}) | hypernet ${(hyper * 1000).toFixed(0)} µs/frame | ` +
      `CPU kernel 160×90 ${cpuMs.toFixed(1)} ms | ${Math.round(w * s)}×${Math.round(h * s)} @30fps: ${((macs * px * 30) / 1e9).toFixed(2)} GMAC/s`);
  });
  console.log(rows.join("\n"));
}, 120_000);
