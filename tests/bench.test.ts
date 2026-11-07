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
