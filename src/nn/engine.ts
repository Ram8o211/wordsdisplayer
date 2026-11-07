// Orchestrates the two networks: loading a tier, streaming words, mode
// selection (traditional / neuromorphic / hybrid) and event-gated hybrid.
import { TransformerNet } from "./ann";
import { SpikingNet } from "./snn";
import { parseWeights, type Manifest } from "./qtensor";
import { CONCEPTS, ANN_BUCKETS, ANN_MAX_LEN, SNN_AFFERENTS, tokenize } from "./text";
import { fuse, interpret, l1, type Interpretation } from "./fusion";
import type { Tier } from "./device";

export type Mode = "ann" | "snn" | "hybrid";

export interface EngineResult {
  mode: Mode;
  final: Interpretation;
  ann?: Interpretation;
  snn?: Interpretation;
  hybridWeightAnn?: number;
  annSkipped: boolean;
  timings: { annMs: number; snnMs: number };
  stats: {
    annMacs: number; snnSops: number; spikeRate: number;
    annBytes: number; snnBytes: number; annSkips: number; annRuns: number;
  };
}

async function loadModel(base: string, model: "ann" | "snn", tier: Tier, fetcher: typeof fetch) {
  const m: Manifest = await (await fetcher(`${base}${model}-${tier}.json`)).json();
  const c = m.constants;
  if (c.annBuckets !== ANN_BUCKETS || c.annMaxLen !== ANN_MAX_LEN || c.snnAfferents !== SNN_AFFERENTS ||
      c.concepts.join() !== CONCEPTS.join()) {
    throw new Error(`model ${m.file} was exported with incompatible constants`);
  }
  const buf = await (await fetcher(base + m.file)).arrayBuffer();
  return { manifest: m, weights: parseWeights(m, buf), bytes: buf.byteLength };
}

export class Engine {
  ann!: TransformerNet;
  snn!: SpikingNet;
  annBytes = 0;
  snnBytes = 0;
  mode: Mode = "hybrid";
  /** hybrid only: skip the Transformer while the SNN says the topic is stable */
  eventGated = true;
  gateThreshold = 0.6;
  maxSkip = 4;
  private tokens: string[] = [];
  private lastAnn?: Interpretation;
  private snnAtLastAnn?: Float32Array;
  private skipped = 0;
  private annRuns = 0;
  private annSkips = 0;
  tier!: Tier;

  async load(base: string, tier: Tier, fetcher: typeof fetch = fetch.bind(globalThis)) {
    const [a, s] = await Promise.all([loadModel(base, "ann", tier, fetcher), loadModel(base, "snn", tier, fetcher)]);
    this.ann = new TransformerNet(a.weights, a.manifest);
    this.snn = new SpikingNet(s.weights, s.manifest);
    this.annBytes = a.bytes;
    this.snnBytes = s.bytes;
    this.tier = tier;
    // replay the context into the freshly loaded SNN
    const ctx = this.tokens.slice(-20);
    for (const t of ctx) this.snn.feedWord(t);
    this.lastAnn = undefined;
  }

  reset() {
    this.tokens = [];
    this.snn?.reset();
    this.lastAnn = undefined;
  }

  /** Push transcribed text (one or more words) and get a new interpretation. */
  push(text: string): EngineResult | null {
    const toks = tokenize(text);
    if (!toks.length) return null;
    this.tokens.push(...toks);
    if (this.tokens.length > 256) this.tokens = this.tokens.slice(-128);
    let snnOut: Interpretation | undefined;
    let annOut: Interpretation | undefined;
    let snnMs = 0, annMs = 0;
    const sops0 = this.snn.sops;
    if (this.mode !== "ann") {
      const t0 = performance.now();
      for (const t of toks) this.snn.feedWord(t);
      snnOut = interpret(this.snn.output().logits);
      snnMs = performance.now() - t0;
    }
    let annSkipped = false;
    if (this.mode !== "snn") {
      const gate = this.mode === "hybrid" && this.eventGated && this.lastAnn && this.snnAtLastAnn && snnOut &&
        this.skipped < this.maxSkip && l1(snnOut.probs, this.snnAtLastAnn) < this.gateThreshold;
      if (gate) {
        annOut = this.lastAnn;
        annSkipped = true;
        this.skipped++;
        this.annSkips++;
      } else {
        const t0 = performance.now();
        annOut = interpret(this.ann.runTokens(this.tokens).logits);
        annMs = performance.now() - t0;
        this.lastAnn = annOut;
        this.snnAtLastAnn = snnOut?.probs.slice();
        this.skipped = 0;
        this.annRuns++;
      }
    }
    let final: Interpretation;
    let w: number | undefined;
    if (annOut && snnOut) {
      const f = fuse(annOut, snnOut);
      final = f;
      w = f.weightA;
    } else final = (annOut ?? snnOut)!;
    const L = Math.min(this.tokens.length, this.ann.window);
    return {
      mode: this.mode, final, ann: annOut, snn: snnOut, hybridWeightAnn: w, annSkipped,
      timings: { annMs, snnMs },
      stats: {
        annMacs: annOut && !annSkipped ? this.ann.macs(L) : 0,
        snnSops: this.snn.sops - sops0,
        spikeRate: this.snn.spikeRate,
        annBytes: this.annBytes, snnBytes: this.snnBytes,
        annSkips: this.annSkips, annRuns: this.annRuns,
      },
    };
  }
}
