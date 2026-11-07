// Bit-exact port of training/common.py (tokenisation + hashing).

export const CONCEPTS = [
  "ocean", "beach", "forest", "mountain", "desert", "city", "space", "night",
  "rain", "snow", "fire", "sunset", "river", "flowers", "storm", "underwater",
  "technology", "music", "love", "clouds", "birds", "fields",
] as const;
export type Concept = (typeof CONCEPTS)[number];
export const NUM_CONCEPTS = CONCEPTS.length;
export const NUM_OUT = NUM_CONCEPTS + 2; // + valence, arousal

export const ANN_BUCKETS = 8192;
export const ANN_MAX_LEN = 48;
export const SNN_AFFERENTS = 4096;
export const SNN_WORD_SPIKES = 10;
export const SNN_BIGRAM_SPIKES = 2;

const encoder = new TextEncoder();

export function normalize(text: string): string {
  return text.toLowerCase().normalize("NFD").replace(/\p{Mn}/gu, "");
}

export function tokenize(text: string): string[] {
  return normalize(text).match(/[a-z0-9]+/g) ?? [];
}

export function fnv1a(s: string): number {
  let h = 0x811c9dc5;
  for (const b of encoder.encode(s)) {
    h ^= b;
    h = Math.imul(h, 0x01000193) >>> 0;
  }
  return h >>> 0;
}

/** Unigram and bigram hash buckets for each token. */
export function annIds(tokens: string[]): { uni: number[]; bi: number[] } {
  const uni: number[] = [];
  const bi: number[] = [];
  let prev = "<s>";
  for (const t of tokens) {
    uni.push(fnv1a("w:" + t) % ANN_BUCKETS);
    bi.push(fnv1a("b:" + prev + "_" + t) % ANN_BUCKETS);
    prev = t;
  }
  return { uni, bi };
}

/** Sparse binary code of a word: indices of afferents that spike. */
export function snnAfferents(prev: string, tok: string): number[] {
  const s = new Set<number>();
  for (let i = 0; i < SNN_WORD_SPIKES; i++) s.add(fnv1a("s:" + tok + ":" + i) % SNN_AFFERENTS);
  for (let i = 0; i < SNN_BIGRAM_SPIKES; i++) s.add(fnv1a("p:" + prev + "_" + tok + ":" + i) % SNN_AFFERENTS);
  return [...s].sort((a, b) => a - b);
}
