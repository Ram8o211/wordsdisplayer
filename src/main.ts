import "./styles.css";
import { Renderer, Director } from "./render/renderer";
import { Mic } from "./audio/mic";
import { createAsr, defaultAsr, type Asr, type AsrKind } from "./asr";
import { profileDevice, TIERS, type Tier, type DeviceProfile } from "./nn/device";
import { CONCEPTS } from "./nn/text";
import type { EngineResult, Mode } from "./nn/engine";
import { registerSW } from "./pwa";

const $ = <T extends HTMLElement>(id: string) => document.getElementById(id) as T;
const store = {
  get: (k: string, d: string) => localStorage.getItem("wd." + k) ?? d,
  set: (k: string, v: string) => localStorage.setItem("wd." + k, v),
};

const DEMO = `Last summer we sailed across the ocean, the waves were calm and the seagulls followed our boat.
Then a terrible storm arrived with thunder and lightning and heavy rain.
At night the sky cleared and we could see the stars, the moon and the whole galaxy.
In the morning we hiked through a pine forest up to the snowy mountain peak.
Back in the city the streets were full of traffic and lights.
In the evening there was a concert, music and dancing everywhere.
And finally we watched a golden sunset on the beach, together, in love.`;

// ---------------------------------------------------------------- state
let profile: DeviceProfile;
let mode = store.get("mode", "hybrid") as Mode;
let tierChoice = store.get("tier", "auto");
let asrKind = store.get("asr", defaultAsr()) as AsrKind;
let lang = store.get("lang", "en");
let gated = store.get("gated", "1") === "1";
let asr: Asr | null = null;
let listening = false;
let last: EngineResult | null = null;
let loaded: { tier: Tier; annBytes: number; snnBytes: number; annWindow: number; snnSteps: number } | null = null;
let demoTimer: number | undefined;
let maxScale = 1;
const director = new Director();
const mic = new Mic();
const worker = new Worker(new URL("./nn/worker.ts", import.meta.url), { type: "module" });
const status = (s: string) => ($("status").textContent = s);

// ---------------------------------------------------------------- engine
function modelBase(): string {
  return new URL("models/", document.baseURI).href;
}

function applyTier() {
  const tier = (tierChoice === "auto" ? profile.tier : tierChoice) as Tier;
  status(`loading ${tier} models…`);
  worker.postMessage({ type: "load", base: modelBase(), tier });
  const rs = { nano: 0.45, mobile: 0.6, desktop: 0.85, ultra: 1 }[tier];
  maxScale = tierChoice === "auto" ? profile.renderScale : rs;
  renderer.scale = maxScale;
  // the video generator network itself is sized per tier (Fourier octaves, sub-experts)
  renderer.setDetail({ nano: 0, mobile: 1, desktop: 2, ultra: 3 }[tier]);
  director.maxLayers = { nano: 3, mobile: 4, desktop: 5, ultra: 6 }[tier];
}

function sendMode() {
  worker.postMessage({ type: "mode", mode, eventGated: gated });
  document.querySelectorAll<HTMLButtonElement>("#mode button").forEach((b) => b.classList.toggle("active", b.dataset.mode === mode));
}

worker.onmessage = (ev) => {
  const m = ev.data;
  if (m.type === "loaded") {
    loaded = m;
    status("");
    updateStats();
  } else if (m.type === "result") {
    last = m.result as EngineResult;
    director.update(last.final);
    showConcepts();
    updateBars();
    updateStats();
  } else if (m.type === "error") {
    status("⚠ " + m.message);
  }
};

function feed(words: string) {
  worker.postMessage({ type: "text", text: words });
  director.hear(words);
}

// ---------------------------------------------------------------- UI: captions
let transcriptWords: string[] = [];
function showTranscript(text: string, final: boolean) {
  const el = $("transcript");
  const words = text.trim().split(/\s+/).filter(Boolean);
  const shown = [...transcriptWords, ...words].slice(-18);
  el.innerHTML = shown.map((w, i) => (i >= shown.length - words.length ? `<span class="new">${esc(w)}</span>` : esc(w))).join(" ");
  if (final) transcriptWords = [...transcriptWords, ...words].slice(-18);
}
const esc = (s: string) => s.replace(/[&<>"]/g, (c) => `&#${c.charCodeAt(0)};`);

function showConcepts() {
  if (!last) return;
  const p = last.final.probs;
  const top = [...p.keys()].filter((i) => p[i] > 0.35).sort((a, b) => p[b] - p[a]).slice(0, director.maxLayers);
  $("concepts").innerHTML = top.map((i) => `<span class="chip">${CONCEPTS[i]} ${(p[i] * 100).toFixed(0)}%</span>`).join("");
}

function updateBars() {
  if (!last || $("panel").hidden) return;
  const a = last.ann?.probs, s = last.snn?.probs;
  const rows = CONCEPTS.map((c, i) => ({ c, i, v: last!.final.probs[i] }))
    .sort((x, y) => y.v - x.v).slice(0, 8)
    .map(({ c, i }) => `<div class="bar"><span>${c}</span><i><b style="width:${((a?.[i] ?? 0) * 100).toFixed(0)}%"></b></i><i class="s"><b style="width:${((s?.[i] ?? 0) * 100).toFixed(0)}%"></b></i></div>`);
  $("bars").innerHTML = `<div class="bar legend"><span></span><span>Transformer</span><span>Spiking</span></div>` + rows.join("");
}

function updateStats() {
  if ($("panel").hidden) return;
  const kb = (b: number) => (b / 1024).toFixed(0) + " KB";
  const st = last?.stats;
  $("stats").textContent = [
    `device: ${profile.cores} cores, ${profile.memoryGB ?? "?"} GB, ${profile.mobile ? "mobile" : "desktop"}`,
    `gpu: ${profile.gpu}  · render scale ${renderer.scale.toFixed(2)}`,
    `js speed: ${profile.mflops} MFLOP/s  score ${profile.score} → auto tier "${profile.tier}"`,
    loaded ? `loaded tier: ${loaded.tier}  (ANN ${kb(loaded.annBytes)}, SNN ${kb(loaded.snnBytes)})` : "loading…",
    loaded ? `ANN window ${loaded.annWindow} words · SNN ${loaded.snnSteps} steps/word` : "",
    last ? `last word: ANN ${last.timings.annMs.toFixed(1)} ms${last.annSkipped ? " (gated, skipped)" : ""}, SNN ${last.timings.snnMs.toFixed(1)} ms` : "",
    st ? `ANN MACs ${(st.annMacs / 1e6).toFixed(2)} M · SNN synaptic ops ${(st.snnSops / 1e3).toFixed(1)} k` : "",
    st ? `SNN spike rate ${(st.spikeRate * 100).toFixed(1)}% · ANN runs ${st.annRuns}, gated skips ${st.annSkips}` : "",
    last?.hybridWeightAnn !== undefined ? `fusion weight: Transformer ${(last.hybridWeightAnn * 100).toFixed(0)}% / SNN ${(100 - last.hybridWeightAnn * 100).toFixed(0)}%` : "",
    last ? `valence ${last.final.valence.toFixed(2)} · arousal ${last.final.arousal.toFixed(2)}` : "",
    videoStats(),
  ].filter(Boolean).join("\n");
}

function videoStats(): string {
  const z = renderer.net.size, c = renderer.cost();
  const px = renderer.backend === "webgl2" ? (document.getElementById("scene") as HTMLCanvasElement) : null;
  const gmac = px ? (c.macs * px.width * px.height * 60) / 1e9 : 0;
  return [
    `video net (${renderer.backend}, detail ${renderer.detail}): ${z.neurons} neurons, ${z.synapses} synapses, ${z.layers} experts, ${z.dynamicParams} generated weights/frame`,
    `this frame: ${c.activeLayers}/${c.totalLayers} experts on · ${c.neurons} neurons, ${c.macs} MAC, ${c.transcendental} sin/exp per pixel` +
      (px ? ` · ${px.width}×${px.height} → ${gmac.toFixed(2)} GMAC/s @60fps` : ""),
  ].join("\n");
}

// ---------------------------------------------------------------- listening
async function startListening() {
  stopDemo();
  listening = true;
  $("mic").classList.add("on");
  try {
    await mic.start();
  } catch {
    /* features are optional (e.g. mic owned by the native recogniser) */
  }
  if (asrKind === "text") {
    status("type in the settings panel");
    return;
  }
  asr = createAsr(asrKind, lang, {
    onWords: feed,
    onTranscript: showTranscript,
    onStatus: status,
  });
  try {
    await asr!.start();
  } catch (e) {
    status("⚠ " + ((e as Error).message ?? e));
    await stopListening();
  }
}

async function stopListening() {
  listening = false;
  $("mic").classList.remove("on");
  await asr?.stop().catch(() => {});
  asr = null;
  mic.stop();
  status("");
}

function stopDemo() {
  if (demoTimer) clearInterval(demoTimer);
  demoTimer = undefined;
}

function playDemo() {
  stopDemo();
  worker.postMessage({ type: "reset" });
  const words = DEMO.split(/\s+/);
  let i = 0;
  let line: string[] = [];
  demoTimer = window.setInterval(() => {
    if (i >= words.length) return stopDemo();
    const w = words[i++];
    line.push(w);
    feed(w);
    const end = /[.!?]$/.test(w);
    showTranscript(line.join(" "), end);
    if (end) line = [];
  }, 380);
}

// ---------------------------------------------------------------- wiring
const renderer = new Renderer($("scene"));

function init() {
  profile = profileDevice();
  ($("tier") as HTMLSelectElement).value = tierChoice;
  ($("asr") as HTMLSelectElement).value = asrKind;
  ($("lang") as HTMLSelectElement).value = lang;
  ($("gated") as HTMLInputElement).checked = gated;
  applyTier();
  sendMode();

  $("mic").onclick = () => (listening ? stopListening() : startListening());
  document.querySelectorAll<HTMLButtonElement>("#mode button").forEach((b) => {
    b.onclick = () => { mode = b.dataset.mode as Mode; store.set("mode", mode); sendMode(); };
  });
  $("toggle-panel").onclick = () => { $("panel").hidden = !$("panel").hidden; updateBars(); updateStats(); };
  $("tier").onchange = (e) => { tierChoice = (e.target as HTMLSelectElement).value; store.set("tier", tierChoice); applyTier(); };
  $("asr").onchange = async (e) => {
    asrKind = (e.target as HTMLSelectElement).value as AsrKind; store.set("asr", asrKind);
    if (listening) { await stopListening(); await startListening(); }
  };
  $("lang").onchange = async (e) => {
    lang = (e.target as HTMLSelectElement).value; store.set("lang", lang);
    if (listening) { await stopListening(); await startListening(); }
  };
  $("gated").onchange = (e) => { gated = (e.target as HTMLInputElement).checked; store.set("gated", gated ? "1" : "0"); sendMode(); };
  $("typed").onsubmit = (e) => {
    e.preventDefault();
    const inp = $("typed-text") as HTMLInputElement;
    if (inp.value.trim()) { feed(inp.value); showTranscript(inp.value, true); }
    inp.value = "";
  };
  $("demo").onclick = playDemo;

  let prev = performance.now();
  let avg = 1 / 60, tuneAt = 0, statsAt = 0;
  const frame = (now: number) => {
    const dt = Math.min(0.1, (now - prev) / 1000);
    prev = now;
    // dynamic resolution: keep ~50+ fps on weak GPUs, climb back when there is headroom
    avg += (dt - avg) * 0.05;
    if (now > tuneAt) {
      if (avg > 1 / 45 && renderer.scale > 0.3) { renderer.scale = Math.max(0.3, renderer.scale * 0.85); tuneAt = now + 1000; }
      else if (avg < 1 / 57 && renderer.scale < maxScale) { renderer.scale = Math.min(maxScale, renderer.scale * 1.1); tuneAt = now + 2000; }
    }
    const au = mic.update();
    director.tick(dt, au);
    renderer.draw(now / 1000, director, au);
    if (now > statsAt) { statsAt = now + 1000; updateStats(); }
    requestAnimationFrame(frame);
  };
  requestAnimationFrame(frame);

  const qs = new URLSearchParams(location.search);
  if (qs.has("demo")) playDemo();
  // ?scene=ocean,sunset  -> preview a scene without the networks (debugging / thumbnails)
  const scene = qs.get("scene");
  if (scene) {
    const probs = new Float32Array(CONCEPTS.length);
    for (const c of scene.split(",")) { const i = CONCEPTS.indexOf(c as never); if (i >= 0) probs[i] = 0.95; }
    director.maxLayers = 6;
    director.update({ probs, valence: Number(qs.get("v") ?? 0.3), arousal: Number(qs.get("a") ?? 0) });
    director.weights.set(director.target);
  }
}

void TIERS;
init();
registerSW();
