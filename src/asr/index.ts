// Speech-to-text front-ends. The two neural networks consume words, so any
// recogniser that streams words works. All options can run on-device:
//   vosk    : Kaldi/Vosk compiled to WebAssembly - fully offline on every
//             platform (browser, Electron, Android/iOS WebView). Model is
//             downloaded once and cached.
//   native  : Android SpeechRecognizer / iOS SFSpeechRecognizer through
//             Capacitor (on-device on recent OS versions).
//   web     : Web Speech API (Chrome uses on-device recognition when
//             `processLocally` is supported, otherwise a cloud service).
//   text    : typed text / demo script (no microphone needed).
import { Capacitor } from "@capacitor/core";

export type AsrKind = "vosk" | "native" | "web" | "text";

export interface AsrCallbacks {
  /** New words recognised (already de-duplicated across partial results). */
  onWords(words: string): void;
  /** Current transcript line for display. */
  onTranscript(text: string, final: boolean): void;
  onStatus(msg: string): void;
}

/** Turns a stream of partial/final hypotheses into newly-heard words. */
export class WordTracker {
  private emitted = 0;
  constructor(private cb: AsrCallbacks) {}
  hypothesis(text: string, final: boolean) {
    const words = text.trim().split(/\s+/).filter(Boolean);
    // keep the last word of a partial back: it is often revised
    const stable = final ? words.length : Math.max(0, words.length - 1);
    if (stable > this.emitted) {
      this.cb.onWords(words.slice(this.emitted, stable).join(" "));
      this.emitted = stable;
    }
    this.cb.onTranscript(text, final);
    if (final) this.emitted = 0;
  }
}

export interface Asr {
  start(): Promise<void>;
  stop(): Promise<void>;
}

export const VOSK_MODELS: Record<string, string> = {
  en: "https://ccoreilly.github.io/vosk-browser/models/vosk-model-small-en-us-0.15.tar.gz",
  it: "https://ccoreilly.github.io/vosk-browser/models/vosk-model-small-it-0.4.tar.gz",
};

class VoskAsr implements Asr {
  private model: any;
  private rec: any;
  private node?: ScriptProcessorNode;
  private ctx?: AudioContext;
  private stream?: MediaStream;
  constructor(private lang: string, private cb: AsrCallbacks, private url?: string) {}
  async start() {
    const tracker = new WordTracker(this.cb);
    this.cb.onStatus("loading offline speech model (first run downloads ~40 MB)…");
    const Vosk = await import("vosk-browser");
    this.model = this.model ?? (await Vosk.createModel(this.url || VOSK_MODELS[this.lang] || VOSK_MODELS.en));
    this.stream = await navigator.mediaDevices.getUserMedia({ audio: { echoCancellation: true, noiseSuppression: true, channelCount: 1 } });
    this.ctx = new AudioContext();
    this.rec = new this.model.KaldiRecognizer(this.ctx.sampleRate);
    this.rec.on("result", (m: any) => m.result?.text && tracker.hypothesis(m.result.text, true));
    this.rec.on("partialresult", (m: any) => m.result?.partial && tracker.hypothesis(m.result.partial, false));
    const src = this.ctx.createMediaStreamSource(this.stream);
    // ScriptProcessor is deprecated but is what vosk-browser's API expects and
    // it works in every WebView; the work done here is a single copy.
    this.node = this.ctx.createScriptProcessor(4096, 1, 1);
    this.node.onaudioprocess = (e) => { try { this.rec.acceptWaveform(e.inputBuffer); } catch { /* model busy */ } };
    src.connect(this.node);
    this.node.connect(this.ctx.destination);
    this.cb.onStatus("listening (Vosk, offline)");
  }
  async stop() {
    this.node?.disconnect();
    this.stream?.getTracks().forEach((t) => t.stop());
    await this.ctx?.close();
    this.rec?.remove();
  }
}

class WebSpeechAsr implements Asr {
  private r: any;
  private running = false;
  constructor(private lang: string, private cb: AsrCallbacks) {}
  async start() {
    const SR = (window as any).SpeechRecognition ?? (window as any).webkitSpeechRecognition;
    if (!SR) throw new Error("Web Speech API not available here – choose Vosk (offline)");
    const tracker = new WordTracker(this.cb);
    const r = (this.r = new SR());
    r.lang = this.lang === "it" ? "it-IT" : "en-US";
    r.continuous = true;
    r.interimResults = true;
    let local = false;
    if ("processLocally" in r) {
      try {
        const st = await SR.available?.({ langs: [r.lang], processLocally: true });
        if (st === "available") { r.processLocally = true; local = true; }
        else if (st === "downloadable") { await SR.install?.({ langs: [r.lang], processLocally: true }); r.processLocally = true; local = true; }
      } catch { /* fall back to default */ }
    }
    r.onresult = (e: any) => {
      for (let i = e.resultIndex; i < e.results.length; i++) tracker.hypothesis(e.results[i][0].transcript, e.results[i].isFinal);
    };
    r.onerror = (e: any) => this.cb.onStatus("speech error: " + e.error);
    r.onend = () => { if (this.running) r.start(); }; // keep listening
    this.running = true;
    r.start();
    this.cb.onStatus(local ? "listening (Web Speech, on-device)" : "listening (Web Speech – may use a cloud service)");
  }
  async stop() {
    this.running = false;
    this.r?.stop();
  }
}

class NativeAsr implements Asr {
  private plugin: any;
  private running = false;
  constructor(private lang: string, private cb: AsrCallbacks) {}
  async start() {
    const { SpeechRecognition } = await import("@capacitor-community/speech-recognition");
    this.plugin = SpeechRecognition;
    if (!(await SpeechRecognition.available()).available) throw new Error("native speech recognition unavailable");
    await SpeechRecognition.requestPermissions();
    const tracker = new WordTracker(this.cb);
    await SpeechRecognition.removeAllListeners();
    await SpeechRecognition.addListener("partialResults", (d: { matches: string[] }) => d.matches?.[0] && tracker.hypothesis(d.matches[0], false));
    await SpeechRecognition.addListener("listeningState", async (d: { status: string }) => {
      if (d.status === "stopped" && this.running) {
        tracker.hypothesis("", true);
        await this.listen(); // mobile recognisers stop after each utterance
      }
    });
    this.running = true;
    await this.listen();
    this.cb.onStatus("listening (native on-device recogniser)");
  }
  private async listen() {
    await this.plugin.start({ language: this.lang === "it" ? "it-IT" : "en-US", partialResults: true, popup: false, maxResults: 1 });
  }
  async stop() {
    this.running = false;
    await this.plugin?.stop();
  }
}

export function defaultAsr(): AsrKind {
  if (Capacitor.isNativePlatform()) return "native";
  return "vosk";
}

export function createAsr(kind: AsrKind, lang: string, cb: AsrCallbacks, voskUrl?: string): Asr | null {
  switch (kind) {
    case "vosk": return new VoskAsr(lang, cb, voskUrl);
    case "web": return new WebSpeechAsr(lang, cb);
    case "native": return new NativeAsr(lang, cb);
    case "text": return null;
  }
}
