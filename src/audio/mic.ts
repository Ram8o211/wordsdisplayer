// Microphone capture + lightweight DSP features (loudness, spectral bands,
// spectral flux). These are not learned: they modulate the arousal of the
// scene and drive audio-reactive visuals, while the semantic content comes
// from the two neural networks.

export interface AudioFeatures {
  rms: number;      // 0..1 loudness
  low: number;      // 0..1 energy bands
  mid: number;
  high: number;
  flux: number;     // onset strength
  centroid: number; // 0..1 spectral brightness
}

export class Mic {
  ctx?: AudioContext;
  stream?: MediaStream;
  source?: MediaStreamAudioSourceNode;
  private analyser?: AnalyserNode;
  private spec?: Float32Array<ArrayBuffer>;
  private prev?: Float32Array;
  readonly features: AudioFeatures = { rms: 0, low: 0, mid: 0, high: 0, flux: 0, centroid: 0 };

  async start(): Promise<void> {
    this.stream = await navigator.mediaDevices.getUserMedia({
      audio: { echoCancellation: true, noiseSuppression: true, autoGainControl: true },
    });
    this.ctx = new AudioContext();
    this.source = this.ctx.createMediaStreamSource(this.stream);
    this.analyser = this.ctx.createAnalyser();
    this.analyser.fftSize = 1024;
    this.analyser.smoothingTimeConstant = 0.6;
    this.source.connect(this.analyser);
    this.spec = new Float32Array(this.analyser.frequencyBinCount);
    this.prev = new Float32Array(this.analyser.frequencyBinCount);
  }

  stop() {
    this.stream?.getTracks().forEach((t) => t.stop());
    this.ctx?.close();
    this.ctx = undefined;
    this.analyser = undefined;
  }

  /** Call once per frame. */
  update(): AudioFeatures {
    const a = this.analyser, s = this.spec, p = this.prev;
    const f = this.features;
    if (!a || !s || !p || !this.ctx) {
      for (const k of Object.keys(f) as (keyof AudioFeatures)[]) f[k] *= 0.9;
      return f;
    }
    a.getFloatFrequencyData(s);
    const n = s.length;
    const hz = this.ctx.sampleRate / 2 / n;
    let tot = 0, lo = 0, mi = 0, hi = 0, cen = 0, flux = 0;
    for (let i = 1; i < n; i++) {
      const m = Math.pow(10, s[i] / 20); // dB -> magnitude
      const fr = i * hz;
      tot += m; cen += m * fr;
      if (fr < 300) lo += m; else if (fr < 2500) mi += m; else hi += m;
      flux += Math.max(0, m - p[i]);
      p[i] = m;
    }
    const sm = (old: number, v: number, k = 0.35) => old + (v - old) * k;
    const norm = (v: number, g: number) => Math.min(1, Math.sqrt(v) * g);
    f.rms = sm(f.rms, norm(tot, 1.2));
    f.low = sm(f.low, norm(lo, 2.5));
    f.mid = sm(f.mid, norm(mi, 1.6));
    f.high = sm(f.high, norm(hi, 3));
    f.flux = sm(f.flux, norm(flux, 2.5), 0.5);
    f.centroid = sm(f.centroid, tot > 1e-6 ? Math.min(1, cen / tot / 4000) : 0);
    return f;
  }
}
