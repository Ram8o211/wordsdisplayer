import { NUM_CONCEPTS } from "../nn/text";
import type { Interpretation } from "../nn/fusion";
import type { AudioFeatures } from "../audio/mic";
import { design, frameInputs, tempo, wordKick, FORM_DIM } from "./neural/design";
import { fnv1a, tokenize } from "../nn/text";
import { compile, type Compiled, type CostStats } from "./neural/compile";

/**
 * Scene director: turns the per-word network outputs into a temporally
 * coherent conditioning code for the video generator. Concepts fade in fast
 * (attack) and out slowly (release); only the top-K concepts above a threshold
 * are kept, with hysteresis, so the video follows the topic of the speech
 * without flickering.
 */
export class Director {
  readonly target = new Float32Array(NUM_CONCEPTS);
  readonly weights = new Float32Array(NUM_CONCEPTS);
  /** heard-word latent: each word kicks the generator's form latent, then fades (τ ≈ 4 s) */
  readonly form = new Float32Array(FORM_DIM);
  valence = 0;
  arousal = 0;
  private tv = 0;
  private ta = 0;
  constructor(public maxLayers = 4, public threshold = 0.35) {}

  update(r: Interpretation) {
    const order = [...r.probs.keys()].sort((a, b) => r.probs[b] - r.probs[a]);
    const next = new Float32Array(NUM_CONCEPTS);
    let any = false;
    for (let k = 0; k < this.maxLayers; k++) {
      const i = order[k];
      // hysteresis: a visible concept stays with a lower threshold
      const thr = this.weights[i] > 0.2 ? this.threshold * 0.6 : this.threshold;
      if (r.probs[i] >= thr) { next[i] = Math.min(1, 0.25 + r.probs[i]); any = true; }
    }
    // small talk with nothing to show keeps the current scene on screen
    if (any) this.target.set(next);
    this.tv = r.valence;
    this.ta = r.arousal;
  }

  /** every heard word leaves a reproducible trace in the generated forms */
  hear(text: string) {
    for (const w of tokenize(text)) {
      if (w.length < 3) continue; // function words barely move it
      const k = wordKick(fnv1a(w));
      for (let i = 0; i < FORM_DIM; i++) this.form[i] = this.form[i] * 0.75 + k[i] * 0.45;
    }
  }

  /** per-frame smoothing */
  tick(dt: number, audio: AudioFeatures) {
    const f = Math.exp(-dt / 4);
    for (let i = 0; i < FORM_DIM; i++) this.form[i] *= f;
    for (let i = 0; i < NUM_CONCEPTS; i++) {
      const up = this.target[i] > this.weights[i];
      const k = 1 - Math.exp(-dt * (up ? 1.6 : 0.35));
      this.weights[i] += (this.target[i] - this.weights[i]) * k;
    }
    const k = 1 - Math.exp(-dt * 0.8);
    this.valence += (this.tv - this.valence) * k;
    // vocal energy raises arousal a little (non-learned modulation)
    this.arousal += (Math.max(-1, Math.min(1, this.ta + 0.6 * (audio.rms - 0.25))) - this.arousal) * k;
  }
}

const VERT = `#version 300 es
in vec2 p;
void main(){ gl_Position = vec4(p,0.,1.); }`;

/**
 * Draws every pixel with the assigned-weight neural generator (neural/design.ts).
 * WebGL2: the compiled network runs as a fragment shader, its per-frame
 * weights in a uniform buffer. Without WebGL2 the same network runs as a
 * compiled JS kernel at low resolution (canvas 2D upscales it).
 */
export class Renderer {
  private gl: WebGL2RenderingContext | null = null;
  private ctx2d: CanvasRenderingContext2D | null = null;
  private prog: WebGLProgram | null = null;
  private ubo: WebGLBuffer | null = null;
  private u: Record<string, WebGLUniformLocation | null> = {};
  private flash = 0;
  private lastTime = -1;
  private image: ImageData | null = null;
  net!: Compiled;
  detail = -1;
  scale = 1; // render resolution scale
  readonly backend: "webgl2" | "cpu";

  constructor(private canvas: HTMLCanvasElement, detail = 2) {
    this.gl = canvas.getContext("webgl2", { antialias: false, powerPreference: "high-performance" });
    this.backend = this.gl ? "webgl2" : "cpu";
    if (!this.gl) this.ctx2d = canvas.getContext("2d");
    this.setDetail(detail);
  }

  /** (Re)build the generator for a tier: 0 nano … 3 ultra. */
  setDetail(detail: number) {
    if (detail === this.detail) return;
    this.detail = detail;
    // the CPU fallback always runs the smallest network
    this.net = compile(design(this.gl ? detail : 0));
    const gl = this.gl;
    if (!gl) return;
    const sh = (type: number, src: string) => {
      const s = gl.createShader(type)!;
      gl.shaderSource(s, src);
      gl.compileShader(s);
      if (!gl.getShaderParameter(s, gl.COMPILE_STATUS)) throw new Error(gl.getShaderInfoLog(s) ?? "shader error");
      return s;
    };
    if (this.prog) gl.deleteProgram(this.prog);
    const p = (this.prog = gl.createProgram()!);
    gl.attachShader(p, sh(gl.VERTEX_SHADER, VERT));
    gl.attachShader(p, sh(gl.FRAGMENT_SHADER, this.net.glsl));
    gl.linkProgram(p);
    if (!gl.getProgramParameter(p, gl.LINK_STATUS)) throw new Error(gl.getProgramInfoLog(p) ?? "link error");
    const buf = gl.createBuffer();
    gl.bindBuffer(gl.ARRAY_BUFFER, buf);
    gl.bufferData(gl.ARRAY_BUFFER, new Float32Array([-1, -1, 1, -1, -1, 1, 1, 1]), gl.STATIC_DRAW);
    const loc = gl.getAttribLocation(p, "p");
    gl.enableVertexAttribArray(loc);
    gl.vertexAttribPointer(loc, 2, gl.FLOAT, false, 0, 0);
    for (const n of ["R", "asp", "aa"]) this.u[n] = gl.getUniformLocation(p, n);
    gl.uniformBlockBinding(p, gl.getUniformBlockIndex(p, "Params"), 0);
    this.ubo ??= gl.createBuffer();
    gl.bindBuffer(gl.UNIFORM_BUFFER, this.ubo);
    gl.bufferData(gl.UNIFORM_BUFFER, this.net.vec4s * 16, gl.DYNAMIC_DRAW);
    gl.bindBufferBase(gl.UNIFORM_BUFFER, 0, this.ubo);
  }

  resize() {
    const cpu = !this.gl;
    const dpr = cpu ? 1 : Math.min(2, window.devicePixelRatio || 1) * this.scale;
    // CPU fallback: ~200 px wide at full scale (dynamic resolution still applies), upscaled by the browser
    const k = cpu ? Math.min(1, Math.max(64, 200 * this.scale) / Math.max(1, this.canvas.clientWidth)) : dpr;
    const w = Math.max(1, Math.round(this.canvas.clientWidth * k));
    const h = Math.max(1, Math.round(this.canvas.clientHeight * k));
    if (this.canvas.width !== w || this.canvas.height !== h) {
      this.canvas.width = w;
      this.canvas.height = h;
    }
  }

  /** per-pixel cost of the frame just drawn */
  cost(): CostStats { return this.net.cost(); }

  draw(time: number, d: Director, audio: AudioFeatures) {
    const dt = this.lastTime < 0 ? 0 : Math.min(0.1, time - this.lastTime);
    this.lastTime = time;
    // random lightning strikes while a storm is on screen
    this.flash *= 0.85;
    if (d.weights[14] > 0.3 && Math.random() < 0.006 + audio.flux * 0.02) this.flash = 1;
    this.resize();
    const W = this.canvas.width, H = this.canvas.height;
    const P = this.net.update(frameInputs({
      concepts: d.weights, valence: d.valence, arousal: d.arousal, audio, form: d.form,
      dt: dt * tempo(d.arousal), flash: this.flash, aspect: W / H, // arousal sets the animation tempo
    }));
    const gl = this.gl;
    if (!gl) {
      if (!this.ctx2d) return;
      if (!this.image || this.image.width !== W || this.image.height !== H) this.image = this.ctx2d.createImageData(W, H);
      this.net.renderCPU(this.image.data, W, H);
      this.ctx2d.putImageData(this.image, 0, 0);
      return;
    }
    gl.viewport(0, 0, W, H);
    gl.useProgram(this.prog);
    gl.bindBuffer(gl.UNIFORM_BUFFER, this.ubo);
    gl.bufferSubData(gl.UNIFORM_BUFFER, 0, P);
    gl.uniform2f(this.u.R, W, H);
    gl.uniform1f(this.u.asp, W / H);
    gl.uniform1f(this.u.aa, 2.2 * H); // edge neurons: ~1.5 px transition at any resolution
    gl.drawArrays(gl.TRIANGLE_STRIP, 0, 4);
  }
}
