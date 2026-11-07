import { VERT, FRAG } from "./shader";
import { NUM_CONCEPTS } from "../nn/text";
import type { Interpretation } from "../nn/fusion";
import type { AudioFeatures } from "../audio/mic";

/**
 * Scene director: turns the per-word network outputs into a temporally
 * coherent scene. Concepts fade in fast (attack) and out slowly (release);
 * only the top-K concepts above a threshold are shown, with hysteresis, so
 * the video follows the topic of the speech without flickering.
 */
export class Director {
  readonly target = new Float32Array(NUM_CONCEPTS);
  readonly weights = new Float32Array(NUM_CONCEPTS);
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

  /** per-frame smoothing */
  tick(dt: number, audio: AudioFeatures) {
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

export class Renderer {
  private gl: WebGL2RenderingContext;
  private prog: WebGLProgram;
  private u: Record<string, WebGLUniformLocation | null> = {};
  private flash = 0;
  quality = 4; // fbm octaves
  scale = 1;   // render resolution scale

  constructor(private canvas: HTMLCanvasElement) {
    const gl = canvas.getContext("webgl2", { antialias: false, powerPreference: "high-performance" });
    if (!gl) throw new Error("WebGL2 is required");
    this.gl = gl;
    const sh = (type: number, src: string) => {
      const s = gl.createShader(type)!;
      gl.shaderSource(s, src);
      gl.compileShader(s);
      if (!gl.getShaderParameter(s, gl.COMPILE_STATUS)) throw new Error(gl.getShaderInfoLog(s) ?? "shader error");
      return s;
    };
    const p = (this.prog = gl.createProgram()!);
    gl.attachShader(p, sh(gl.VERTEX_SHADER, VERT));
    gl.attachShader(p, sh(gl.FRAGMENT_SHADER, FRAG));
    gl.linkProgram(p);
    if (!gl.getProgramParameter(p, gl.LINK_STATUS)) throw new Error(gl.getProgramInfoLog(p) ?? "link error");
    const buf = gl.createBuffer();
    gl.bindBuffer(gl.ARRAY_BUFFER, buf);
    gl.bufferData(gl.ARRAY_BUFFER, new Float32Array([-1, -1, 1, -1, -1, 1, 1, 1]), gl.STATIC_DRAW);
    const loc = gl.getAttribLocation(p, "p");
    gl.enableVertexAttribArray(loc);
    gl.vertexAttribPointer(loc, 2, gl.FLOAT, false, 0, 0);
    for (const n of ["R", "t", "C", "val", "aro", "au", "Q", "flash"]) this.u[n] = gl.getUniformLocation(p, n);
  }

  resize() {
    const dpr = Math.min(2, window.devicePixelRatio || 1) * this.scale;
    const w = Math.max(1, Math.round(this.canvas.clientWidth * dpr));
    const h = Math.max(1, Math.round(this.canvas.clientHeight * dpr));
    if (this.canvas.width !== w || this.canvas.height !== h) {
      this.canvas.width = w;
      this.canvas.height = h;
    }
  }

  draw(time: number, d: Director, audio: AudioFeatures) {
    const gl = this.gl;
    this.resize();
    gl.viewport(0, 0, this.canvas.width, this.canvas.height);
    gl.useProgram(this.prog);
    // random lightning strikes while a storm is on screen
    this.flash *= 0.85;
    if (d.weights[14] > 0.3 && Math.random() < 0.006 + audio.flux * 0.02) this.flash = 1;
    gl.uniform2f(this.u.R, this.canvas.width, this.canvas.height);
    gl.uniform1f(this.u.t, time);
    gl.uniform1fv(this.u.C, d.weights);
    gl.uniform1f(this.u.val, d.valence);
    gl.uniform1f(this.u.aro, d.arousal);
    gl.uniform4f(this.u.au, audio.rms, audio.low, audio.mid, audio.flux);
    gl.uniform1f(this.u.Q, this.quality);
    gl.uniform1f(this.u.flash, this.flash);
    gl.drawArrays(gl.TRIANGLE_STRIP, 0, 4);
  }
}
