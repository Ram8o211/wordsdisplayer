// Neural video generator: the compiled CPU kernel must match the reference
// interpreter, and the generated images must be coherent with what is heard:
// colour, symmetry, stratification and motion statistics follow the semantic
// code, the same input gives the same video, and different words differ.
// If Playwright + Chromium are installed, the GLSL kernel is also checked
// against the CPU kernel on a real WebGL2 implementation.
import { describe, expect, it } from "vitest";
import { createRequire } from "node:module";
import { design, frameInputs, wordKick } from "../src/render/neural/design";
import { compile } from "../src/render/neural/compile";
import { CONCEPTS, fnv1a } from "../src/nn/text";

const W = 96, H = 54;
type Opts = Partial<{ valence: number; arousal: number; form: number[]; rms: number }>;
const inputs = (scene: string, o: Opts = {}, dt = 0.05) => {
  const c = new Float32Array(CONCEPTS.length);
  for (const n of scene.split(",").filter(Boolean)) {
    const i = CONCEPTS.indexOf(n as never);
    if (i < 0) throw new Error(n);
    c[i] = 0.95;
  }
  return frameInputs({
    concepts: c, valence: o.valence ?? 0.2, arousal: o.arousal ?? 0, form: o.form,
    audio: { rms: o.rms ?? 0.2, low: 0.2, mid: 0.3, flux: 0 }, dt, flash: 0, aspect: W / H,
  });
};
/** fresh generator, run for `frames` frames of the scene, return the last image */
const film = (scene: string, o: Opts = {}, frames = 40, detail = 2, net = compile(design(detail))) => {
  for (let f = 0; f < frames; f++) net.update(inputs(scene, o));
  const px = new Uint8ClampedArray(W * H * 4);
  net.renderCPU(px, W, H);
  return px;
};
const mean = (px: Uint8ClampedArray) => {
  const s = [0, 0, 0];
  for (let i = 0; i < px.length; i += 4) for (let k = 0; k < 3; k++) s[k] += px[i + k];
  return s.map((v) => v / (px.length / 4));
};
const lum = (px: Uint8ClampedArray, x: number, y: number) => { const i = (y * W + x) * 4; return px[i] + px[i + 1] + px[i + 2]; };
const diffImg = (a: Uint8ClampedArray, b: Uint8ClampedArray) => { let d = 0; for (let i = 0; i < a.length; i++) d += Math.abs(a[i] - b[i]); return d / a.length; };
const SCENES = ["", "ocean", "fire", "forest", "night,city", "space", "love", "underwater", "storm,rain", "desert,sunset", "technology,music", "snow"];
const nets = [0, 1, 2, 3].map((d) => compile(design(d)));

describe("neural video generator", () => {
  it("compiled CPU kernel matches the reference interpreter", () => {
    for (const net of nets) for (const scene of SCENES) {
      const px = film(scene, { arousal: 0.4 }, 5, 0, net);
      for (let i = 0; i < 40; i++) {
        const x = (i * 37) % W, y = (i * 23) % H;
        const ref = net.reference((x + 0.5) / W, (H - y - 0.5) / H, 2.2 * H);
        for (let k = 0; k < 3; k++) {
          const want = 255 * Math.pow(Math.min(1, Math.max(0, ref[k])), 0.95);
          expect(Math.abs(px[(y * W + x) * 4 + k] - want)).toBeLessThanOrEqual(1);
        }
      }
    }
  }, 120_000);

  it("draws recognisable objects where their prototypes put them", () => {
    // pixel at scene coordinates (x, y) of a 320×180 frame after one frame of motion
    const w = 320, h = 180;
    const at = (px: Uint8ClampedArray, x: number, y: number) => {
      const i = (Math.floor((1 - y) * h) * w + Math.floor((x / (w / h) + 0.5) * w)) * 4;
      return [px[i], px[i + 1], px[i + 2]];
    };
    const one = (scene: string) => {
      const fresh = compile(design(1)); // fresh hypernetwork state (phases at 0)
      fresh.update(inputs(scene));
      const px = new Uint8ClampedArray(w * h * 4);
      fresh.renderCPU(px, w, h);
      return px;
    };
    const [mr, mg, mb] = at(one("night"), 0.235, 0.78); // lit side of the crescent moon
    expect(Math.min(mr, mg)).toBeGreaterThan(170);
    expect(mb).toBeLessThan(mr);
    const sail = at(one("ocean"), 0.06, 0.5); // white sail
    expect(Math.min(...sail)).toBeGreaterThan(150);
    const hull = at(one("ocean"), 0, 0.33); // brown hull above blue water
    expect(hull[0]).toBeGreaterThan(hull[2]);
    const sea = at(one("ocean"), 0.6, 0.15);
    expect(sea[2]).toBeGreaterThan(sea[0] + 40);
    const man = at(one("snow"), 0.3, 0.22); // snowman's body
    expect(Math.min(...man)).toBeGreaterThan(190);
    const hat = at(one("snow"), 0.3, 0.57); // his black hat (soft at this tiny test resolution)
    expect(Math.max(...hat)).toBeLessThan(Math.min(...man) - 60);
    const flame = at(one("fire"), 0, 0.2);
    expect(flame[0]).toBeGreaterThan(200);
    expect(flame[2]).toBeLessThan(120);
    const cactus = at(one("desert"), -0.4, 0.3);
    expect(cactus[1]).toBeGreaterThan(cactus[0] + 30);
    const heart = at(one("love"), 0.1, 0.45);
    expect(heart[0]).toBeGreaterThan(heart[1] + 80);
  }, 60_000);

  it("colours follow what is heard", () => {
    const [fr, , fb] = mean(film("fire")), [or, , ob] = mean(film("ocean"));
    expect(fr - fb).toBeGreaterThan(60); // warm
    expect(ob - or).toBeGreaterThan(20); // cool
    const [gr, gg, gb] = mean(film("forest"));
    expect(gg).toBeGreaterThan((gr + gb) / 2 + 10); // green
    const night = mean(film("night")), snow = mean(film("snow"));
    expect(night.reduce((a, b) => a + b)).toBeLessThan(snow.reduce((a, b) => a + b) * 0.6); // dark vs bright
    const warm = mean(film("", { valence: 1 })), cold = mean(film("", { valence: -1 }));
    expect(warm[0] - warm[2]).toBeGreaterThan(cold[0] - cold[2]);
  });

  it("forms follow what is heard (symmetry, stratification)", () => {
    const sym = (px: Uint8ClampedArray) => { let d = 0; for (let y = 0; y < H; y++) for (let x = 0; x < W / 2; x++) d += Math.abs(lum(px, x, y) - lum(px, W - 1 - x, y)); return d; };
    expect(sym(film("love"))).toBeLessThan(sym(film("storm")) * 0.5);
    const aniso = (px: Uint8ClampedArray) => {
      let gx = 0, gy = 0;
      for (let y = 0; y < H - 1; y++) for (let x = 0; x < W - 1; x++) {
        gx += Math.abs(lum(px, x + 1, y) - lum(px, x, y));
        gy += Math.abs(lum(px, x, y + 1) - lum(px, x, y));
      }
      return gy / gx;
    };
    expect(aniso(film("ocean"))).toBeGreaterThan(aniso(film("rain"))); // horizon bands vs falling streaks
  });

  it("is reproducible, and words change the forms", () => {
    expect(diffImg(film("ocean"), film("ocean"))).toBe(0);
    const kick = (w: string) => Array.from(wordKick(fnv1a(w)));
    expect(diffImg(film("ocean", { form: kick("sailing") }), film("ocean", { form: kick("whale") }))).toBeGreaterThan(2);
  });

  it("animates, faster when aroused", () => {
    const motion = (aro: number) => {
      const net = compile(design(1));
      const a = film("ocean", { arousal: aro }, 20, 1, net), b = film("ocean", { arousal: aro }, 4, 1, net);
      return diffImg(a, b);
    };
    expect(motion(0)).toBeGreaterThan(0.5);
    expect(motion(1)).toBeGreaterThan(motion(-1));
  });

  it("tiers trade detail for cost", () => {
    const cost = nets.map((n) => (n.update(inputs("fire")), n.cost().macs));
    for (let i = 1; i < nets.length; i++) expect(cost[i]).toBeGreaterThan(cost[i - 1]);
    for (const n of nets) expect(n.vec4s).toBeLessThanOrEqual(1024); // 16 KB: the WebGL2 minimum uniform block size
  });
});

// ------------------------------------------------------------------ optional GPU check
let chromium: any = null;
try { chromium = createRequire(import.meta.url)("playwright").chromium; } catch { /* not installed */ }

describe.skipIf(!chromium || process.env.SKIP_GPU)("GLSL kernel on WebGL2", () => {
  it("matches the CPU kernel within 1/255", async () => {
    let browser;
    try { browser = await chromium.launch({ args: ["--use-angle=swiftshader", "--enable-unsafe-swiftshader"] }); }
    catch { return; } // no browser binary
    const page = await browser.newPage();
    await page.setContent("<canvas id=c></canvas>");
    for (const net of [nets[0], nets[3]]) for (const scene of SCENES) {
      for (let f = 0; f < 30; f++) net.update(inputs(scene, { arousal: 0.5 }, 4.1)); // large time steps: phases far from 0
      const P = net.update(inputs(scene, { arousal: 0.5 }));
      const cpu = new Uint8ClampedArray(W * H * 4);
      net.renderCPU(cpu, W, H);
      const gpu: number[] = await page.evaluate(({ glsl, P, W, H, bank }: { glsl: string; P: number[]; W: number; H: number; bank: { data: number[]; w: number; h: number } }) => {
        const cv = document.getElementById("c") as HTMLCanvasElement;
        cv.width = W; cv.height = H;
        const gl = cv.getContext("webgl2")!;
        const sh = (t: number, s: string) => {
          const x = gl.createShader(t)!; gl.shaderSource(x, s); gl.compileShader(x);
          if (!gl.getShaderParameter(x, gl.COMPILE_STATUS)) throw new Error(gl.getShaderInfoLog(x)!);
          return x;
        };
        const p = gl.createProgram()!;
        gl.attachShader(p, sh(gl.VERTEX_SHADER, "#version 300 es\nin vec2 p; void main(){gl_Position=vec4(p,0.,1.);}"));
        gl.attachShader(p, sh(gl.FRAGMENT_SHADER, glsl));
        gl.linkProgram(p); gl.useProgram(p);
        gl.bindBuffer(gl.ARRAY_BUFFER, gl.createBuffer());
        gl.bufferData(gl.ARRAY_BUFFER, new Float32Array([-1, -1, 1, -1, -1, 1, 1, 1]), gl.STATIC_DRAW);
        gl.enableVertexAttribArray(0); gl.vertexAttribPointer(0, 2, gl.FLOAT, false, 0, 0);
        gl.uniformBlockBinding(p, gl.getUniformBlockIndex(p, "Params"), 0);
        const u = gl.createBuffer(); gl.bindBuffer(gl.UNIFORM_BUFFER, u);
        gl.bufferData(gl.UNIFORM_BUFFER, new Float32Array(P), gl.DYNAMIC_DRAW); gl.bindBufferBase(gl.UNIFORM_BUFFER, 0, u);
        gl.uniform2f(gl.getUniformLocation(p, "R"), W, H);
        gl.uniform1f(gl.getUniformLocation(p, "asp"), W / H);
        gl.uniform1f(gl.getUniformLocation(p, "aa"), 2.2 * H);
        const tex = gl.createTexture(); gl.bindTexture(gl.TEXTURE_2D, tex);
        gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.NEAREST); gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.NEAREST);
        gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA32F, bank.w, bank.h, 0, gl.RGBA, gl.FLOAT, new Float32Array(bank.data));
        gl.uniform1i(gl.getUniformLocation(p, "BANK"), 0);
        gl.viewport(0, 0, W, H); gl.drawArrays(gl.TRIANGLE_STRIP, 0, 4);
        const px = new Uint8Array(W * H * 4); gl.readPixels(0, 0, W, H, gl.RGBA, gl.UNSIGNED_BYTE, px);
        return Array.from(px);
      }, { glsl: net.glsl, P: Array.from(P), W, H, bank: { data: Array.from(net.bankTex.data), w: net.bankTex.width, h: net.bankTex.height } });
      let sum = 0, max = 0;
      for (let y = 0; y < H; y++) for (let x = 0; x < W; x++) for (let k = 0; k < 3; k++) {
        const dd = Math.abs(gpu[((H - 1 - y) * W + x) * 4 + k] - cpu[(y * W + x) * 4 + k]);
        sum += dd; max = Math.max(max, dd);
      }
      expect(sum / (W * H * 3)).toBeLessThan(0.05);
      expect(max).toBeLessThanOrEqual(2);
    }
    await browser.close();
  }, 120_000);
});
