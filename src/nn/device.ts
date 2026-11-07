// Hardware profiler -> precision tier.
//
// Tiers (produced by training/export.py with sensitivity-driven mixed precision):
//   nano    : int4/int8 mix (~55% of int8 size), ANN window 16 words, SNN T=2
//   mobile  : int4/int8/fp16 mix (int8 budget), window 24, T=4
//   desktop : int8/fp16 mix (fp16 budget),      window 32, T=6
//   ultra   : fp32,                             window 48, T=8

export const TIERS = ["nano", "mobile", "desktop", "ultra"] as const;
export type Tier = (typeof TIERS)[number];

export interface DeviceProfile {
  cores: number;
  memoryGB: number | null;
  mobile: boolean;
  gpu: string;
  mflops: number; // measured scalar JS throughput
  score: number;
  tier: Tier;
  renderScale: number;
}

function benchmark(ms = 60): number {
  const n = 128;
  const a = new Float32Array(n * n).map(() => Math.random());
  const x = new Float32Array(n).map(() => Math.random());
  const y = new Float32Array(n);
  let ops = 0;
  const t0 = performance.now();
  while (performance.now() - t0 < ms) {
    for (let r = 0; r < n; r++) {
      let s = 0;
      for (let c = 0; c < n; c++) s += a[r * n + c] * x[c];
      y[r] = s;
    }
    ops += 2 * n * n;
  }
  return ops / ((performance.now() - t0) * 1000) + y[0] * 0; // MFLOP/s
}

function gpuName(): string {
  try {
    if (typeof document === "undefined") return "unknown";
    const gl = document.createElement("canvas").getContext("webgl2");
    if (!gl) return "no-webgl2";
    const ext = gl.getExtension("WEBGL_debug_renderer_info");
    return String(ext ? gl.getParameter(ext.UNMASKED_RENDERER_WEBGL) : gl.getParameter(gl.RENDERER));
  } catch {
    return "unknown";
  }
}

export function profileDevice(): DeviceProfile {
  const nav: any = typeof navigator !== "undefined" ? navigator : {};
  const cores = nav.hardwareConcurrency ?? 4;
  const memoryGB = typeof nav.deviceMemory === "number" ? nav.deviceMemory : null;
  const ua = String(nav.userAgent ?? "");
  const mobile = nav.userAgentData?.mobile ?? /Android|iPhone|iPad|Mobile/i.test(ua);
  const gpu = gpuName();
  const mflops = benchmark();
  // Score: measured speed dominates, cores/memory/GPU class refine it.
  let score = Math.log2(Math.max(1, mflops)) + Math.log2(cores) * 0.5 + (memoryGB ? Math.log2(memoryGB) * 0.5 : 0.5);
  if (/Apple M\d|RTX|Radeon RX|Arc/i.test(gpu)) score += 1;
  if (/Mali-[GT]\d{2}\b|Adreno \(TM\) [3-5]\d\d|PowerVR|SwiftShader|llvmpipe/i.test(gpu)) score -= 1;
  if (mobile) score -= 1;
  const tier: Tier = score >= 13 ? "ultra" : score >= 11.5 ? "desktop" : score >= 9.5 ? "mobile" : "nano";
  const renderScale = { nano: 0.45, mobile: 0.6, desktop: 0.85, ultra: 1 }[tier];
  return { cores, memoryGB, mobile, gpu, mflops: Math.round(mflops), score: +score.toFixed(2), tier, renderScale };
}
