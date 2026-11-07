// Quantised tensor storage + the two kernels both networks need.
//
// Layout: a 2-D [rows, cols] matrix; quantisation groups run along `cols`.
//   f32 : plain floats
//   f16 : IEEE half, decoded through a 64k lookup table (half the memory)
//   q8  : int8, one fp32 scale per row        (per-output-channel)
//   q4  : int4 packed 2/byte (offset +8), one fp32 scale per 32 columns
//
// Weights stay quantised in memory; they are dequantised on the fly inside
// the kernels (weight-only quantisation, fp32 accumulation).

export type DType = "f32" | "f16" | "q8" | "q4";

let HALF: Float32Array | null = null;
function halfTable(): Float32Array {
  if (HALF) return HALF;
  HALF = new Float32Array(65536);
  for (let h = 0; h < 65536; h++) {
    const s = h & 0x8000 ? -1 : 1;
    const e = (h >> 10) & 0x1f;
    const m = h & 0x3ff;
    HALF[h] = e === 0 ? s * m * 2 ** -24 : e === 31 ? (m ? NaN : s * Infinity) : s * (1 + m / 1024) * 2 ** (e - 15);
  }
  return HALF;
}

export class QTensor {
  readonly rows: number;
  readonly cols: number;
  private f32?: Float32Array;
  private f16?: Uint16Array;
  private i8?: Int8Array;
  private u4?: Uint8Array;
  private scales?: Float32Array;
  private group: number;
  private ngroups: number;

  constructor(
    readonly dtype: DType,
    shape: number[],
    data: ArrayBuffer,
    offset: number,
    length: number,
    scales?: Float32Array,
    group?: number,
  ) {
    this.rows = shape[0];
    this.cols = shape.length > 1 ? shape[1] : 1;
    this.group = group ?? this.cols;
    this.ngroups = Math.ceil(this.cols / this.group);
    this.scales = scales;
    switch (dtype) {
      case "f32": this.f32 = new Float32Array(data, offset, length / 4); break;
      case "f16": this.f16 = new Uint16Array(data, offset, length / 2); halfTable(); break;
      case "q8": this.i8 = new Int8Array(data, offset, length); break;
      case "q4": this.u4 = new Uint8Array(data, offset, length); break;
    }
  }

  /** Bytes used by weights + scales. */
  get bytes(): number {
    const d = this.f32?.byteLength ?? this.f16?.byteLength ?? this.i8?.byteLength ?? this.u4?.byteLength ?? 0;
    return d + (this.scales?.byteLength ?? 0);
  }

  /** Plain fp32 view of a 1-D / f32 tensor. */
  vector(): Float32Array {
    if (!this.f32) throw new Error("vector() needs an f32 tensor");
    return this.f32;
  }

  /** dot(W[r, :], x[xOff : xOff + cols]) */
  rowDot(r: number, x: Float32Array, xOff = 0): number {
    const C = this.cols;
    const base = r * C;
    let acc = 0;
    switch (this.dtype) {
      case "f32": {
        const w = this.f32!;
        for (let c = 0; c < C; c++) acc += w[base + c] * x[xOff + c];
        return acc;
      }
      case "f16": {
        const w = this.f16!, t = HALF!;
        for (let c = 0; c < C; c++) acc += t[w[base + c]] * x[xOff + c];
        return acc;
      }
      case "q8": {
        const w = this.i8!;
        for (let c = 0; c < C; c++) acc += w[base + c] * x[xOff + c];
        return acc * this.scales![r];
      }
      case "q4": {
        const w = this.u4!, s = this.scales!, G = this.group;
        for (let g = 0; g < this.ngroups; g++) {
          let part = 0;
          const end = Math.min(C, (g + 1) * G);
          for (let c = g * G; c < end; c++) {
            const i = base + c;
            const nib = i & 1 ? w[i >> 1] >> 4 : w[i >> 1] & 15;
            part += (nib - 8) * x[xOff + c];
          }
          acc += part * s[r * this.ngroups + g];
        }
        return acc;
      }
    }
  }

  /** out[outOff + c] += k * W[r, c]   (event-driven / embedding gather kernel) */
  addRow(r: number, out: Float32Array, k = 1, outOff = 0): void {
    const C = this.cols;
    const base = r * C;
    switch (this.dtype) {
      case "f32": {
        const w = this.f32!;
        for (let c = 0; c < C; c++) out[outOff + c] += k * w[base + c];
        return;
      }
      case "f16": {
        const w = this.f16!, t = HALF!;
        for (let c = 0; c < C; c++) out[outOff + c] += k * t[w[base + c]];
        return;
      }
      case "q8": {
        const w = this.i8!;
        const ks = k * this.scales![r];
        for (let c = 0; c < C; c++) out[outOff + c] += ks * w[base + c];
        return;
      }
      case "q4": {
        const w = this.u4!, s = this.scales!, G = this.group;
        for (let g = 0; g < this.ngroups; g++) {
          const ks = k * s[r * this.ngroups + g];
          const end = Math.min(C, (g + 1) * G);
          for (let c = g * G; c < end; c++) {
            const i = base + c;
            const nib = i & 1 ? w[i >> 1] >> 4 : w[i >> 1] & 15;
            out[outOff + c] += ks * (nib - 8);
          }
        }
        return;
      }
    }
  }

  /** y = W x + b for W [out, in] */
  matvec(x: Float32Array, y: Float32Array, bias?: Float32Array, xOff = 0, yOff = 0): void {
    for (let r = 0; r < this.rows; r++) y[yOff + r] = this.rowDot(r, x, xOff) + (bias ? bias[r] : 0);
  }
}

export interface TensorEntry {
  shape: number[];
  dtype: DType;
  offset: number;
  length: number;
  group?: number;
  scaleOffset?: number;
  scaleLength?: number;
}

export interface Manifest {
  format: string;
  model: "ann" | "snn";
  tier: string;
  config: Record<string, number>;
  knobs: { ann_window: number; snn_steps: number };
  constants: {
    concepts: string[];
    annBuckets: number;
    annMaxLen: number;
    snnAfferents: number;
    snnWordSpikes: number;
    snnBigramSpikes: number;
    snnWords: number;
  };
  tensors: Record<string, TensorEntry>;
  bytes: number;
  file: string;
}

export type Weights = Record<string, QTensor>;

export function parseWeights(m: Manifest, buf: ArrayBuffer): Weights {
  const out: Weights = {};
  for (const [name, e] of Object.entries(m.tensors)) {
    const scales = e.scaleOffset !== undefined ? new Float32Array(buf, e.scaleOffset, e.scaleLength! / 4) : undefined;
    out[name] = new QTensor(e.dtype, e.shape, buf, e.offset, e.length, scales, e.group);
  }
  return out;
}
