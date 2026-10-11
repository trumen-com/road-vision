// YOLO11n (COCO) detector with NMS baked into the ONNX graph: output [1, 300, 6] = x1, y1, x2, y2, score, class.
// The ONNX runtime is injected so the same code runs in the browser (onnxruntime-web) and in Node (onnxruntime-node).
import { Detection, IN_H, IN_W, ROAD_USERS } from "./types";

export interface OrtLike {
  Tensor: new (type: "float32", data: Float32Array, dims: number[]) => unknown;
}
export interface SessionLike {
  inputNames: readonly string[];
  run(feeds: Record<string, unknown>): Promise<Record<string, { data: unknown }>>;
}

export class Detector {
  private input: Float32Array;
  constructor(private ort: OrtLike, private session: SessionLike, public minScore = 0.25) {
    this.input = new Float32Array(3 * IN_W * IN_H);
  }

  /** pixels: IN_W x IN_H image, RGB (3 channels) or RGBA (4 channels), row-major. */
  async detect(pixels: Uint8Array | Uint8ClampedArray, channels: 3 | 4): Promise<Detection[]> {
    const n = IN_W * IN_H, inp = this.input;
    for (let i = 0, p = 0; i < n; i++, p += channels) {
      inp[i] = pixels[p] / 255;
      inp[i + n] = pixels[p + 1] / 255;
      inp[i + 2 * n] = pixels[p + 2] / 255;
    }
    const tensor = new this.ort.Tensor("float32", inp, [1, 3, IN_H, IN_W]);
    const out = await this.session.run({ [this.session.inputNames[0]]: tensor });
    const data = Object.values(out)[0].data as Float32Array;
    const dets: Detection[] = [];
    for (let k = 0; k + 5 < data.length; k += 6) {
      const score = data[k + 4];
      if (score < this.minScore) continue;
      const cls = Math.round(data[k + 5]);
      if (!ROAD_USERS.has(cls)) continue;
      dets.push({ x1: data[k], y1: data[k + 1], x2: data[k + 2], y2: data[k + 3], score, cls });
    }
    return dets;
  }
}
