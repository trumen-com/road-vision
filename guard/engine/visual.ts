// Optional visual crash-risk model: MobileNetV3-Large backbone (frozen, ImageNet) -> 1920-d embedding per frame at
// 4 fps; a small temporal head trained on the Nexar training split turns the last 8 embeddings (2 s) into a risk.
import { OrtLike, SessionLike } from "./detector";

export const VIS_W = 288, VIS_H = 160, VIS_T = 8, VIS_DIM = 1920, VIS_PERIOD = 0.25;

export class VisualRisk {
  private ring: Float32Array[] = [];
  private lastT = -Infinity;
  private input = new Float32Array(3 * VIS_W * VIS_H);
  constructor(private ort: OrtLike, private backbone: SessionLike, private head: SessionLike) {}

  reset() { this.ring = []; this.lastT = -Infinity; }

  /** Feed a VIS_W x VIS_H frame (RGB or RGBA). Returns the risk when a new estimate was computed, else null. */
  async step(pixels: Uint8Array | Uint8ClampedArray, channels: 3 | 4, t: number): Promise<number | null> {
    if (t - this.lastT < VIS_PERIOD - 1e-3) return null;
    this.lastT = t;
    const n = VIS_W * VIS_H, inp = this.input;
    for (let i = 0, p = 0; i < n; i++, p += channels) {
      inp[i] = pixels[p] / 255; inp[i + n] = pixels[p + 1] / 255; inp[i + 2 * n] = pixels[p + 2] / 255;
    }
    const emb = await this.backbone.run({ [this.backbone.inputNames[0]]: new this.ort.Tensor("float32", inp, [1, 3, VIS_H, VIS_W]) });
    this.ring.push(new Float32Array(Object.values(emb)[0].data as Float32Array));
    while (this.ring.length > VIS_T) this.ring.shift();
    const seq = new Float32Array(VIS_T * VIS_DIM);
    for (let k = 0; k < VIS_T; k++) seq.set(this.ring[Math.max(0, this.ring.length - VIS_T + k)] ?? this.ring[0], k * VIS_DIM);
    const out = await this.head.run({ [this.head.inputNames[0]]: new this.ort.Tensor("float32", seq, [1, VIS_T, VIS_DIM]) });
    return (Object.values(out)[0].data as Float32Array)[0];
  }
}
