// Browser runtime: loads the ONNX detector (WebGPU when available, else WASM) and feeds frames to the Guard engine.
import * as ort from "onnxruntime-web/webgpu";
import { Detector } from "../../engine/detector";
import { Calibrator, Guard } from "../../engine/guard";
import { TreeModel } from "../../engine/model";
import { IN_H, IN_W } from "../../engine/types";
import { VIS_H, VIS_W, VisualRisk } from "../../engine/visual";

ort.env.wasm.numThreads = Math.min(4, navigator.hardwareConcurrency || 2);

export interface Runtime {
  guard: Guard; backend: string; frame: HTMLCanvasElement; ctx: CanvasRenderingContext2D;
  visual: VisualRisk | null; vframe: HTMLCanvasElement; vctx: CanvasRenderingContext2D; modelInfo: string;
}

export async function createRuntime(base: string, onStatus: (s: string) => void): Promise<Runtime> {
  const url = `${base}models/yolo11n_512x288_nms.onnx`;
  onStatus("Downloading the detector (10 MB, once)…");
  const buf = await (await fetch(url)).arrayBuffer();
  let session: ort.InferenceSession | null = null, backend = "wasm";
  if ("gpu" in navigator) {
    try { onStatus("Starting the detector on the GPU…"); session = await ort.InferenceSession.create(buf, { executionProviders: ["webgpu"] }); backend = "webgpu"; }
    catch { session = null; }
  }
  if (!session) { onStatus("Starting the detector on the CPU…"); session = await ort.InferenceSession.create(buf, { executionProviders: ["wasm"] }); }
  const guard = new Guard(new Detector(ort as never, session as never, 0.25));
  let modelInfo = "hand-set weights";
  try {
    const m = await (await fetch(`${base}models/risk_model.json`)).json() as TreeModel;
    if (m && Array.isArray(m.trees)) {
      guard.model = m; modelInfo = "learned risk model";
      // alert thresholds chosen on the training split ship inside the model file
      const al = (m as unknown as { alert?: Partial<typeof guard.alert> }).alert;
      if (al) Object.assign(guard.alert, al);
    }
  } catch { /* fall back to the calibrator or the hand-set score */ }
  if (!guard.model) {
    try {
      const cal = await (await fetch(`${base}models/calibrator.json`)).json() as Calibrator;
      if (cal && Array.isArray(cal.coef)) { guard.calibrator = cal; modelInfo = "logistic risk model"; }
    } catch { /* hand-set score */ }
  }
  // optional visual model (published after training on the Nexar training split)
  let visual: VisualRisk | null = null;
  try {
    const head = await fetch(`${base}models/visual_head.onnx`);
    if (head.ok) {
      onStatus("Downloading the visual model (13 MB, once)…");
      const [bbBuf, hdBuf] = await Promise.all([(await fetch(`${base}models/visual_backbone.onnx`)).arrayBuffer(), head.arrayBuffer()]);
      const eps = backend === "webgpu" ? ["webgpu"] : ["wasm"];
      const bb = await ort.InferenceSession.create(bbBuf, { executionProviders: eps });
      const hd = await ort.InferenceSession.create(hdBuf, { executionProviders: ["wasm"] });
      visual = new VisualRisk(ort as never, bb as never, hd as never);
      modelInfo += " + visual model";
    }
  } catch { visual = null; }
  const vframe = document.createElement("canvas");
  vframe.width = VIS_W; vframe.height = VIS_H;
  const vctx = vframe.getContext("2d", { willReadFrequently: true })!;
  const frame = document.createElement("canvas");
  frame.width = IN_W; frame.height = IN_H;
  const ctx = frame.getContext("2d", { willReadFrequently: true })!;
  // warm-up so the first real frame is not slow
  ctx.fillStyle = "#555"; ctx.fillRect(0, 0, IN_W, IN_H);
  await guard.detector.detect(ctx.getImageData(0, 0, IN_W, IN_H).data, 4);
  return { guard, backend, frame, ctx, visual, vframe, vctx, modelInfo };
}

/** The 16:9 band of the source that the model sees (centre crop: in portrait it keeps the road band). */
export function cropRect(sw: number, sh: number): { sx: number; sy: number; sw: number; sh: number } {
  const target = IN_W / IN_H;
  if (sw / sh > target) { const w = sh * target; return { sx: (sw - w) / 2, sy: 0, sw: w, sh }; }
  const h = sw / target;
  return { sx: 0, sy: Math.max(0, (sh - h) * 0.5), sw, sh: h };
}
