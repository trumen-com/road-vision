// Stage 1: run the detector over dashcam clips and cache detections (same Detector class the web app uses).
//   npx tsx eval/extract_dets.ts --list clips.txt --out cache/dets --fps 10 [--start-from-end 12]
// clips.txt: one path per line. Frames are decoded and resized to the model input by ffmpeg.
import { spawn } from "node:child_process";
import { existsSync, mkdirSync, readFileSync, writeFileSync } from "node:fs";
import { basename, dirname, join } from "node:path";
import * as ort from "onnxruntime-node";
import { Detector } from "../engine/detector";
import { IN_H, IN_W } from "../engine/types";

const args = Object.fromEntries(process.argv.slice(2).reduce<string[][]>((a, v, i, arr) => (v.startsWith("--") ? [...a, [v.slice(2), arr[i + 1]]] : a), []));
const fps = Number(args.fps ?? 10);
const model = args.model ?? join(dirname(new URL(import.meta.url).pathname), "../app/public/models/yolo11n_512x288_nms.onnx");

function probeDuration(path: string): Promise<number> {
  return new Promise((res) => {
    const p = spawn("ffprobe", ["-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", path]);
    let s = ""; p.stdout.on("data", (d) => (s += d)); p.on("close", () => res(Number(s) || 0));
  });
}

async function* frames(path: string, start: number): AsyncGenerator<Uint8Array> {
  const size = IN_W * IN_H * 3;
  const ff = spawn("ffmpeg", ["-v", "error", "-ss", String(start), "-i", path, "-vf", `fps=${fps},scale=${IN_W}:${IN_H}`, "-f", "rawvideo", "-pix_fmt", "rgb24", "-"]);
  let buf = Buffer.alloc(0);
  for await (const chunk of ff.stdout) {
    buf = Buffer.concat([buf, chunk as Buffer]);
    while (buf.length >= size) { yield new Uint8Array(buf.subarray(0, size)); buf = buf.subarray(size); }
  }
}

async function main() {
  const session = await ort.InferenceSession.create(model, { intraOpNumThreads: Number(args.threads ?? 2), executionMode: "sequential" });
  const det = new Detector(ort as never, session as never, 0.2);
  const out = args.out ?? "cache/dets";
  mkdirSync(out, { recursive: true });
  // list lines: "path" or "path,start,duration" (seconds)
  const list = readFileSync(args.list, "utf8").split("\n").map((s) => s.trim()).filter(Boolean);
  let done = 0, ms = 0, nf = 0;
  for (const line of list) {
    const [path, st, du] = line.split(",");
    const key = basename(path, ".mp4");
    const dst = join(out, `${key}.json`);
    if (existsSync(dst)) { done++; continue; }
    const dur = await probeDuration(path);
    if (!existsSync(path)) continue;
    const start = st ? Math.max(0, Number(st)) : args["start-from-end"] ? Math.max(0, dur - Number(args["start-from-end"])) : 0;
    const maxFrames = du ? Math.round(Number(du) * fps) : Infinity;
    const rec: { t: number; d: number[][] }[] = [];
    let i = 0;
    for await (const px of frames(path, start)) {
      if (i >= maxFrames) break;
      const t0 = performance.now();
      const ds = await det.detect(px, 3);
      ms += performance.now() - t0; nf++;
      rec.push({ t: +(start + i / fps).toFixed(3), d: ds.map((d) => [+d.x1.toFixed(1), +d.y1.toFixed(1), +d.x2.toFixed(1), +d.y2.toFixed(1), +d.score.toFixed(3), d.cls]) });
      i++;
    }
    writeFileSync(dst, JSON.stringify({ clip: key, fps, start, duration: dur, frames: rec }));
    done++;
    if (done % 10 === 0) console.log(`${done}/${list.length}  ${(ms / Math.max(nf, 1)).toFixed(1)} ms/frame`);
  }
  console.log(`done ${done} clips, detector ${(ms / Math.max(nf, 1)).toFixed(1)} ms/frame on ${args.threads ?? 2} CPU threads`);
}
main();
