// Stage 2: replay the Guard engine over cached detections (fast, no detector) and write
//   - a Nexar-format submission (id,score): the risk the engine reports at the end of each clip (causal),
//   - optionally per-frame features for calibrator training, and per-clip alert timelines.
//   npx tsx eval/score.ts --dets cache/dets --out results/sub.csv [--features feats.jsonl] [--calibrator cal.json] [--timeline tl.json]
import { readdirSync, readFileSync, writeFileSync } from "node:fs";
import { join } from "node:path";
import { Calibrator, Guard } from "../engine/guard";
import { corridorAt } from "../engine/threat";
import { DEFAULT_CORRIDOR, Detection, IN_W } from "../engine/types";

const args = Object.fromEntries(process.argv.slice(2).reduce<string[][]>((a, v, i, arr) => (v.startsWith("--") ? [...a, [v.slice(2), arr[i + 1]]] : a), []));
const cal: Calibrator | null = args.calibrator ? JSON.parse(readFileSync(args.calibrator, "utf8")) : null;
const overrides = args.params ? JSON.parse(args.params) : {};

const guard = new Guard(null as never);
guard.calibrator = cal;
if (args.model) guard.model = JSON.parse(readFileSync(args.model, "utf8"));
if (args["no-model"]) guard.model = null;
guard.exportAgg = !!args.agg;
// out-of-fold scoring: each clip is scored by the model that did not see it during training
const foldDir = args["fold-models"];
const foldOf: Record<string, number> = foldDir ? JSON.parse(readFileSync(join(foldDir, "folds.json"), "utf8")) : {};
const foldModels = foldDir ? [0, 1, 2, 3, 4].map((k) => JSON.parse(readFileSync(join(foldDir, `fold${k}.json`), "utf8"))) : [];
Object.assign(guard.threat, overrides.threat ?? {});
Object.assign(guard.alert, overrides.alert ?? {});
Object.assign(guard.corridor, overrides.corridor ?? {});

// optional visual-risk timelines (eval/visual_timelines.py): fused exactly like the browser (latest value at time t)
const visTl: Record<string, [number, number][]> = args.visual ? JSON.parse(readFileSync(args.visual, "utf8")) : {};
const rows: string[] = ["id,score,max1s,final_level"];
const feats: string[] = [];
const timelines: Record<string, { t: number; r: number; l: number }[]> = {};
for (const f of readdirSync(args.dets).filter((x) => x.endsWith(".json")).sort()) {
  const rec = JSON.parse(readFileSync(join(args.dets, f), "utf8"));
  if (foldDir) { const k = foldOf[rec.clip]; if (k === undefined) continue; guard.model = foldModels[k]; }
  guard.reset();
  const tl: { t: number; r: number; l: number }[] = [];
  let last = 0, lastLevel = 0;
  for (const fr of rec.frames) {
    const dets: Detection[] = fr.d.map((d: number[]) => ({ x1: d[0], y1: d[1], x2: d[2], y2: d[3], score: d[4], cls: d[5] }));
    const vt = visTl[rec.clip];
    if (vt) {
      let v: number | null = null;
      for (const [tv, rv] of vt) { if (tv <= fr.t + 1e-6) v = rv; else break; }
      guard.visualRisk = v;
    }
    const st = guard.stepDetections(dets, fr.t);
    if (args.baseline) {
      // naive streaming baseline: width x confidence of the biggest road user in the ego corridor, this frame only
      let b = 0;
      for (const d of fr.d) {
        const cor = corridorAt(DEFAULT_CORRIDOR, d[3]), cx = (d[0] + d[2]) / 2, w = d[2] - d[0];
        if (Math.abs(cx - cor.cx) < cor.hw + 0.35 * w) b = Math.max(b, (w / IN_W) * d[4]);
      }
      tl.push({ t: fr.t, r: +b.toFixed(4), l: 0 });
      continue;
    }
    tl.push({ t: fr.t, r: +st.risk.toFixed(4), l: st.level });
    last = st.risk; lastLevel = st.level;
    if (args.features) feats.push(JSON.stringify(args.agg ? { clip: rec.clip, t: fr.t, x: st.aggregate!.map((v) => +v.toFixed(4)) } : { clip: rec.clip, t: fr.t, ...st.features }));
  }
  if (args.baseline && rec.frames.length) {
    // naive baseline: width of the biggest road user in the ego corridor on the last frame (no tracking, no TTC)
    const lastF = rec.frames[rec.frames.length - 1];
    let b = 0;
    for (const d of lastF.d) {
      const cor = corridorAt(DEFAULT_CORRIDOR, d[3]), cx = (d[0] + d[2]) / 2, w = d[2] - d[0];
      if (Math.abs(cx - cor.cx) < cor.hw + 0.35 * w) b = Math.max(b, (w / IN_W) * d[4]);
    }
    last = b;
  }
  const tEnd = tl.length ? tl[tl.length - 1].t : 0;
  const max1 = Math.max(0, ...tl.filter((x) => x.t >= tEnd - 1.0).map((x) => x.r));
  rows.push(`${rec.clip},${last.toFixed(5)},${max1.toFixed(5)},${lastLevel}`);
  if (args.timeline) timelines[rec.clip] = tl;
}
writeFileSync(args.out, rows.join("\n") + "\n");
if (args.features) writeFileSync(args.features, feats.join("\n") + "\n");
if (args.timeline) writeFileSync(args.timeline, JSON.stringify(timelines));
console.log(`scored ${rows.length - 1} clips -> ${args.out}`);
