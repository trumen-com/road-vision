// Dump per-frame engine states for one clip (for the recorded backup demo and for choosing sample clips).
//   npx tsx eval/demo_states.ts --dets ../guard_data/dets_test/00020.json --model app/public/models/risk_model.json --out /tmp/s.json
import { readFileSync, writeFileSync } from "node:fs";
import { Guard } from "../engine/guard";

const args = Object.fromEntries(process.argv.slice(2).reduce<string[][]>((a, v, i, arr) => (v.startsWith("--") ? [...a, [v.slice(2), arr[i + 1]]] : a), []));
const g = new Guard(null as never);
if (args.model) g.model = JSON.parse(readFileSync(args.model, "utf8"));
Object.assign(g.alert, args.alert ? JSON.parse(args.alert) : {});
const rec = JSON.parse(readFileSync(args.dets, "utf8"));
const out = rec.frames.map((fr: { t: number; d: number[][] }) => {
  const st = g.stepDetections(fr.d.map((d) => ({ x1: d[0], y1: d[1], x2: d[2], y2: d[3], score: d[4], cls: d[5] })), fr.t);
  return { t: fr.t, risk: st.risk, level: st.level, threats: st.threats.map((th) => ({ box: th.box, ttc: th.ttc, risk: th.risk, cls: th.cls, path: th.inPathNow || th.inPathSoon })), top: st.top ? { cls: st.top.cls, ttc: st.top.ttc, now: st.top.inPathNow, soon: st.top.inPathSoon } : null };
});
writeFileSync(args.out, JSON.stringify({ clip: rec.clip, frames: out }));
