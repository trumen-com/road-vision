// Learned risk model over a short history of frame features.
//   aggregate(): per-feature [last, max over 1 s, min over 1 s, mean over 2 s, change over the last 0.5 s]
//   Gradient-boosted trees (sklearn HistGradientBoosting, exported to JSON by eval/train_model.py) -> probability.
import { FrameFeatures } from "./types";

export const MODEL_FEATURES: (keyof FrameFeatures)[] = [
  "maxLooming", "minTtcPath", "minTtcSoon", "ttcOfBiggest", "maxProxPath", "maxWidthPath", "maxHeightPath",
  "nPath", "vulnPath", "vulnNear", "minAbsOffset", "egoShift", "nTracks", "heuristic", "tauDotPath", "minTauDot",
];
export const AGGS = ["last", "max1", "min1", "mean2", "delta"] as const;

export function aggregate(hist: { t: number; f: FrameFeatures }[]): number[] {
  const n = hist.length, tEnd = hist[n - 1].t;
  const out: number[] = [];
  for (const k of MODEL_FEATURES) {
    let mx = -Infinity, mn = Infinity, s2 = 0, c2 = 0, sRecent = 0, cRecent = 0, sPrev = 0, cPrev = 0;
    for (const h of hist) {
      const v = h.f[k] as number, age = tEnd - h.t;
      if (age <= 1.0) { mx = Math.max(mx, v); mn = Math.min(mn, v); }
      if (age <= 2.0) { s2 += v; c2++; }
      if (age <= 0.5) { sRecent += v; cRecent++; } else if (age <= 1.5) { sPrev += v; cPrev++; }
    }
    const last = hist[n - 1].f[k] as number;
    const recent = cRecent ? sRecent / cRecent : last;
    out.push(last, mx, mn, c2 ? s2 / c2 : last, recent - (cPrev ? sPrev / cPrev : recent));
  }
  return out;
}

/** One tree: flat arrays; leaf when left < 0. */
export interface Tree { f: number[]; thr: number[]; left: number[]; right: number[]; value: number[] }
export interface TreeModel {
  kind: "hgb"; baseline: number; trees: Tree[]; features: string[]; trained_on?: string;
  /** monotonic piecewise-linear display scale [[raw, shown], ...]: the shipping caution point reads 0.5, brake 0.75 */
  remap?: [number, number][];
}

export function remap(m: TreeModel, p: number): number {
  const k = m.remap;
  if (!k || k.length < 2) return p;
  for (let i = 1; i < k.length; i++) {
    if (p <= k[i][0]) { const [x0, y0] = k[i - 1], [x1, y1] = k[i]; return y0 + ((p - x0) / Math.max(x1 - x0, 1e-9)) * (y1 - y0); }
  }
  return k[k.length - 1][1];
}

export function predictTrees(m: TreeModel, x: number[]): number {
  let z = m.baseline;
  for (const t of m.trees) {
    let i = 0;
    while (t.left[i] >= 0) i = x[t.f[i]] <= t.thr[i] ? t.left[i] : t.right[i];
    z += t.value[i];
  }
  return remap(m, 1 / (1 + Math.exp(-z)));
}
