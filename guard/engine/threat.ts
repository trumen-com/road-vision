// Per-object threat from monocular geometry.
//
// Time to collision from looming: for an object approaching at constant relative speed, its image size s grows so
// that d(ln s)/dt = 1/TTC. We fit ln s against time over the last ~0.8 s by least squares (robust to box jitter) and
// use the width when the box is cut at the bottom of the frame, the height when it is cut at a side.
//
// Path test: the ego corridor is a trapezoid from the bottom of the frame to the vanishing point. An object threatens
// us if it is in the corridor now, or if its lateral motion (with the scene-wide sideways shift removed, i.e. our own
// turning) brings it into the corridor by the time contact would happen: cut-ins and pedestrians stepping out.
import { Track } from "./tracker";
import { Corridor, IN_H, IN_W, TrackThreat, VULNERABLE } from "./types";

export interface ThreatParams {
  window: number;        // s of history used for the TTC fit (extended up to maxWindow at low frame rates)
  maxWindow: number;
  minSpan: number;       // s, minimum time span of the fit
  minSamples: number;
  ttcMid: number;        // s, TTC at which the heuristic threat is 0.5
  ttcWidth: number;      // s, softness of the TTC ramp
  horizon: number;       // s, look-ahead for the lateral path prediction
  minHeightPx: number;   // ignore tiny far objects (jitter dominates)
  halfLaneM: number;     // half of our own vehicle width + margin, metres
}
export const DEFAULT_THREAT: ThreatParams = {
  window: 0.8, minSpan: 0.3, minSamples: 4, maxWindow: 1.6, ttcMid: 2.2, ttcWidth: 0.45, horizon: 2.5, minHeightPx: 14, halfLaneM: 1.1,
};

export interface Fit { looming: number; vx: number; ok: boolean; span: number; resid: number; lag: number }

/** Typical real-world widths (m) used to turn pixel offsets into metres. */
export const WIDTH_M: Record<number, number> = { 0: 0.5, 1: 0.6, 2: 1.8, 3: 0.8, 5: 2.5, 7: 2.4, 16: 0.4, 17: 0.6, 18: 0.5, 19: 0.8 };

/** Least-squares slope of ln(size) and of the centre x over the recent window. */
export function fitTrack(tr: Track, t: number, p: ThreatParams): Fit {
  const xs: number[] = [], ls: number[] = [], cs: number[] = [];
  for (let i = tr.samples.length - 1; i >= 0; i--) {
    const s = tr.samples[i];
    if (t - s.t > p.maxWindow || (t - s.t > p.window && xs.length >= p.minSamples)) break;
    if (s.cut & 4) continue;                              // top cut: unreliable
    const w = s.x2 - s.x1, h = s.y2 - s.y1;
    // height does not change when a car turns and shows its side (width does); width only when the bottom is cut
    const size = (s.cut & 2) ? ((s.cut & 1) ? 0 : w) : h;
    if (size <= 2) continue;
    xs.push(s.t); ls.push(Math.log(size)); cs.push((s.x1 + s.x2) / 2);
  }
  const n = xs.length;
  if (n < p.minSamples || xs[0] - xs[n - 1] < p.minSpan) return { looming: 0, vx: 0, ok: false, span: n ? xs[0] - xs[n - 1] : 0, resid: 0, lag: 0 };
  let mt = 0, ml = 0, mc = 0;
  for (let i = 0; i < n; i++) { mt += xs[i]; ml += ls[i]; mc += cs[i]; }
  mt /= n; ml /= n; mc /= n;
  let stt = 0, stl = 0, stc = 0;
  for (let i = 0; i < n; i++) { const d = xs[i] - mt; stt += d * d; stl += d * (ls[i] - ml); stc += d * (cs[i] - mc); }
  // Theil-Sen (median of pairwise slopes) for ln size: robust to a single jumpy box (occlusion, detector flicker)
  const slopes: number[] = [];
  for (let i = 0; i < n; i++) for (let j = i + 1; j < n; j++) { const dt = xs[i] - xs[j]; if (Math.abs(dt) > 0.05) slopes.push((ls[i] - ls[j]) / dt); }
  slopes.sort((p1, p2) => p1 - p2);
  const b = slopes.length ? slopes[slopes.length >> 1] : stl / stt;
  let rss = 0;
  for (let i = 0; i < n; i++) { const e = ls[i] - (ml + b * (xs[i] - mt)); rss += e * e; }
  // the fit describes the middle of the window; under constant closing speed TTC(now) = TTC(mid) - (now - mid)
  return { looming: b, vx: stc / stt, ok: true, span: xs[0] - xs[n - 1], resid: Math.sqrt(rss / n), lag: t - mt };
}

export function corridorAt(c: Corridor, y: number): { cx: number; hw: number } {
  const vpY = c.vpY * IN_H, f = Math.max(0, (y - vpY) / (IN_H - vpY));
  return { cx: (c.vpX * IN_W) + f * (IN_W / 2 - c.vpX * IN_W), hw: f * c.bottomHalfWidth * IN_W };
}

const sigmoid = (x: number) => 1 / (1 + Math.exp(-x));

export function assessTracks(tracks: Track[], t: number, c: Corridor, p: ThreatParams): { threats: TrackThreat[]; egoShift: number; fits: Map<number, Fit> } {
  const fits = new Map<number, Fit>();
  const lateral: number[] = [];
  for (const tr of tracks) {
    if (tr.lastT < t - 1e-6 || tr.hits < 3) continue;
    const f = fitTrack(tr, t, p);
    fits.set(tr.id, f);
    if (f.ok) lateral.push(f.vx);
  }
  lateral.sort((a, b) => a - b);
  // our own turning shifts everything sideways; remove the scene-wide median (needs a few objects to be meaningful)
  const egoShift = lateral.length >= 3 ? lateral[lateral.length >> 1] : 0;
  const threats: TrackThreat[] = [];
  const vpY = c.vpY * IN_H;
  for (const tr of tracks) {
    const f = fits.get(tr.id);
    if (!f) continue;
    const s = tr.samples[tr.samples.length - 1];
    const h = s.y2 - s.y1, w = s.x2 - s.x1;
    if (h < p.minHeightPx || s.y2 < vpY) continue;
    const ttc = f.ok && f.looming > 0.05 ? Math.max(0.05, 1 / f.looming - f.lag) : Infinity;
    // tau-dot (Lee, 1976): slope of TTC over time. Controlled braking keeps it above about -0.5; on a collision course
    // TTC falls by about one second per second (tau-dot near -1).
    tr.ttcHist ??= [];
    if (Number.isFinite(ttc) && ttc < 10) tr.ttcHist.push({ t, ttc });
    while (tr.ttcHist.length && t - tr.ttcHist[0].t > 0.8) tr.ttcHist.shift();
    let tauDot = 0.5;
    if (tr.ttcHist.length >= 3 && tr.ttcHist[tr.ttcHist.length - 1].t - tr.ttcHist[0].t >= 0.25) {
      const hs = tr.ttcHist, n = hs.length;
      let mt = 0, mv = 0; for (const q of hs) { mt += q.t; mv += q.ttc; } mt /= n; mv /= n;
      let a = 0, b = 0; for (const q of hs) { a += (q.t - mt) * (q.ttc - mv); b += (q.t - mt) ** 2; }
      tauDot = Math.max(-3, Math.min(3, a / b));
    }
    // Lateral offset from our path in real-world metres, using the object's own width as the scale: the ratio
    // (offset px / width px) does not change with distance, so a parked car stays "1.5 widths to the right" all the
    // way in, while a car in our lane stays near zero. Our path centre is the line from the bottom centre to the VP.
    const widthM = WIDTH_M[tr.cls] ?? 1.8;
    const cx = (s.x1 + s.x2) / 2, centre = corridorAt(c, s.y2).cx;
    // a box cut by the image side has an unreliable width: reuse the last offset measured while fully visible
    // (objects at the image edge are almost always passing beside us)
    let offM: number;
    if (s.cut & 1) offM = tr.lastOffM ?? Math.sign(cx - centre) * 9;
    else { offM = ((cx - centre) / Math.max(w, 1)) * widthM; tr.lastOffM = offM; }
    const reach = p.halfLaneM + widthM / 2;
    const inPathNow = Math.abs(offM) < reach;
    const look = Math.min(Number.isFinite(ttc) ? ttc : p.horizon, p.horizon);
    // sideways speed = slope of the metric offset over the last 0.8 s (pixel motion / width misses the approach term);
    // our own turning is removed with the scene-wide sideways shift converted to metres at this object's scale
    tr.offHist ??= [];
    if (!(s.cut & 1)) tr.offHist.push({ t, off: offM });
    while (tr.offHist.length && t - tr.offHist[0].t > 0.8) tr.offHist.shift();
    let latMs = 0;
    if (tr.offHist.length >= 3 && tr.offHist[tr.offHist.length - 1].t - tr.offHist[0].t >= 0.25) {
      const hs = tr.offHist, n = hs.length;
      let mt = 0, mo = 0; for (const q of hs) { mt += q.t; mo += q.off; } mt /= n; mo /= n;
      let a = 0, b = 0; for (const q of hs) { a += (q.t - mt) * (q.off - mo); b += (q.t - mt) ** 2; }
      latMs = a / b - (egoShift / Math.max(w, 1)) * widthM;
    }
    const offSoon = offM + latMs * look;
    const inPathSoon = !inPathNow && Math.abs(offSoon) < reach && Math.abs(latMs) > 0.3 && Math.sign(latMs) !== Math.sign(offM);
    const proximity = Math.min(1, Math.max(0, (s.y2 - vpY) / (IN_H - vpY)));
    const ttcTerm = Number.isFinite(ttc) ? sigmoid((p.ttcMid - ttc) / p.ttcWidth) : 0;
    const pathW = inPathNow ? 1 : inPathSoon ? 0.75 : 0.1;
    const classW = VULNERABLE.has(tr.cls) ? 1.0 : 0.9;
    // TTC not shrinking: the driver is already braking in control; shrinking faster than physically plausible: noise
    const control = tauDot > -0.4 ? 0.6 : tauDot < -2.2 ? 0.5 : 1.0;
    const risk = ttcTerm * pathW * classW * Math.min(1, 0.6 + s.score) * (f.span >= p.minSpan ? 1 : 0.5) * control;
    threats.push({ id: tr.id, cls: tr.cls, box: [s.x1, s.y1, s.x2, s.y2], ttc, inPathNow, inPathSoon, proximity, risk, offsetM: offM, lateralMs: latMs, tauDot });
  }
  return { threats, egoShift: Math.abs(egoShift), fits };
}
