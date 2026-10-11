// Trumen Guard engine: frame in -> calibrated collision risk + alert level out. Causal: uses only past frames.
import { Detector } from "./detector";
import { aggregate, predictTrees, TreeModel } from "./model";
import { assessTracks, DEFAULT_THREAT, ThreatParams } from "./threat";
import { Tracker } from "./tracker";
import { AlertLevel, Corridor, DEFAULT_CORRIDOR, FrameFeatures, GuardState, IN_H, IN_W, TrackThreat, VULNERABLE } from "./types";

/** Logistic calibrator over frame features, learned on the Nexar training split (eval/train_calibrator.py). */
export interface Calibrator { names: (keyof FrameFeatures)[]; mean: number[]; scale: number[]; coef: number[]; intercept: number }

export interface AlertParams { caution: number; brake: number; release: number; holdFrames: number; attackHalfLife: number; releaseHalfLife: number }
export const DEFAULT_ALERT: AlertParams = { caution: 0.5, brake: 0.75, release: 0.35, holdFrames: 2, attackHalfLife: 0.05, releaseHalfLife: 0.6 };

export function featuresOf(threats: TrackThreat[], egoShift: number): FrameFeatures {
  const cap = (x: number) => (Number.isFinite(x) ? Math.min(x, 10) : 10);
  let minTtcPath = 10, minTtcSoon = 10, maxLooming = 0, maxProxPath = 0, maxWidthPath = 0, nPath = 0, vulnPath = 0, heuristic = 0;
  let maxHeightPath = 0, ttcOfBiggest = 10, minAbsOffset = 5, vulnNear = 0, bigW = 0, tauDotPath = 0.5, bestTtc = Infinity, minTauDot = 0.5;
  for (const th of threats) {
    heuristic = Math.max(heuristic, th.risk);
    if (th.inPathNow) {
      nPath++;
      minTtcPath = Math.min(minTtcPath, cap(th.ttc));
      maxLooming = Math.max(maxLooming, Number.isFinite(th.ttc) ? 1 / th.ttc : 0);
      maxProxPath = Math.max(maxProxPath, th.proximity);
      maxWidthPath = Math.max(maxWidthPath, (th.box[2] - th.box[0]) / IN_W);
      maxHeightPath = Math.max(maxHeightPath, (th.box[3] - th.box[1]) / IN_H);
      if (th.box[2] - th.box[0] > bigW) { bigW = th.box[2] - th.box[0]; ttcOfBiggest = cap(th.ttc); }
    }
    if (Number.isFinite(th.ttc)) minAbsOffset = Math.min(minAbsOffset, Math.abs(th.offsetM));
    if ((th.inPathNow || th.inPathSoon) && th.ttc < bestTtc) { bestTtc = th.ttc; tauDotPath = th.tauDot; }
    if (Number.isFinite(th.ttc) && Math.abs(th.offsetM) < 1.5) minTauDot = Math.min(minTauDot, th.tauDot);
    if ((th.inPathNow || th.inPathSoon) && VULNERABLE.has(th.cls)) vulnNear = Math.max(vulnNear, th.proximity);
    if (th.inPathSoon) minTtcSoon = Math.min(minTtcSoon, cap(th.ttc));
    if ((th.inPathNow || th.inPathSoon) && VULNERABLE.has(th.cls)) vulnPath = 1;
  }
  return { minTtcPath, minTtcSoon, maxLooming, maxProxPath, maxWidthPath, nPath, vulnPath, egoShift: egoShift / IN_W, heuristic,
    maxHeightPath, ttcOfBiggest, minAbsOffset, nTracks: threats.length, vulnNear, tauDotPath, minTauDot };
}

export function applyCalibrator(cal: Calibrator | null, f: FrameFeatures): number {
  if (!cal) return f.heuristic;
  let z = cal.intercept;
  cal.names.forEach((n, i) => { z += cal.coef[i] * ((f[n] as number) - cal.mean[i]) / cal.scale[i]; });
  return 1 / (1 + Math.exp(-z));
}

export class Guard {
  tracker = new Tracker();
  corridor: Corridor = { ...DEFAULT_CORRIDOR };
  threat: ThreatParams = { ...DEFAULT_THREAT };
  alert: AlertParams = { ...DEFAULT_ALERT };
  calibrator: Calibrator | null = null;
  model: TreeModel | null = null;           // learned risk model over 2 s of features (preferred when loaded)
  visualRisk: number | null = null;         // latest output of the optional visual model (set by the app)
  private hist: { t: number; f: FrameFeatures }[] = [];
  private smooth = 0; private lastT = -1; private above = 0; private level: AlertLevel = 0; private belowSince = -1;

  constructor(public detector: Detector) {}

  reset() { this.tracker.reset(); this.hist = []; this.visualRisk = null; this.smooth = 0; this.lastT = -1; this.above = 0; this.level = 0; this.belowSince = -1; }

  async step(pixels: Uint8Array | Uint8ClampedArray, channels: 3 | 4, t: number): Promise<GuardState> {
    const t0 = (globalThis.performance ?? Date).now();
    const dets = await this.detector.detect(pixels, channels);
    const inferMs = (globalThis.performance ?? Date).now() - t0;
    return this.stepDetections(dets, t, inferMs);
  }

  private lastAgg(_f: FrameFeatures): number[] | undefined { return this.exportAgg ? aggregate(this.hist) : undefined; }
  exportAgg = false;

  stepDetections(dets: Parameters<Tracker["update"]>[0], t: number, inferMs = 0): GuardState {
    const tracks = this.tracker.update(dets, t);
    const { threats, egoShift } = assessTracks(tracks, t, this.corridor, this.threat);
    const features = featuresOf(threats, egoShift);
    this.hist.push({ t, f: features });
    while (this.hist.length && t - this.hist[0].t > 2.0) this.hist.shift();
    const geo = this.model ? predictTrees(this.model, aggregate(this.hist)) : applyCalibrator(this.calibrator, features);
    // late fusion with the visual model when present (equal weights, fixed in advance, not tuned on the test set)
    const raw = this.visualRisk == null ? geo : 0.5 * geo + 0.5 * this.visualRisk;
    // asymmetric smoothing: react within a frame, release slowly (one danger episode = one alarm)
    const dt = this.lastT < 0 ? 0 : Math.max(0, t - this.lastT);
    const hl = raw > this.smooth ? this.alert.attackHalfLife : this.alert.releaseHalfLife;
    const a = dt === 0 ? 1 : 1 - Math.pow(0.5, dt / hl);
    this.smooth += a * (raw - this.smooth);
    this.lastT = t;
    const r = this.smooth;
    this.above = r >= this.alert.caution ? this.above + 1 : 0;
    if (this.above >= this.alert.holdFrames) this.level = r >= this.alert.brake ? 2 : Math.max(this.level, 1) as AlertLevel;
    if (this.level > 0 && r < this.alert.release) {
      if (this.belowSince < 0) this.belowSince = t;
      if (t - this.belowSince > 0.5) this.level = 0;
    } else this.belowSince = -1;
    if (this.level === 2 && r < this.alert.caution) this.level = 1;
    let top: TrackThreat | undefined;
    for (const th of threats) if (!top || th.risk > top.risk) top = th;
    return { t, detections: dets.length, threats, features, risk: r, level: this.level, top, inferMs, geoRisk: geo, aggregate: this.lastAgg(features) };
  }
}
