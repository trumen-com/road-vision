// Lightweight online tracker: constant-velocity prediction + greedy IoU association (high-score pass, then low-score
// pass, after ByteTrack). Deterministic, allocation-light, fast enough for 10-30 fps in a phone browser.
import { Detection, IN_H, IN_W, PERSON, Sample } from "./types";

export interface Track {
  id: number;
  cls: number;
  samples: Sample[];      // most recent last, ~2 s of history
  lastT: number;
  hits: number;
  vx: number; vy: number; vw: number; vh: number;   // px/s, smoothed
  ttcHist?: { t: number; ttc: number }[];           // recent time-to-contact estimates (for tau-dot)
  lastOffM?: number;                                // last lateral offset (m) measured with an uncut box
  offHist?: { t: number; off: number }[];           // recent lateral offsets (m), for the sideways speed
}

const family = (c: number) => (c === PERSON || c === 1 || c === 3 ? 1 : c >= 16 ? 2 : 0); // people+2-wheelers / animals / vehicles

function iou(a: number[], b: number[]): number {
  const ix = Math.max(0, Math.min(a[2], b[2]) - Math.max(a[0], b[0]));
  const iy = Math.max(0, Math.min(a[3], b[3]) - Math.max(a[1], b[1]));
  const inter = ix * iy;
  const u = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter;
  return u > 0 ? inter / u : 0;
}

export function cutFlags(x1: number, y1: number, x2: number, y2: number): number {
  const m = 2;
  return (x1 <= m || x2 >= IN_W - m ? 1 : 0) | (y2 >= IN_H - m ? 2 : 0) | (y1 <= m ? 4 : 0);
}

export class Tracker {
  tracks: Track[] = [];
  private nextId = 1;
  constructor(public highScore = 0.45, public newScore = 0.5, public maxAge = 0.7, public history = 2.0) {}

  reset() { this.tracks = []; this.nextId = 1; }

  private predict(tr: Track, t: number): number[] {
    const s = tr.samples[tr.samples.length - 1];
    const dt = Math.min(t - tr.lastT, 0.5);
    const cx = (s.x1 + s.x2) / 2 + tr.vx * dt, cy = (s.y1 + s.y2) / 2 + tr.vy * dt;
    const w = Math.max(2, s.x2 - s.x1 + tr.vw * dt), h = Math.max(2, s.y2 - s.y1 + tr.vh * dt);
    return [cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2];
  }

  update(dets: Detection[], t: number): Track[] {
    const preds = this.tracks.map((tr) => this.predict(tr, t));
    const used = new Set<number>(), matched = new Set<number>();
    for (const pass of [0, 1]) {
      const pairs: [number, number, number][] = [];
      dets.forEach((d, j) => {
        if (used.has(j) || (pass === 0 ? d.score < this.highScore : d.score >= this.highScore)) return;
        const db = [d.x1, d.y1, d.x2, d.y2];
        this.tracks.forEach((tr, i) => {
          if (matched.has(i) || family(tr.cls) !== family(d.cls)) return;
          const v = iou(preds[i], db);
          if (v >= (pass === 0 ? 0.2 : 0.35)) pairs.push([v, i, j]);
        });
      });
      pairs.sort((a, b) => b[0] - a[0]);
      for (const [, i, j] of pairs) {
        if (matched.has(i) || used.has(j)) continue;
        matched.add(i); used.add(j);
        this.extend(this.tracks[i], dets[j], t);
      }
    }
    dets.forEach((d, j) => {
      if (used.has(j) || d.score < this.newScore) return;
      this.tracks.push({ id: this.nextId++, cls: d.cls, samples: [], lastT: t, hits: 0, vx: 0, vy: 0, vw: 0, vh: 0 });
      this.extend(this.tracks[this.tracks.length - 1], d, t);
    });
    this.tracks = this.tracks.filter((tr) => t - tr.lastT <= this.maxAge);
    return this.tracks;
  }

  private extend(tr: Track, d: Detection, t: number) {
    const prev = tr.samples[tr.samples.length - 1];
    if (prev) {
      const dt = Math.max(t - prev.t, 1e-3), a = Math.min(1, dt / 0.25);     // EMA over ~0.25 s
      tr.vx += a * (((d.x1 + d.x2) - (prev.x1 + prev.x2)) / 2 / dt - tr.vx);
      tr.vy += a * (((d.y1 + d.y2) - (prev.y1 + prev.y2)) / 2 / dt - tr.vy);
      tr.vw += a * (((d.x2 - d.x1) - (prev.x2 - prev.x1)) / dt - tr.vw);
      tr.vh += a * (((d.y2 - d.y1) - (prev.y2 - prev.y1)) / dt - tr.vh);
    }
    // a person on a bike/motorbike is reported by its vehicle class; keep the most "vulnerable-vehicle" class seen
    if (tr.hits === 0 || (d.score > 0.5 && d.cls !== tr.cls && tr.cls === PERSON)) tr.cls = d.cls;
    tr.samples.push({ t, x1: d.x1, y1: d.y1, x2: d.x2, y2: d.y2, score: d.score, cut: cutFlags(d.x1, d.y1, d.x2, d.y2) });
    while (tr.samples.length && t - tr.samples[0].t > this.history) tr.samples.shift();
    tr.lastT = t;
    tr.hits++;
  }
}
