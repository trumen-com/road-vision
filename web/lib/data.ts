// Types for the JSON written by tools/export_site.py and returned by the demo API.

export type Segment = [number, number, string];

export interface RawEvent {
  label: string;
  start: number;
  end: number;
  tracks: number[];
  score: number;
  note: string;
}

export interface VideoDoc {
  video: string;
  fps: number;
  n_frames: number;
  width: number;
  height: number;
  duration: number;
  stride: number;
  segments: Segment[];
  raw: RawEvent[];
  risk: [number, number][];
  frame_stats?: { t: number; brightness: number; contrast: number }[];
  counts?: Record<string, number | null>[];
  speed_hist?: number[];
  n_tracks?: Record<string, number>;
  timings?: Record<string, number | string | boolean>;
  media?: Record<string, string>;
  signals?: { t: number[]; states: Record<string, string[]> };
}

export type LightRun = [number, number, string];

/** Traffic-light state as runs [start, end, "red" | "green"] for the first signal head read in the video. */
export function lightRuns(doc: VideoDoc): LightRun[] {
  const sig = doc.signals;
  const id = sig ? Object.keys(sig.states)[0] : undefined;
  if (!sig || !id) return [];
  const st = sig.states[id], t = sig.t, out: LightRun[] = [];
  for (let i = 0; i < t.length; i++) {
    const end = i + 1 < t.length ? t[i + 1] : doc.duration;
    const last = out[out.length - 1];
    if (last && last[2] === st[i]) last[1] = end;
    else out.push([t[i], end, st[i]]);
  }
  // absorb flicker (runs under 2 s) into the previous phase, then re-merge equal neighbours
  const merged: LightRun[] = [];
  for (const r of out.filter((q) => q[2] === "red" || q[2] === "green")) {
    const last = merged[merged.length - 1];
    if (last && (last[2] === r[2] || r[1] - r[0] < 2)) last[1] = r[1];
    else merged.push([...r] as LightRun);
  }
  return merged;
}

export interface IndexEntry {
  id: string;
  file: string;
  duration: number;
  fps: number;
  width: number;
  height: number;
  events: number;
  classes: string[];
}

export const BASE = process.env.NEXT_PUBLIC_BASE_PATH ?? "";
export const API_URL = process.env.NEXT_PUBLIC_API_URL ?? "";

export async function fetchJSON<T>(path: string): Promise<T> {
  const r = await fetch(`${BASE}/${path}`);
  if (!r.ok) throw new Error(`${path}: ${r.status}`);
  return r.json() as Promise<T>;
}

export function fmtTime(t: number): string {
  const m = Math.floor(t / 60);
  const s = t - m * 60;
  return `${m}:${s.toFixed(1).padStart(4, "0")}`;
}
