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
