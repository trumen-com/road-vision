"use client";

import { useRef, useState } from "react";
import VideoPanel from "@/components/VideoPanel";
import { API_URL, type VideoDoc } from "@/lib/data";

const MAX_MB = 100;
const MAX_SEC = 120;

type Job = { id: string; status: string; progress: number; stage?: string; error?: string; result?: VideoDoc; video_url?: string };

export default function Demo() {
  const [file, setFile] = useState<File | null>(null);
  const [job, setJob] = useState<Job | null>(null);
  const [err, setErr] = useState<string | null>(null);
  const timer = useRef<ReturnType<typeof setInterval> | null>(null);

  const check = async (f: File): Promise<string | null> => {
    if (!f.name.toLowerCase().endsWith(".mp4")) return "Please choose an .mp4 file.";
    if (f.size > MAX_MB * 1024 * 1024) return `File is larger than ${MAX_MB} MB.`;
    const dur = await new Promise<number>((res) => {
      const v = document.createElement("video");
      v.preload = "metadata";
      v.onloadedmetadata = () => res(v.duration);
      v.onerror = () => res(0);
      v.src = URL.createObjectURL(f);
    });
    if (dur > MAX_SEC + 1) return `Video is ${dur.toFixed(0)} s long; the demo accepts up to ${MAX_SEC} s.`;
    return null;
  };

  const start = async () => {
    if (!file) return;
    setErr(null);
    const problem = await check(file);
    if (problem) { setErr(problem); return; }
    if (!API_URL) { setErr("The demo backend URL is not configured (NEXT_PUBLIC_API_URL)."); return; }
    const body = new FormData();
    body.append("file", file);
    setJob({ id: "", status: "uploading", progress: 0, stage: "Uploading" });
    try {
      const r = await fetch(`${API_URL}/api/jobs`, { method: "POST", body });
      const j = await r.json();
      if (!r.ok) throw new Error(j.detail ?? r.statusText);
      setJob(j);
      timer.current = setInterval(async () => {
        const s = await fetch(`${API_URL}/api/jobs/${j.id}`).then((x) => x.json()).catch(() => null);
        if (!s) return;
        setJob(s);
        if (s.status === "done" || s.status === "error") { if (timer.current) clearInterval(timer.current); }
      }, 1500);
    } catch (e) {
      setErr(`Upload failed: ${e instanceof Error ? e.message : String(e)}`);
      setJob(null);
    }
  };

  const busy = job && job.status !== "done" && job.status !== "error";
  return (
    <>
      <h1>Live demo</h1>
      <p className="lead">Upload a road video and get the detected events back: a timeline, an annotated playback and the accident-risk curve.</p>
      <div className="card">
        <div style={{ display: "flex", gap: 12, alignItems: "center", flexWrap: "wrap" }}>
          <input type="file" accept="video/mp4" onChange={(e) => { setFile(e.target.files?.[0] ?? null); setJob(null); setErr(null); }} />
          <button className="btn" disabled={!file || !!busy} onClick={start}>Analyse video</button>
        </div>
        <p className="muted" style={{ marginTop: 10 }}>
          Accepts .mp4 up to {MAX_SEC} s and {MAX_MB} MB. Runs on CPU with a smaller detector (YOLO11n, every few frames), so expect roughly
          1–2 minutes for a 2-minute clip. Results are best on footage from the competition camera, since the scene map is drawn for it;
          on other footage the learned flow field still gives directions, congestion, stopped vehicles and collisions.
        </p>
        {err && <div className="callout" style={{ borderColor: "var(--fam-collision)" }}>{err}</div>}
        {job && (
          <div style={{ marginTop: 12 }}>
            <div className="muted">{job.status === "error" ? `Failed: ${job.error}` : `${job.stage ?? job.status} · ${Math.round(job.progress * 100)}%`}</div>
            <div className="progress" style={{ marginTop: 6 }}><div style={{ width: `${Math.round(job.progress * 100)}%` }} /></div>
          </div>
        )}
      </div>
      {job?.status === "done" && job.result && (
        <div style={{ marginTop: 16 }}>
          <VideoPanel doc={job.result} videoUrl={job.video_url ? `${API_URL}${job.video_url}` : undefined} />
          <div className="card" style={{ marginTop: 16 }}>
            <h3>Raw output</h3>
            <pre>{JSON.stringify(job.result.segments, null, 0)}</pre>
          </div>
        </div>
      )}
    </>
  );
}
