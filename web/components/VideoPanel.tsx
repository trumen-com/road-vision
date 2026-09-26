"use client";

import { useRef, useState } from "react";
import { RiskChart, Timeline } from "@/components/charts";
import { classColor, className } from "@/lib/classes";
import { fmtTime, type VideoDoc } from "@/lib/data";

/** Annotated playback with a clickable event timeline, risk curve and event table. */
export default function VideoPanel({ doc, videoUrl }: { doc: VideoDoc; videoUrl?: string }) {
  const ref = useRef<HTMLVideoElement>(null);
  const [time, setTime] = useState(0);
  const seek = (t: number) => {
    if (ref.current) { ref.current.currentTime = t; ref.current.play().catch(() => undefined); }
    setTime(t);
  };
  const accidents = doc.segments.filter((s) => s[2] === "accident");
  const perClass = doc.segments.reduce<Record<string, number>>((m, s) => ({ ...m, [s[2]]: (m[s[2]] ?? 0) + 1 }), {});
  return (
    <div className="grid" style={{ gap: 14 }}>
      {videoUrl && (
        <video ref={ref} src={videoUrl} controls playsInline preload="metadata"
               onTimeUpdate={(e) => setTime(e.currentTarget.currentTime)} />
      )}
      <div className="card">
        <h3>Event timeline</h3>
        <p className="muted">Click an event, or anywhere on the track, to jump the video there.</p>
        <Timeline segments={doc.segments} duration={doc.duration} time={time} onSeek={seek} />
      </div>
      {doc.risk?.length > 0 && (
        <div className="card">
          <h3>Accident risk (Part B, causal)</h3>
          <p className="muted">P(accident starts within 5 s), computed frame by frame from past frames only. Shaded bands are the 5 s before each detected accident.</p>
          <RiskChart risk={doc.risk} duration={doc.duration} accidents={accidents} time={time} onSeek={seek} />
        </div>
      )}
      <div className="grid cols-2">
        <div className="card">
          <h3>Events ({doc.segments.length})</h3>
          <div className="table-scroll" style={{ maxHeight: 320, overflowY: "auto" }}>
            <table className="tbl">
              <thead><tr><th>Class</th><th>Start</th><th>End</th><th>Length</th></tr></thead>
              <tbody>
                {doc.segments.map(([s, e, l], i) => (
                  <tr key={i} className="clickable" onClick={() => seek(s)}>
                    <td><span className="pill"><i className="dot" style={{ background: classColor(l) }} />{className(l)}</span></td>
                    <td className="mono">{fmtTime(s)}</td><td className="mono">{fmtTime(e)}</td><td className="mono">{(e - s).toFixed(1)} s</td>
                  </tr>
                ))}
                {doc.segments.length === 0 && <tr><td colSpan={4}>No events detected.</td></tr>}
              </tbody>
            </table>
          </div>
        </div>
        <div className="card">
          <h3>Summary</h3>
          <table className="tbl">
            <tbody>
              <tr><td>Video</td><td className="mono">{doc.video}</td></tr>
              <tr><td>Resolution · fps</td><td>{doc.width}×{doc.height} · {doc.fps.toFixed(2)}</td></tr>
              <tr><td>Duration</td><td>{doc.duration.toFixed(1)} s</td></tr>
              <tr><td>Frames analysed</td><td>{doc.stride === 1 ? "every frame" : `every ${doc.stride}${doc.stride === 2 ? "nd" : doc.stride === 3 ? "rd" : "th"} frame`} ({(doc.fps / doc.stride).toFixed(1)} per second)</td></tr>
              {doc.timings?.total_sec !== undefined && !doc.timings?.cached && <tr><td>Processing time</td><td>{String(doc.timings.total_sec)} s on {String(doc.timings.device ?? "")}</td></tr>}
              {Object.entries(perClass).map(([k, v]) => (
                <tr key={k}><td><i className="dot" style={{ background: classColor(k) }} /> {className(k)}</td><td>{v}</td></tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  );
}
