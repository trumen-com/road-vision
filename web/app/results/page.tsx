"use client";

import { useEffect, useMemo, useState } from "react";
import VideoPanel from "@/components/VideoPanel";
import { Bars } from "@/components/charts";
import { CLASSES, classColor, className } from "@/lib/classes";
import { BASE, fetchJSON, type IndexEntry, type VideoDoc } from "@/lib/data";
import { FAILURES, SPOTCHECK } from "@/content/report";

export default function Results() {
  const [index, setIndex] = useState<IndexEntry[]>([]);
  const [docs, setDocs] = useState<Record<string, VideoDoc>>({});
  const [sel, setSel] = useState<string | null>(null);
  const [err, setErr] = useState<string | null>(null);

  useEffect(() => {
    fetchJSON<{ videos: IndexEntry[] }>("data/index.json")
      .then(async (d) => {
        setIndex(d.videos);
        setSel(d.videos[0]?.id ?? null);
        const all = await Promise.all(d.videos.map((v) => fetchJSON<VideoDoc>(`data/videos/${v.id}.json`)));
        setDocs(Object.fromEntries(all.map((doc, i) => [d.videos[i].id, doc])));
      })
      .catch((e) => setErr(String(e)));
  }, []);

  const doc = sel ? docs[sel] : undefined;
  // operator dashboard: events per class across all sample videos
  const perClass = useMemo(() => {
    const m: Record<string, number> = {};
    Object.values(docs).forEach((d) => d.segments.forEach((s) => { m[s[2]] = (m[s[2]] ?? 0) + 1; }));
    return CLASSES.filter((c) => m[c.id]).map((c) => ({ id: c.id, n: m[c.id] }));
  }, [docs]);
  const totalMin = Object.values(docs).reduce((a, d) => a + d.duration, 0) / 60;

  return (
    <>
      <h1>Results on the sample videos</h1>
      <p className="lead">Every sample video, annotated by our own renderer: tracked road users, the scene map, live event banners, the event timeline and the causal risk bar.</p>
      {err && <div className="callout">Could not load results: {err}. Run <code>python tools/export_site.py --videos samples</code>.</div>}

      <div className="grid cols-3" style={{ margin: "16px 0" }}>
        <div className="card stat"><div className="value">{index.length}</div><div className="label">sample videos</div></div>
        <div className="card stat"><div className="value">{Object.values(docs).reduce((a, d) => a + d.segments.length, 0)}</div><div className="label">events detected</div></div>
        <div className="card stat"><div className="value">{totalMin ? (Object.values(docs).reduce((a, d) => a + d.segments.length, 0) / totalMin).toFixed(1) : "–"}</div><div className="label">events per minute of footage</div></div>
      </div>

      {perClass.length > 0 && (
        <div className="card" style={{ marginBottom: 16 }}>
          <h3>Operator view: events by class, all samples</h3>
          <Bars values={perClass.map((p) => p.n)} labels={perClass.map((p) => className(p.id))} color="var(--series-1)" height={200} />
        </div>
      )}

      <div className="tabs">
        {index.map((v) => (
          <button key={v.id} className={sel === v.id ? "on" : ""} onClick={() => setSel(v.id)}>
            {v.file} <span className="muted">· {v.events} events</span>
          </button>
        ))}
      </div>
      {doc ? <VideoPanel key={sel} doc={doc} videoUrl={doc.media?.annotated ? `${BASE}/${doc.media.annotated}` : undefined} />
           : !err && <p className="muted">Loading…</p>}

      {doc && doc.raw.length > 0 && (
        <div className="card" style={{ marginTop: 16 }}>
          <h3>Evidence behind each event</h3>
          <p className="muted">Raw rule hits before merging: which tracked objects fired which rule, and the rule&apos;s confidence.</p>
          <div className="table-scroll" style={{ maxHeight: 300, overflowY: "auto" }}>
            <table className="tbl">
              <thead><tr><th>Class</th><th>Start</th><th>End</th><th>Track IDs</th><th>Score</th><th>Note</th></tr></thead>
              <tbody>{doc.raw.slice(0, 200).map((r, i) => (
                <tr key={i}><td><i className="dot" style={{ background: classColor(r.label) }} /> {className(r.label)}</td>
                  <td className="mono">{r.start.toFixed(2)}</td><td className="mono">{r.end.toFixed(2)}</td>
                  <td className="mono">{r.tracks.join(", ")}</td><td>{r.score.toFixed(2)}</td><td>{r.note}</td></tr>
              ))}</tbody>
            </table>
          </div>
        </div>
      )}

      <h2>How we tuned without labels</h2>
      <p>
        We had no ground truth for the samples, so we reviewed random rule hits on the video, found why the wrong ones fired, and fixed
        the rule, never a single clip. Every change is a general condition (see the failure cases below). The last column is our own visual
        judgement on a random sample, not a benchmark.
      </p>
      <div className="card table-scroll" style={{ marginBottom: 16 }}>
        <table className="tbl">
          <thead><tr><th>Class</th><th>Raw hits before</th><th>After</th><th>Looked real before</th><th>Looked real after</th></tr></thead>
          <tbody>{SPOTCHECK.map((r) => (
            <tr key={r.cls}><td><i className="dot" style={{ background: classColor(r.cls) }} /> {className(r.cls)}</td>
              <td className="mono">{r.before}</td><td className="mono">{r.after}</td><td>{r.looked_real_before}</td><td>{r.looked_real_after}</td></tr>
          ))}</tbody>
        </table>
      </div>

      <h2>Honest failure cases</h2>
      <div className="grid cols-2">
        {FAILURES.map((f) => (
          <div key={f.title} className="card"><h3>{f.title}</h3><p>{f.text}</p><p className="muted">What we did about it: {f.fix}</p></div>
        ))}
      </div>
    </>
  );
}
