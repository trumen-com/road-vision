"use client";

import { useEffect, useMemo, useState } from "react";
import { Bars, LineChart } from "@/components/charts";
import { BASE, fetchJSON, type IndexEntry, type VideoDoc } from "@/lib/data";
import { FINDINGS } from "@/content/findings";

const COUNT_SERIES = [
  { key: "car", label: "Cars", color: "var(--series-1)" },
  { key: "truck", label: "Trucks", color: "var(--series-2)" },
  { key: "bus", label: "Buses", color: "var(--series-3)" },
  { key: "person", label: "People", color: "var(--series-4)" },
  { key: "motorcycle", label: "Motorbikes", color: "var(--series-5)" },
];

export default function EDA() {
  const [index, setIndex] = useState<IndexEntry[]>([]);
  const [docs, setDocs] = useState<Record<string, VideoDoc>>({});
  const [sel, setSel] = useState<string | null>(null);

  useEffect(() => {
    fetchJSON<{ videos: IndexEntry[] }>("data/index.json").then(async (d) => {
      setIndex(d.videos);
      setSel(d.videos[0]?.id ?? null);
      const all = await Promise.all(d.videos.map((v) => fetchJSON<VideoDoc>(`data/videos/${v.id}.json`)));
      setDocs(Object.fromEntries(all.map((doc, i) => [d.videos[i].id, doc])));
    }).catch(() => undefined);
  }, []);

  const doc = sel ? docs[sel] : undefined;
  const series = useMemo(() => COUNT_SERIES.filter((s) => doc?.counts?.some((c) => (c[s.key] ?? 0) > 0)), [doc]);
  const density = useMemo(() => {
    if (!doc?.counts) return { v: [] as number[], l: [] as string[] };
    const bins: number[] = [];
    doc.counts.forEach((c, i) => { const b = Math.floor(i / 10); bins[b] = (bins[b] ?? 0) + Number(c.moving_vehicles ?? 0); });
    return { v: bins.map((x) => Math.round(x / 10)), l: bins.map((_, i) => `${i * 10}s`) };
  }, [doc]);

  return (
    <>
      <h1>Exploratory analysis of the sample videos</h1>
      <p className="lead">What the camera sees, and the findings that shaped the solution.</p>

      <h2>Findings that shaped the solution</h2>
      <div className="grid cols-2">
        {FINDINGS.map((f) => <div key={f.title} className="card"><h3>{f.title}</h3><p>{f.text}</p><p className="muted">→ {f.impact}</p></div>)}
      </div>

      <h2>Video properties</h2>
      <div className="card table-scroll">
        <table className="tbl">
          <thead><tr><th>File</th><th>Resolution</th><th>fps</th><th>Duration</th><th>Mean brightness</th><th>Tracks by class</th></tr></thead>
          <tbody>{index.map((v) => {
            const d = docs[v.id];
            const b = d?.frame_stats?.length ? d.frame_stats.reduce((a, s) => a + s.brightness, 0) / d.frame_stats.length : null;
            return <tr key={v.id}><td className="mono">{v.file}</td><td>{v.width}×{v.height}</td><td>{v.fps.toFixed(2)}</td>
              <td>{v.duration.toFixed(1)} s</td><td>{b !== null ? b.toFixed(0) + " / 255" : "–"}</td>
              <td>{d?.n_tracks ? Object.entries(d.n_tracks).map(([k, n]) => `${k} ${n}`).join(" · ") : "–"}</td></tr>;
          })}</tbody>
        </table>
      </div>

      <h2>Per video</h2>
      <div className="tabs">{index.map((v) => <button key={v.id} className={sel === v.id ? "on" : ""} onClick={() => setSel(v.id)}>{v.file}</button>)}</div>
      {doc && (
        <div className="grid" style={{ gap: 16 }}>
          <div className="card">
            <h3>Road users in view over time</h3>
            <p className="muted">Distinct tracked objects per second, by class.</p>
            <LineChart data={doc.counts ?? []} xKey="t" series={series} yLabel="objects per second" />
          </div>
          <div className="grid cols-2">
            <div className="card">
              <h3>Lighting</h3>
              <p className="muted">Mean frame brightness (0–255) per second: day/night and exposure changes.</p>
              <LineChart data={(doc.frame_stats ?? []) as unknown as Record<string, number>[]} xKey="t"
                         series={[{ key: "brightness", label: "Brightness", color: "var(--series-4)" }]} height={170} />
            </div>
            <div className="card">
              <h3>Traffic density</h3>
              <p className="muted">Average moving vehicles in view, per 10-second window.</p>
              <Bars values={density.v} labels={density.l} height={170} />
            </div>
          </div>
          <div className="grid cols-2">
            <div className="card"><h3>Motion heatmap</h3><p className="muted">Accumulated frame differences: where things move.</p>
              {doc.media?.motion && <img className="fig" src={`${BASE}/${doc.media.motion}`} alt="Motion heatmap" />}</div>
            <div className="card"><h3>Vehicle trajectories</h3><p className="muted">Every track, coloured by travel direction; people in white.</p>
              {doc.media?.trajectories && <img className="fig" src={`${BASE}/${doc.media.trajectories}`} alt="Trajectories" />}</div>
          </div>
          <div className="grid cols-2">
            <div className="card"><h3>Learned lane directions</h3><p className="muted">Dominant flow per cell; the two colours are the two directions of travel found automatically.</p>
              {doc.media?.flow && <img className="fig" src={`${BASE}/${doc.media.flow}`} alt="Flow field" />}</div>
            <div className="card"><h3>Vehicle speed distribution</h3><p className="muted">Relative speed in object sizes per second (all vehicle samples). The spike near 0 is stationary traffic.</p>
              <Bars values={doc.speed_hist ?? []} labels={(doc.speed_hist ?? []).map((_, i) => (i * 0.25).toFixed(2))} color="var(--series-3)" /></div>
          </div>
        </div>
      )}
    </>
  );
}
