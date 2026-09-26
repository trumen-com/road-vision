"use client";

// Small dependency-free SVG charts: event timeline, risk curve, multi-line, bars.
// Thin marks, recessive grid, hover tooltips; identity is always carried by a text label.

import { useEffect, useRef, useState } from "react";
import { classColor, className } from "@/lib/classes";
import { fmtTime, type Segment } from "@/lib/data";

function useWidth<T extends HTMLElement>(): [React.RefObject<T | null>, number] {
  const ref = useRef<T>(null);
  const [w, setW] = useState(600);
  useEffect(() => {
    if (!ref.current) return;
    const ro = new ResizeObserver(([e]) => setW(Math.max(260, Math.floor(e.contentRect.width))));
    ro.observe(ref.current);
    return () => ro.disconnect();
  }, []);
  return [ref, w];
}

function ticks(max: number, n = 6): number[] {
  if (max <= 0) return [0];
  const raw = max / n;
  const mag = 10 ** Math.floor(Math.log10(raw));
  const step = [1, 2, 5, 10].map((m) => m * mag).find((s) => s >= raw) ?? raw;
  const out = [];
  for (let v = 0; v <= max + 1e-9; v += step) out.push(Math.round(v * 1000) / 1000);
  return out;
}

/** Axis ticks for a value axis: always ends at or above the data maximum. */
function valueTicks(max: number, n = 4): number[] {
  const t = ticks(max, n);
  if (t.length > 1 && t[t.length - 1] < max) t.push(Math.round((t[t.length - 1] + (t[1] - t[0])) * 1000) / 1000);
  return t;
}

// ---------------------------------------------------------------- Timeline
export function Timeline({ segments, duration, time, onSeek }: {
  segments: Segment[]; duration: number; time?: number; onSeek?: (t: number) => void;
}) {
  const [ref, w] = useWidth<HTMLDivElement>();
  const [tip, setTip] = useState<{ x: number; y: number; text: string } | null>(null);
  const labels = Array.from(new Set(segments.map((s) => s[2])));
  const left = Math.min(150, w * 0.32), right = 8, rowH = 22, top = 4;
  const H = top + Math.max(1, labels.length) * rowH + 24;
  const x = (t: number) => left + (t / Math.max(duration, 1e-3)) * (w - left - right);
  const click = (ev: React.MouseEvent<SVGSVGElement>) => {
    if (!onSeek) return;
    const r = ev.currentTarget.getBoundingClientRect();
    const px = ev.clientX - r.left;
    if (px < left) return;
    onSeek(Math.max(0, Math.min(duration, ((px - left) / (w - left - right)) * duration)));
  };
  return (
    <div ref={ref} style={{ position: "relative" }}>
      <svg className="chart" width={w} height={H} onClick={click} style={{ cursor: onSeek ? "pointer" : "default" }}
           role="img" aria-label="Event timeline">
        {ticks(duration, Math.max(3, Math.floor(w / 110))).map((t) => (
          <g key={t}>
            <line className="gridline" x1={x(t)} x2={x(t)} y1={top} y2={H - 20} />
            <text x={x(t)} y={H - 6} textAnchor="middle">{fmtTime(t)}</text>
          </g>
        ))}
        {labels.length === 0 && <text x={left} y={top + 15}>No events detected</text>}
        {labels.map((lab, r) => (
          <g key={lab}>
            <text x={0} y={top + r * rowH + 15} style={{ fill: "var(--text-2)", fontSize: 12 }}>{className(lab)}</text>
            {segments.filter((s) => s[2] === lab).map(([s, e], i) => (
              <rect key={i} x={x(s)} y={top + r * rowH + 4} width={Math.max(3, x(e) - x(s))} height={rowH - 8} rx={4}
                    fill={classColor(lab)}
                    onMouseMove={(ev) => {
                      const b = (ev.currentTarget.ownerSVGElement as SVGSVGElement).getBoundingClientRect();
                      setTip({ x: ev.clientX - b.left, y: ev.clientY - b.top, text: `${className(lab)} · ${fmtTime(s)}–${fmtTime(e)} (${(e - s).toFixed(1)} s)` });
                    }}
                    onMouseLeave={() => setTip(null)}
                    onClick={(ev) => { ev.stopPropagation(); onSeek?.(s); }} />
            ))}
          </g>
        ))}
        {time !== undefined && <line x1={x(time)} x2={x(time)} y1={0} y2={H - 20} stroke="var(--text)" strokeWidth={1.5} />}
      </svg>
      {tip && <div className="tooltip" style={{ left: Math.min(tip.x + 12, w - 220), top: tip.y + 12 }}>{tip.text}</div>}
    </div>
  );
}

// ---------------------------------------------------------------- Risk curve
export function RiskChart({ risk, duration, accidents = [], time, onSeek, theta = 0.5 }: {
  risk: [number, number][]; duration: number; accidents?: Segment[]; time?: number; onSeek?: (t: number) => void; theta?: number;
}) {
  const [ref, w] = useWidth<HTMLDivElement>();
  const [hover, setHover] = useState<number | null>(null);
  const H = 150, left = 34, right = 8, top = 8, bottom = 22;
  const x = (t: number) => left + (t / Math.max(duration, 1e-3)) * (w - left - right);
  const y = (v: number) => top + (1 - v) * (H - top - bottom);
  const path = risk.map(([t, v], i) => `${i ? "L" : "M"}${x(t).toFixed(1)},${y(v).toFixed(1)}`).join("");
  const area = risk.length ? `${path}L${x(risk[risk.length - 1][0])},${y(0)}L${x(risk[0][0])},${y(0)}Z` : "";
  const nearest = (t: number) => {
    let lo = 0, hi = risk.length - 1;
    while (lo < hi) { const m = (lo + hi) >> 1; if (risk[m][0] < t) lo = m + 1; else hi = m; }
    return risk[lo];
  };
  const move = (ev: React.MouseEvent<SVGSVGElement>) => {
    const r = ev.currentTarget.getBoundingClientRect();
    const t = ((ev.clientX - r.left - left) / (w - left - right)) * duration;
    setHover(t >= 0 && t <= duration ? t : null);
  };
  const hv = hover !== null && risk.length ? nearest(hover) : null;
  return (
    <div ref={ref} style={{ position: "relative" }}>
      <svg className="chart" width={w} height={H} onMouseMove={move} onMouseLeave={() => setHover(null)}
           onClick={() => hover !== null && onSeek?.(hover)} role="img" aria-label="Accident risk over time">
        {accidents.map(([s], i) => (
          <rect key={i} x={x(Math.max(0, s - 5))} y={top} width={x(s) - x(Math.max(0, s - 5))} height={H - top - bottom}
                fill="var(--fam-collision)" opacity={0.12} />
        ))}
        {[0, 0.25, 0.5, 0.75, 1].map((v) => (
          <g key={v}>
            <line className="gridline" x1={left} x2={w - right} y1={y(v)} y2={y(v)} />
            <text x={left - 6} y={y(v) + 4} textAnchor="end">{v}</text>
          </g>
        ))}
        <line x1={left} x2={w - right} y1={y(theta)} y2={y(theta)} stroke="var(--text-3)" strokeDasharray="4 4" />
        <text x={w - right} y={y(theta) - 4} textAnchor="end">alarm θ = {theta}</text>
        <path d={area} fill="var(--risk)" opacity={0.12} />
        <path d={path} fill="none" stroke="var(--risk)" strokeWidth={2} />
        {ticks(duration, Math.max(3, Math.floor(w / 110))).map((t) => (
          <text key={t} x={x(t)} y={H - 6} textAnchor="middle">{fmtTime(t)}</text>
        ))}
        {time !== undefined && <line x1={x(time)} x2={x(time)} y1={top} y2={H - bottom} stroke="var(--text)" strokeWidth={1.5} />}
        {hv && <>
          <line x1={x(hv[0])} x2={x(hv[0])} y1={top} y2={H - bottom} stroke="var(--text-3)" />
          <circle cx={x(hv[0])} cy={y(hv[1])} r={4} fill="var(--risk)" stroke="var(--surface)" strokeWidth={2} />
        </>}
      </svg>
      {hv && <div className="tooltip" style={{ left: Math.min(x(hv[0]) + 12, w - 160), top: 8 }}>
        {fmtTime(hv[0])} · risk <b>{hv[1].toFixed(2)}</b>
      </div>}
    </div>
  );
}

// ---------------------------------------------------------------- Multi-line chart
export function LineChart({ data, xKey, series, height = 200, yLabel }: {
  data: Record<string, number | null>[]; xKey: string; series: { key: string; label: string; color: string }[];
  height?: number; yLabel?: string;
}) {
  const [ref, w] = useWidth<HTMLDivElement>();
  const [hi, setHi] = useState<number | null>(null);
  const left = 38, right = 8, top = 10, bottom = 22, H = height;
  const xs = data.map((d) => Number(d[xKey]));
  const xmax = Math.max(1, ...xs);
  const ymax = Math.max(1, ...data.flatMap((d) => series.map((s) => Number(d[s.key] ?? 0))));
  const yt = valueTicks(ymax, 4);
  const ytop = yt[yt.length - 1] || 1;
  const x = (v: number) => left + (v / xmax) * (w - left - right);
  const y = (v: number) => top + (1 - v / ytop) * (H - top - bottom);
  const move = (ev: React.MouseEvent<SVGSVGElement>) => {
    const r = ev.currentTarget.getBoundingClientRect();
    const v = ((ev.clientX - r.left - left) / (w - left - right)) * xmax;
    let best = 0;
    xs.forEach((xv, i) => { if (Math.abs(xv - v) < Math.abs(xs[best] - v)) best = i; });
    setHi(data.length ? best : null);
  };
  return (
    <div ref={ref} style={{ position: "relative" }}>
      {series.length > 1 && (
        <div className="legend">{series.map((s) => <span key={s.key}><i className="dot" style={{ background: s.color }} />{s.label}</span>)}</div>
      )}
      <svg className="chart" width={w} height={H} onMouseMove={move} onMouseLeave={() => setHi(null)} role="img" aria-label={yLabel}>
        {yt.map((v) => (
          <g key={v}><line className="gridline" x1={left} x2={w - right} y1={y(v)} y2={y(v)} /><text x={left - 6} y={y(v) + 4} textAnchor="end">{v}</text></g>
        ))}
        {ticks(xmax, Math.max(3, Math.floor(w / 110))).map((t) => <text key={t} x={x(t)} y={H - 6} textAnchor="middle">{fmtTime(t)}</text>)}
        {series.map((s) => (
          <path key={s.key} fill="none" stroke={s.color} strokeWidth={2} strokeLinejoin="round"
                d={data.map((d, i) => `${i ? "L" : "M"}${x(Number(d[xKey])).toFixed(1)},${y(Number(d[s.key] ?? 0)).toFixed(1)}`).join("")} />
        ))}
        {hi !== null && <line x1={x(xs[hi])} x2={x(xs[hi])} y1={top} y2={H - bottom} stroke="var(--text-3)" />}
        {hi !== null && series.map((s) => (
          <circle key={s.key} cx={x(xs[hi])} cy={y(Number(data[hi][s.key] ?? 0))} r={4} fill={s.color} stroke="var(--surface)" strokeWidth={2} />
        ))}
      </svg>
      {hi !== null && (
        <div className="tooltip" style={{ left: Math.min(x(xs[hi]) + 12, w - 180), top: 24 }}>
          <div className="muted">{fmtTime(xs[hi])}</div>
          {series.map((s) => (
            <div key={s.key}><i className="dot" style={{ background: s.color }} /> {s.label}: <b>{data[hi][s.key] ?? "–"}</b></div>
          ))}
        </div>
      )}
    </div>
  );
}

// ---------------------------------------------------------------- Bars
export function Bars({ values, labels, color = "var(--series-1)", height = 180, unit = "" }: {
  values: number[]; labels: string[]; color?: string; height?: number; unit?: string;
}) {
  const [ref, w] = useWidth<HTMLDivElement>();
  const [hi, setHi] = useState<number | null>(null);
  const left = 38, right = 8, top = 10, bottom = 26, H = height;
  const yt = valueTicks(Math.max(1, ...values), 4);
  const ytop = yt[yt.length - 1] || 1;
  const bw = (w - left - right) / Math.max(1, values.length);
  const y = (v: number) => top + (1 - v / ytop) * (H - top - bottom);
  const every = Math.ceil(values.length / Math.max(1, Math.floor(w / 60)));
  return (
    <div ref={ref} style={{ position: "relative" }}>
      <svg className="chart" width={w} height={H} role="img">
        {yt.map((v) => <g key={v}><line className="gridline" x1={left} x2={w - right} y1={y(v)} y2={y(v)} /><text x={left - 6} y={y(v) + 4} textAnchor="end">{v}</text></g>)}
        {values.map((v, i) => (
          <g key={i} onMouseEnter={() => setHi(i)} onMouseLeave={() => setHi(null)}>
            <rect x={left + i * bw} y={top} width={bw} height={H - top - bottom} fill="transparent" />
            <rect x={left + i * bw + 1} y={y(v)} width={Math.max(1, bw - 2)} height={Math.max(0, y(0) - y(v))} rx={Math.min(4, bw / 3)} fill={color} opacity={hi === null || hi === i ? 1 : 0.55} />
            {i % every === 0 && <text x={left + i * bw + bw / 2} y={H - 8} textAnchor="middle">{labels[i]}</text>}
          </g>
        ))}
      </svg>
      {hi !== null && <div className="tooltip" style={{ left: Math.min(left + hi * bw + 10, w - 140), top: 6 }}>{labels[hi]}: <b>{values[hi]}{unit}</b></div>}
    </div>
  );
}
