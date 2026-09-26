// Pipeline diagram (inline SVG, theme-aware). Left: Part A (offline). Right: Part B (causal).

type Box = { x: number; y: number; w: number; h: number; title: string; sub: string; kind: "input" | "learned" | "rule" | "out" };

const BOXES: Record<string, Box> = {
  cam: { x: 350, y: 10, w: 300, h: 56, title: "Road camera", sub: "fixed CCTV, 25 fps", kind: "input" },
  det: { x: 330, y: 100, w: 340, h: 56, title: "YOLO11 detector", sub: "road users, lights, animals", kind: "learned" },
  trkA: { x: 40, y: 200, w: 300, h: 56, title: "Tracker + stitching (Part A)", sub: "every 2nd frame, whole video", kind: "rule" },
  diary: { x: 40, y: 290, w: 300, h: 56, title: "Path diary", sub: "smoothed routes, speeds, headings", kind: "rule" },
  scene: { x: 380, y: 200, w: 250, h: 56, title: "Scene map", sub: "drawn once: lanes, stop lines…", kind: "input" },
  flow: { x: 380, y: 290, w: 250, h: 56, title: "Flow field + signal reader", sub: "legal directions, light colour", kind: "learned" },
  rules: { x: 120, y: 390, w: 420, h: 56, title: "Rule checker", sub: "13 event rules on trajectories", kind: "rule" },
  post: { x: 120, y: 480, w: 420, h: 56, title: "Segment post-processing", sub: "union, gap merge, blip filter, tuned offsets", kind: "rule" },
  events: { x: 170, y: 570, w: 320, h: 50, title: "[start, end, label] events", sub: "", kind: "out" },
  trkB: { x: 680, y: 200, w: 280, h: 56, title: "Online tracker (Part B)", sub: "own pass, past frames only", kind: "rule" },
  risk: { x: 680, y: 330, w: 280, h: 56, title: "Danger forecaster", sub: "time-to-collision, hard braking", kind: "rule" },
  alarm: { x: 700, y: 570, w: 240, h: 50, title: "Risk per frame → alarm ≥ 0.5", sub: "", kind: "out" },
};

const EDGES: [string, string][] = [
  ["cam", "det"], ["det", "trkA"], ["det", "trkB"], ["trkA", "diary"], ["diary", "rules"], ["scene", "rules"],
  ["flow", "rules"], ["diary", "flow"], ["rules", "post"], ["post", "events"], ["trkB", "risk"], ["risk", "alarm"],
];

const FILL: Record<Box["kind"], string> = {
  input: "var(--surface-2)", learned: "color-mix(in srgb, var(--series-1) 16%, var(--surface))",
  rule: "color-mix(in srgb, var(--series-3) 14%, var(--surface))", out: "color-mix(in srgb, var(--series-2) 16%, var(--surface))",
};

function anchor(b: Box, other: Box): [number, number] {
  const cx = b.x + b.w / 2, cy = b.y + b.h / 2;
  const ox = other.x + other.w / 2, oy = other.y + other.h / 2;
  if (Math.abs(oy - cy) > b.h / 2 + 10) return [Math.max(b.x + 20, Math.min(b.x + b.w - 20, ox)), oy > cy ? b.y + b.h : b.y];
  return [ox > cx ? b.x + b.w : b.x, cy];
}

export default function Pipeline() {
  return (
    <svg viewBox="0 0 1000 670" style={{ width: "100%", height: "auto", display: "block" }} role="img"
         aria-label="Pipeline: camera, detector, Part A tracking and rules to events; Part B causal tracker and risk forecaster to alarms">
      <defs>
        <marker id="arr" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse">
          <path d="M0,0 L10,5 L0,10 z" fill="var(--text-3)" />
        </marker>
      </defs>
      <rect x={20} y={180} width={630} height={475} rx={14} fill="none" stroke="var(--line)" strokeDasharray="6 6" />
      <text x={34} y={645} style={{ fill: "var(--text-3)", fontSize: 12 }}>PART A · offline, whole video</text>
      <rect x={665} y={180} width={310} height={475} rx={14} fill="none" stroke="var(--line)" strokeDasharray="6 6" />
      <text x={679} y={645} style={{ fill: "var(--text-3)", fontSize: 12 }}>PART B · causal, frame by frame</text>
      {EDGES.map(([a, b]) => {
        const A = BOXES[a], B = BOXES[b];
        const [x1, y1] = anchor(A, B), [x2, y2] = anchor(B, A);
        return <line key={a + b} x1={x1} y1={y1} x2={x2} y2={y2} stroke="var(--text-3)" strokeWidth={1.5} markerEnd="url(#arr)" />;
      })}
      {Object.entries(BOXES).map(([k, b]) => (
        <g key={k}>
          <rect x={b.x} y={b.y} width={b.w} height={b.h} rx={10} fill={FILL[b.kind]} stroke="var(--line)" />
          <text x={b.x + b.w / 2} y={b.y + (b.sub ? 24 : 30)} textAnchor="middle" style={{ fill: "var(--text)", fontSize: 15, fontWeight: 600 }}>{b.title}</text>
          {b.sub && <text x={b.x + b.w / 2} y={b.y + 43} textAnchor="middle" style={{ fill: "var(--text-2)", fontSize: 12 }}>{b.sub}</text>}
        </g>
      ))}
      <g transform="translate(700,440)" style={{ fontSize: 12 }}>
        {([["learned", "Learned model"], ["rule", "Rules / geometry"], ["input", "Input / hand-drawn"], ["out", "Output"]] as const).map(([k, l], i) => (
          <g key={k} transform={`translate(0, ${i * 22})`}>
            <rect width={14} height={14} rx={3} fill={FILL[k]} stroke="var(--line)" />
            <text x={22} y={11} style={{ fill: "var(--text-2)" }}>{l}</text>
          </g>
        ))}
      </g>
    </svg>
  );
}
