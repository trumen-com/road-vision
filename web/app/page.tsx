import Link from "next/link";
import Pipeline from "@/components/Pipeline";
import { CLASSES, FAMILY_COLOR } from "@/lib/classes";

const MODELS = [
  ["YOLO11m (COCO)", "Detects people, bikes, motorbikes, cars, buses, trucks, traffic lights and animals", "Ultralytics, AGPL-3.0", "Learned (pre-trained, not fine-tuned)"],
  ["ByteTrack-style tracker", "Two-stage association + Kalman filter; our own implementation", "Method: Zhang et al. 2022 (MIT)", "Algorithmic"],
  ["Flow field", "Dominant direction per image cell, learned from moving vehicles", "Ours", "Learned from the video itself"],
  ["Traffic-light reader", "HSV colour of the lit lamp inside each signal ROI", "Ours", "Rule-based"],
  ["Event rules", "One rule per class on trajectories + scene map", "Ours", "Rule-based, thresholds tuned on our dev labels"],
  ["Risk forecaster", "Causal time-to-collision + hard braking + density", "Ours", "Rule-based, calibrated to the metric"],
];

export default function Home() {
  return (
    <>
      <section style={{ padding: "12px 0 8px" }}>
        <span className="pill">WIUT Hackathon 2026 · Computer Vision track</span>
        <h1 style={{ marginTop: 14 }}>Traffic events and early crash warnings from one fixed road camera</h1>
        <p className="lead">
          We turn CCTV footage into a list of timed traffic events (14 classes) and a frame-by-frame accident-risk
          score that looks only at the past. Detection is learned; everything after it is geometry and physics
          you can audit, which is what a traffic centre needs when an alert fires.
        </p>
        <div style={{ display: "flex", gap: 10, flexWrap: "wrap" }}>
          <Link className="btn" href="/demo/">Try the live demo</Link>
          <Link className="btn secondary" href="/results/">See results on the samples</Link>
        </div>
      </section>

      <div className="grid cols-4" style={{ marginTop: 28 }}>
        <div className="card stat"><div className="value">13 / 14</div><div className="label">event classes handled</div></div>
        <div className="card stat"><div className="value">≤ 1.1×</div><div className="label">real time per part on a T4 (budget 3×)</div></div>
        <div className="card stat"><div className="value">0</div><div className="label">future frames used by the risk score</div></div>
        <div className="card stat"><div className="value">1 cmd</div><div className="label"><code>python run_submission.py</code>, fully offline</div></div>
      </div>

      <h2>Pipeline</h2>
      <p>
        One detection pass feeds two independent branches. Part A (left) sees the whole video and may smooth over
        time; Part B (right) runs its own tracker frame by frame, so the risk score is causal by construction.
      </p>
      <div className="card"><Pipeline /></div>

      <h2>How each event is decided</h2>
      <p>Color marks the family: {Object.entries(FAMILY_COLOR).map(([f, c]) => (
        <span key={f} className="pill" style={{ marginRight: 6 }}><i className="dot" style={{ background: c }} />{f}</span>
      ))}</p>
      <div className="card table-scroll">
        <table className="tbl">
          <thead><tr><th>Class</th><th>Rule</th><th>Needs from the scene map</th></tr></thead>
          <tbody>
            {CLASSES.map((c) => (
              <tr key={c.id}>
                <td style={{ whiteSpace: "nowrap" }}><i className="dot" style={{ background: FAMILY_COLOR[c.family] }} /> <b style={{ color: "var(--text)" }}>{c.name}</b><div className="muted mono">{c.id}</div></td>
                <td>{c.rule}</td>
                <td>{c.needs}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>

      <h2>Models, data and what is learned</h2>
      <div className="card table-scroll">
        <table className="tbl">
          <thead><tr><th>Component</th><th>Role</th><th>Source / licence</th><th>Learned or rule-based</th></tr></thead>
          <tbody>{MODELS.map((m) => <tr key={m[0]}>{m.map((v, i) => <td key={i}>{i === 0 ? <b style={{ color: "var(--text)" }}>{v}</b> : v}</td>)}</tr>)}</tbody>
        </table>
      </div>
      <div className="callout">
        Why rules on top of a detector, not an end-to-end video model? The test set comes from the same camera but
        contains event types that the samples do not show. A learned classifier cannot learn a class it never saw;
        a rule on trajectories and a hand-drawn scene map transfers to it on day one. Every threshold is
        in <code>configs/params.json</code>, and every boundary shift is tuned against our own labels of the samples.
      </div>

      <h2>One map, many recordings</h2>
      <p>
        The scene map (carriageway, three zebra crossings, islands, median, bus stop, the stop line facing the camera
        and its signal head) is drawn once on a reference frame. The camera shifts slightly between recordings (up to
        ~60 px and ~1° in the samples), so every video is first registered to that frame with SIFT features and a
        RANSAC similarity transform, and the map moves with it. Footage from another camera does not match the
        reference and runs on what is learned from the video alone.
      </p>
      <img className="fig" src="/scene_map.jpg" alt="Scene map drawn on the reference frame" style={{ marginBottom: 12 }} />

      <h2>Speed units without calibration</h2>
      <p>
        Speed is measured in <i>object sizes per second</i> (pixel velocity ÷ √(box area)). A car near the camera and
        one far away get comparable numbers without a homography, so a single threshold works across the whole frame.
        Contact is tested on ground footprints normalised by the pair&apos;s size, so vehicles that only overlap in the
        image (different depths) do not count as touching.
      </p>
    </>
  );
}
