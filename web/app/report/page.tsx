import { ABLATIONS, DIDNT, NEXT, WORKED } from "@/content/report";

export default function Report() {
  return (
    <>
      <h1>Technical report</h1>
      <p className="lead">
        What we built, what worked, what did not, and what we would do next. The full pipeline is in
        the repository; every threshold mentioned here lives in <code>configs/params.json</code>.
      </p>

      <h2>What we built</h2>
      <p>
        A YOLO11 detector runs on every second frame. For Part A, a ByteTrack-style tracker with offline stitching
        builds trajectories. We smooth them over 0.8 s and derive speed in object sizes per second. Thirteen rules
        turn trajectories, a hand-drawn scene map, a flow field learned from the video and a traffic-light colour
        reader into raw events. A post-processor unions same-class hits, merges gaps, drops blips and shifts
        boundaries by per-class offsets tuned on our own labels. Part B runs a separate online tracker on the frames
        it is given and scores time-to-collision, hard braking and traffic density into a causal risk.
      </p>
      <pre>{`pip install -r requirements.txt            # or: docker build -t trumen .
python run_submission.py --videos /data/test --out predictions.json
python evaluate.py --pred predictions.json --gt ground_truth.json`}</pre>

      <div className="grid cols-2">
        <div className="card">
          <h2 style={{ marginTop: 0 }}>What worked</h2>
          {WORKED.map((w) => <div key={w.title}><h3>{w.title}</h3><p>{w.text}</p></div>)}
        </div>
        <div className="card">
          <h2 style={{ marginTop: 0 }}>What did not</h2>
          {DIDNT.map((w) => <div key={w.title}><h3>{w.title}</h3><p>{w.text}</p></div>)}
          <h2>What we would do next</h2>
          <ul>{NEXT.map((n) => <li key={n}><p>{n}</p></li>)}</ul>
        </div>
      </div>

      {ABLATIONS.length > 0 && (
        <>
          <h2>Ablations</h2>
          <p>Part A on our own labels of the sample videos (tools/ablate.py). Small dev set: treat differences below ~0.03 as noise.</p>
          <div className="card table-scroll"><table className="tbl">
            <thead><tr><th>Variant</th><th>Score A (macro F1)</th><th>Micro F1 @0.5</th><th>Class-agnostic F1 @0.5</th></tr></thead>
            <tbody>{ABLATIONS.map((a) => <tr key={a.name}><td>{a.name}</td><td>{a.score_a.toFixed(3)}</td><td>{a.micro_f1_05.toFixed(3)}</td><td>{a.agnostic_f1_05.toFixed(3)}</td></tr>)}</tbody>
          </table></div>
        </>
      )}

      <h2>Runtime</h2>
      <p>
        On a T4, Part A (YOLO11m at 960 px, stride 2, batch 8, FP16) and Part B (same weights, one frame at a time,
        stride 2) each take about real time or less, well inside the 3× budget. The model loads and warms up when
        <code> solution.py</code> is imported, before the harness starts its clock. On CPU (the demo), the stride grows
        automatically from measured throughput so a 2-minute clip always finishes.
      </p>
      <h2>Reproducibility</h2>
      <p>
        Seeds are fixed and cuDNN is deterministic. The GPU stride is fixed (no timing-dependent choices), and the
        tracker is our own deterministic implementation. Two runs produce the same <code>predictions.json</code>.
        Our output on the samples is committed as <code>predictions_samples.json</code>.
      </p>
    </>
  );
}
