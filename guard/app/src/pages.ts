// Content pages. Test results are rendered from results/results.json, written by eval/report.py: the site shows only
// numbers that the evaluation produced.
interface Cond { name: string; n: number; n_pos: number; map: number; ci: [number, number] }
interface Results {
  system: string; n_clips: number; n_pos: number; n_neg: number; map: number; map_ci: [number, number]; auc: number;
  by_horizon: { horizon_s: number; n: number; ap: number; ci: [number, number] }[];
  by_condition: Cond[];
  operating_point: { warned_before_event: number; warned_before_event_by_horizon: Record<string, number>; negative_clips_with_warning: number; false_warnings_per_hour: number; negative_footage_minutes: number };
  baseline?: { name: string; map: number; auc: number };
  tradeoff?: { threshold: number; warned: number; fa_per_hour: number }[];
  visual_ablation?: { visual: number; fused: number; geometric: number; shipped: boolean };
  baseline_tradeoff?: { threshold: number; warned: number; fa_per_hour: number }[];
  shipping?: { threshold: number; warned: number; fa_per_hour: number; chosen_on: string };
  heuristic?: { map: number; auc: number };
  speed?: { node_cpu_ms: number; note: string };
  lead?: { n: number; median_lead_s: number; share_warned: number; median_alert_lead_s: number; fa_per_hour: number; neg_minutes: number };
}

const pct = (x: number) => `${Math.round(x * 100)}%`;
const f2 = (x: number) => x.toFixed(2);
const ci = (c?: [number, number]) => (c && c[0] != null ? `${f2(c[0])}–${f2(c[1])}` : "");

export function renderPages(base: string) {
  renderHow();
  renderBusiness();
  renderAbout();
  fetch(`${base}results/results.json`).then((r) => (r.ok ? r.json() : Promise.reject(r.status))).then((d: Results) => renderResults(d))
    .catch(() => { document.getElementById("resultsRoot")!.innerHTML = `<h1>Test results</h1><p>The evaluation output is not published yet.</p>`; });
}

function renderResults(d: Results) {
  const op = d.operating_point;
  const h = (s: number) => d.by_horizon.find((x) => x.horizon_s === s);
  document.getElementById("resultsRoot")!.innerHTML = `
  <h1>Tested on ${d.n_clips} real dashcam clips</h1>
  <p class="lead">Our test set is the public test split of the Nexar Collision Prediction dataset: ${d.n_pos} clips that end
  0.5, 1.0 or 1.5 seconds before a real collision or near miss, and ${d.n_neg} clips of normal driving, recorded by dashcams
  in real traffic, by day and night, in rain and on highways. The engine never saw these clips while we built it. Every number below is
  produced by <code>eval/report.py</code> from the same code that runs in your browser.</p>

  <div class="figure-row">
    <div class="figure"><b>${f2(d.map)}</b><span>mean average precision (Nexar protocol), 95% interval ${ci(d.map_ci)}</span></div>
    <div class="figure warn"><b>${pct(op.warned_before_event)}</b><span>of crash and near-miss clips get a warning before the event</span></div>
    <div class="figure bad"><b>${op.false_warnings_per_hour.toFixed(1)}</b><span>warnings per hour of normal driving (${op.negative_footage_minutes.toFixed(0)} min tested)</span></div>
    <div class="figure"><b>${f2(d.auc)}</b><span>ROC AUC, crash vs normal clips</span></div>
  </div>

  <h2>How early can it tell?</h2>
  <p>Each crash clip is cut before the event. The closer to the event, the more evidence there is; at 1.5 s before impact the task is hardest.</p>
  <div class="scroll"><table class="data"><thead><tr><th>Clip ends before the event</th><th class="num">Clips</th><th class="num">Average precision</th><th>95% interval</th><th class="num">Warned before the event</th></tr></thead><tbody>
  ${[0.5, 1, 1.5].map((s) => { const r = h(s); const w = op.warned_before_event_by_horizon[s.toFixed(3)]; return r ? `<tr><td>${s.toFixed(1)} s</td><td class="num">${r.n}</td><td class="num">${f2(r.ap)}</td><td>${ci(r.ci)}</td><td class="num">${w != null ? pct(w) : ""}</td></tr>` : ""; }).join("")}
  </tbody></table></div>

  ${d.lead ? `<h2>On full recordings</h2>
  <p>On ${d.lead.n} held-out full-length recordings from the training split (not used for tuning), the median warning came
  <b>${d.lead.median_lead_s.toFixed(1)} s</b> before the collision or near miss, for the ${pct(d.lead.share_warned)} of events it caught.
  Nexar's annotators marked the earliest moment a human could have been warned at a median of ${d.lead.median_alert_lead_s.toFixed(1)} s before the event.
  Normal driving in the same split produced ${d.lead.fa_per_hour.toFixed(1)} warnings per hour (${d.lead.neg_minutes.toFixed(0)} min).</p>` : ""}

  ${d.tradeoff ? `<h2>Every warning system trades misses for false alarms</h2>
  <p>Each point is one alert threshold: the share of crash clips warned before the event against false warnings per hour of normal driving.
  ${d.shipping ? `The app ships with the threshold chosen on the <b>training</b> split (${d.shipping.chosen_on}); on the test clips it warns before <b>${pct(d.shipping.warned)}</b> of crashes at <b>${d.shipping.fa_per_hour.toFixed(0)}</b> false warnings per hour.` : ""}
  Normal-driving clips in this dataset are dense city and highway traffic, so these rates are a stress test, and the app adds a speed gate (silent below 8 km/h) that the dataset cannot use.</p>
  ${tradeoffSvg(d.tradeoff, d.shipping, d.baseline_tradeoff)}
  ${d.baseline_tradeoff ? `<p>The grey dashed line is the naive baseline (warn when something big is in front) run as a live warning. ${budgetLine(d.tradeoff, d.baseline_tradeoff)}</p>` : ""}` : ""}

  <h2>Night, rain and road type</h2>
  <p>The dataset labels light, weather and scene for every clip. Small groups have wide intervals, which we show rather than hide.</p>
  <div class="scroll"><table class="data"><thead><tr><th>Condition</th><th class="num">Clips</th><th class="num">Crash clips</th><th class="num">mAP</th><th>95% interval</th></tr></thead><tbody>
  ${d.by_condition.map((c) => `<tr><td>${c.name}</td><td class="num">${c.n}</td><td class="num">${c.n_pos}</td><td class="num">${f2(c.map)}</td><td>${ci(c.ci)}</td></tr>`).join("")}
  </tbody></table></div>

  <h2>What each part adds</h2>
  <div class="scroll"><table class="data"><thead><tr><th>Version</th><th class="num">mAP</th><th class="num">ROC AUC</th></tr></thead><tbody>
  ${d.baseline ? `<tr><td>${d.baseline.name}</td><td class="num">${f2(d.baseline.map)}</td><td class="num">${f2(d.baseline.auc)}</td></tr>` : ""}
  ${d.heuristic ? `<tr><td>Tracking + time to contact + lane test (hand-set weights)</td><td class="num">${f2(d.heuristic.map)}</td><td class="num">${f2(d.heuristic.auc)}</td></tr>` : ""}
  <tr><td><b>Trumen Guard</b> (the above, with gradient-boosted trees over 2 s of features, learned on the Nexar training split)</td><td class="num"><b>${f2(d.map)}</b></td><td class="num"><b>${f2(d.auc)}</b></td></tr>
  ${d.visual_ablation ? `<tr><td>Tried, not shipped: a visual model (MobileNetV3 + temporal head) alone / averaged with the above</td><td class="num">${f2(d.visual_ablation.visual)} / ${f2(d.visual_ablation.fused)}</td><td></td></tr>` : ""}
  </tbody></table></div>
  <div class="note">For context: the winning entries of Nexar's Kaggle challenge reached about 0.9 mAP with large video networks running on
  server GPUs over the whole clip. Trumen Guard is a different trade-off: it runs on a phone, in real time, frame by frame, and explains every
  warning with an object, a lane and a time to contact.</div>

  <h2>Known failure cases</h2>
  <ul>
    <li><b>Our own sharp braking or a bump.</b> The whole image shifts and objects briefly look like they grow; the warning needs two consecutive frames above the threshold to filter most of these.</li>
    <li><b>Crashes from the side or behind.</b> A single forward camera cannot see them. Nexar positives include these, and they lower our recall.</li>
    <li><b>Very dark scenes without street lights.</b> The detector can miss unlit pedestrians. The twilight-and-dark group above is small (${(d.by_condition.find((c) => c.name.startsWith("Twilight")) || { n: 0 }).n} clips), so its interval is wide.</li>
    <li><b>Wrong lane setting.</b> If the phone is mounted off-centre and the lane is not adjusted, cars in the next lane can trigger cautions.</li>
  </ul>`;
}

type Pt = { threshold: number; warned: number; fa_per_hour: number };
function bestAt(c: Pt[], budget: number) { return Math.max(0, ...c.filter((p) => p.fa_per_hour <= budget).map((p) => p.warned)); }
function budgetLine(ours: Pt[], base: Pt[]) {
  const parts = [60, 100, 150].map((b) => `${pct(bestAt(ours, b))} vs ${pct(bestAt(base, b))} at ${b} per hour`);
  return `At the same false-warning rate Trumen Guard warns before more crashes: ${parts.join(", ")}. At very strict budgets (20–40 per hour) both catch few, which is why the pilot starts with tuning on local roads.`;
}
function tradeoffSvg(curve: Pt[], ship?: { warned: number; fa_per_hour: number }, base?: Pt[]) {
  const W = 640, H = 300, L = 56, B = 44, T = 14, Rr = 16;
  const maxFa = Math.min(400, Math.max(10, ...curve.map((c) => c.fa_per_hour)));
  const x = (fa: number) => L + (fa / maxFa) * (W - L - Rr), y = (w: number) => T + (1 - w) * (H - T - B);
  const pts = [...curve].filter((c) => c.fa_per_hour <= maxFa).sort((a, b) => a.fa_per_hour - b.fa_per_hour);
  const path = pts.map((c, i) => `${i ? "L" : "M"}${x(c.fa_per_hour).toFixed(1)},${y(c.warned).toFixed(1)}`).join("");
  const ticksX = [0, 0.25, 0.5, 0.75, 1].map((k) => Math.round(maxFa * k));
  return `<svg class="chart" viewBox="0 0 ${W} ${H}" role="img" aria-label="Share of crashes warned against false warnings per hour">
    ${[0, 0.25, 0.5, 0.75, 1].map((v) => `<line x1="${L}" x2="${W - Rr}" y1="${y(v)}" y2="${y(v)}" stroke="#D5D8D3"/><text x="${L - 8}" y="${y(v) + 4}" text-anchor="end" font-size="12" fill="#4A5058">${Math.round(v * 100)}%</text>`).join("")}
    ${ticksX.map((v) => `<text x="${x(v)}" y="${H - B + 18}" text-anchor="middle" font-size="12" fill="#4A5058">${v}</text>`).join("")}
    <text x="${(W + L) / 2}" y="${H - 6}" text-anchor="middle" font-size="13" fill="#4A5058">false warnings per hour of normal driving</text>
    <text x="14" y="${(H - B) / 2}" text-anchor="middle" font-size="13" fill="#4A5058" transform="rotate(-90 14 ${(H - B) / 2})">crashes warned</text>
    ${base ? `<path d="${[...base].filter((c) => c.fa_per_hour <= maxFa).sort((a, b) => a.fa_per_hour - b.fa_per_hour).map((c, i) => `${i ? "L" : "M"}${x(c.fa_per_hour).toFixed(1)},${y(c.warned).toFixed(1)}`).join("")}" fill="none" stroke="#9AA1A9" stroke-width="2" stroke-dasharray="5 4"/>` : ""}
    <path d="${path}" fill="none" stroke="#1D5AA6" stroke-width="2.5"/>
    ${pts.map((c) => `<circle cx="${x(c.fa_per_hour)}" cy="${y(c.warned)}" r="3" fill="#1D5AA6"><title>threshold ${c.threshold}: ${Math.round(c.warned * 100)}% warned, ${c.fa_per_hour.toFixed(0)}/h</title></circle>`).join("")}
    ${ship ? `<circle cx="${x(ship.fa_per_hour)}" cy="${y(ship.warned)}" r="7" fill="#F2C230" stroke="#23262B" stroke-width="2"/><text x="${x(ship.fa_per_hour) + 12}" y="${y(ship.warned) - 8}" font-size="13" font-weight="600" fill="#1C1F23">shipping setting</text>` : ""}
  </svg>`;
}

function renderHow() {
  document.getElementById("howRoot")!.innerHTML = `
  <h1>How it works</h1>
  <p class="lead">A phone camera, a small object detector and the physics of looming. Everything runs inside the browser on the phone:
  no video is uploaded, so there is no server cost per car and no privacy risk from stored footage.</p>

  <svg class="diagram" viewBox="0 0 920 210" role="img" aria-label="Pipeline: camera, detector, tracker, time to contact and lane test, risk model, warning">
    <defs><marker id="ar" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto"><path d="M0,0L10,5L0,10z" fill="#4A5058"/></marker></defs>
    ${[["Phone camera", "720p, 10–30 fps", 10], ["Detector", "YOLO11n, 512×288", 160], ["Tracker", "IoU + motion", 310], ["Time to contact", "looming fit, lane test", 460], ["Risk model", "learned on Nexar", 610], ["Warning", "sound, vibration", 760]]
      .map(([t, s, x], i) => `<g><rect x="${x}" y="60" width="140" height="78" rx="8" fill="${i === 5 ? "#F2C230" : "#fff"}" stroke="#1D5AA6" stroke-width="2"/>
      <text x="${(x as number) + 70}" y="94" text-anchor="middle" font-family="Barlow Semi Condensed" font-weight="700" font-size="19" fill="#1C1F23">${t}</text>
      <text x="${(x as number) + 70}" y="118" text-anchor="middle" font-family="IBM Plex Sans" font-size="13" fill="#4A5058">${s}</text>
      ${i < 5 ? `<line x1="${(x as number) + 142}" y1="99" x2="${(x as number) + 158}" y2="99" stroke="#4A5058" stroke-width="2" marker-end="url(#ar)"/>` : ""}</g>`).join("")}
    <text x="10" y="30" font-family="IBM Plex Sans" font-size="14" fill="#4A5058">Inside the phone browser (onnxruntime-web, WebGPU or WebAssembly)</text>
    <rect x="4" y="44" width="908" height="110" rx="10" fill="none" stroke="#D5D8D3" stroke-dasharray="6 5"/>
    <text x="10" y="186" font-family="IBM Plex Sans" font-size="14" fill="#4A5058">Optional, only with consent: the trip report (events, no video) is shared with the dealer or fleet manager.</text>
  </svg>

  <h2>Time to contact from a single camera</h2>
  <p>A car approaching at constant speed grows in the image. If its size is <i>s</i>, then d(ln <i>s</i>)/dt = 1 / TTC, where TTC is the time
  to contact. We do not need the distance or our own speed. For every tracked object we fit ln <i>s</i> against time over the last 0.8 s by least
  squares (median of pairwise slopes, so one jumpy box cannot fool it) and use the height, because a car that turns gets wider
  but not taller. The fit describes the middle of that window, so we subtract the elapsed half-window: under constant closing speed this
  gives the time to contact now, not 0.4 s ago. Unit tests on synthetic approaching cars check the estimate against the true value.</p>
  <h2>Is it in my path?</h2>
  <p>An object's sideways offset from our path divided by its own width does not change with distance. With a typical width per class
  (car 1.8 m, person 0.5 m, truck 2.4 m) that gives the offset in metres. An object is a threat if it is within half our width plus half
  its width of our path now, or will be when contact would happen: we track the offset over time to get its sideways speed in m/s. That
  catches cut-ins and a pedestrian stepping off the kerb, and ignores parked cars we drive past. When we turn, everything slides sideways;
  we subtract the scene-wide sideways motion so turning does not look like everyone cutting in.</p>
  <h2>From geometry to a warning</h2>
  <p>Per frame we compute nine features: shortest time to contact in the lane and entering it, looming rate, how close and how wide the
  nearest object in the lane is, how many there are, whether one is a pedestrian, cyclist, motorcyclist or animal, how much we are turning,
  and the hand-set threat score. A logistic model learned on the Nexar training split turns them into a probability. The warning fires when
  that probability stays above 0.5 for two frames (caution) and escalates above 0.75 (brake), then holds for half a second so one danger
  gives one alarm. Below 8 km/h (from GPS) it stays silent, so queues and parking do not nag.</p>

  <h2>Speed and cost</h2>
  <div class="scroll"><table class="data"><thead><tr><th>Where it runs</th><th>Speed</th><th>Cost</th></tr></thead><tbody>
    <tr><td>Phone or laptop browser, WebGPU</td><td>The frame rate is shown live on the Drive page</td><td>No per-car server cost: inference uses the phone's own chip</td></tr>
    <tr><td>Phone browser, WebAssembly (no GPU)</td><td>Slower; the time-to-contact fit adapts down to about 4 frames per second</td><td>Same</td></tr>
    <tr><td>Our test machine, Node.js, 2 CPU threads</td><td id="nodeSpeed">about 45 ms per frame for the detector</td><td>Used to run the 667-clip evaluation</td></tr>
  </tbody></table></div>
  <p>The site itself is static files (12 MB on first load, then cached). Hosting for a pilot costs nothing on GitHub Pages; at scale a CDN costs
  well under one cent per active car per month.</p>

  <h2>Night, rain and local vehicles</h2>
  <p>The test set has twilight and dark clips and rain clips, reported separately on the results page. The detector (COCO classes) knows cars,
  buses, trucks, motorcycles, bicycles, people and common animals, which covers the local fleet: Chevrolet Cobalt, Nexia and Damas, BYD and Toyota
  models are all just "car" to it. No number-plate reading is used, so local plates do not matter.</p>
  <h2>What is ours and what is not</h2>
  <ul>
    <li>Ours: the tracker, the looming-based time-to-contact fit, the lane and cut-in test, the turning compensation, the features, the learned risk model, the alert logic, the evaluation harness and the app.</li>
    <li>Not ours: the YOLO11n detector weights (Ultralytics, AGPL-3.0, pre-trained on COCO) and onnxruntime-web (Microsoft, MIT).</li>
    <li>From our elimination round: the idea of a causal risk score with time to contact, and the tracker design, rewritten for a moving camera.</li>
  </ul>`;
}

function renderBusiness() {
  document.getElementById("businessRoot")!.innerHTML = `
  <h1>A partner product for Royal Motors after-sales</h1>
  <p class="lead">Trumen Guard is sold and supported by the after-sales departments of Toyota Royal Motors and BYD Royal Motors. It changes one
  process: the vehicle handover. Every car that leaves a Royal Motors centre, new, serviced or traded in, can leave with a collision warning,
  set up in 30 seconds on the customer's phone.</p>

  <h2>The problem</h2>
  <div class="figure-row">
    <div class="figure bad"><b>2,188</b><span>people killed on Uzbekistan's roads in 2025, 9,226 crashes (Gazeta.uz, Jan 2026)</span></div>
    <div class="figure warn"><b>46%</b><span>of all crashes in Jan–Sep 2025 involved hitting a pedestrian (Gazeta.uz, Nov 2025)</span></div>
    <div class="figure"><b>≈ $2 bn</b><span>yearly cost of road crashes, about 3% of GDP (Asian Transport Observatory)</span></div>
    <div class="figure"><b>27%</b><span>fewer rear-end crashes in cars with forward collision warning (IIHS)</span></div>
  </div>
  <p>Most of Uzbekistan's 5.5 million registered vehicles have no collision warning. Factory systems come only with new, higher trims;
  aftermarket units such as Mobileye cost hundreds of dollars plus installation. Almost every driver already carries a capable camera and processor: their phone.</p>

  <h2>Who uses it, who pays</h2>
  <div class="cards">
    <div class="card"><h3>Driver</h3><p>Gets audible warnings about the car ahead and pedestrians stepping out, and a trip report. Pays nothing extra for the basic warning.</p></div>
    <div class="card"><h3>Royal Motors after-sales</h3><p>Owns the product. Sells the Safety Pack (mount, set-up, one year of trip reports) at handover and at service visits; keeps the customer connected to the dealer between visits.</p></div>
    <div class="card"><h3>Fleet customers</h3><p>Taxi and corporate fleets buying Toyota and BYD in bulk pay per vehicle per month for the fleet dashboard: warnings and near misses per driver, for coaching.</p></div>
  </div>

  <h2>One phone at every after-sales step</h2>
  <p>The same idea runs through the whole after-sales cycle: the customer's or the advisor's phone does the seeing, the analysis runs
  on the phone, nothing is uploaded, and every number is measured on real data before it is shown.</p>
  <div class="cards">
    <div class="card"><h3>Car comes in: trade-in</h3><p><a href="paint/">Paint Check</a>. The trade-in advisor photographs the 14 gaps between body panels. Panels whose paint is off the same way against every neighbour are marked for a gauge check before the price is set. Screening, not a verdict.</p></div>
    <div class="card"><h3>Car goes out: handover</h3><p>Trumen Guard. The customer leaves with a collision warning on their phone, set up in 30 seconds.</p></div>
    <div class="card"><h3>Service and claims</h3><p><a href="scan/">Trumen Scan</a>. A walk-around becomes a 3D model of that car, split into parts, for estimates and parts search.</p></div>
  </div>
  <p>Paint Check on real photos (held-out half: 425 public car photos, 921 joints): original paint is flagged at 26% of joints and 13% of panels are wrongly
  called repainted; simulated 2 ΔE whole-panel repaints are flagged for a gauge check 90% of the time. Real repaints are not yet measured: that is
  part of the pilot below.</p>

  <h2>Pilot in three months</h2>
  <ol>
    <li>Month 1: install in Royal Motors' own loaner and test-drive cars (20 cars); record local footage with consent; adjust thresholds on Tashkent streets, day and night.</li>
    <li>Month 2: offer the Safety Pack at handover in one Toyota and one BYD centre; one fleet customer with 50 vehicles. At the same centres' trade-in desks, run Paint Check on every trade-in next to a paint thickness gauge, to measure it on real repaints.</li>
    <li>Month 3: measure attach rate, warnings per 100 km, driver feedback and any avoided incidents; decide on roll-out.</li>
  </ol>
  <p>No vehicle modification and no type approval is needed: it is an app on the driver's phone, like a navigation app, and it only warns. Video stays on the phone, so no personal data is collected without consent.</p>

  <h2>Budget and payback</h2>
  <p>All figures are our estimates, to be confirmed with Royal Motors during the pilot.</p>
  <div class="scroll"><table class="data"><thead><tr><th>Pilot cost (3 months)</th><th class="num">USD</th></tr></thead><tbody>
    <tr><td>Two engineers, 3 months (local rates)</td><td class="num">9,000</td></tr>
    <tr><td>20 Android phones for cars without a suitable phone, 100 windscreen mounts</td><td class="num">3,400</td></tr>
    <tr><td>Local test data: driving, consent, labelling of 300 clips</td><td class="num">1,200</td></tr>
    <tr><td>Hosting, CDN, app-store fees</td><td class="num">300</td></tr>
    <tr><td>Legal and privacy review</td><td class="num">1,000</td></tr>
    <tr><td><b>Total</b></td><td class="num"><b>14,900</b></td></tr>
  </tbody></table></div>
  <div class="scroll"><table class="data"><thead><tr><th>Unit economics (assumptions)</th><th>Value</th></tr></thead><tbody>
    <tr><td>Safety Pack price at handover (mount, set-up, 12 months of reports)</td><td>$39</td></tr>
    <tr><td>Direct cost per pack (mount $8, 10 min of staff time $2, support $3)</td><td>$13</td></tr>
    <tr><td>Contribution per pack</td><td>$26</td></tr>
    <tr><td>Fleet dashboard</td><td>$3 per vehicle per month, about $0.05 running cost</td></tr>
    <tr><td>Handovers and service visits per year, two centres (40 cars a day, 300 days)</td><td>24,000</td></tr>
    <tr><td>Attach rate</td><td>5% → 1,200 packs a year → $31,200 contribution</td></tr>
    <tr><td>Fleets: 300 vehicles in year one</td><td>$10,800 a year</td></tr>
    <tr><td><b>Payback of the pilot</b></td><td><b>about 4 months after roll-out</b> ($14,900 ÷ $42,000 a year)</td></tr>
  </tbody></table></div>
  <p>Not counted: fewer crash repairs and downtime for fleets (IIHS measured 27% fewer rear-end crashes with forward collision warning; we assume
  half of that until local data proves more), and insurer discounts for drivers who keep warnings on.</p>

  <h2>Why another team cannot copy it in a weekend</h2>
  <ul>
    <li>A measured, causal, on-device pipeline: the tracking, looming fit, lane and turning logic and alert timing are tuned on hundreds of real crash clips, and every number is reproducible.</li>
    <li>The evaluation harness runs the exact browser code headless over the dataset, so changes are tested before they ship.</li>
    <li>Distribution and data: inside Royal Motors after-sales, each handover adds a car; trip reports (with consent) build a local near-miss dataset of Tashkent streets that no outside team has.</li>
  </ul>`;
}

function renderAbout() {
  document.getElementById("aboutRoot")!.innerHTML = `
  <h1>Team Trumen</h1>
  <div class="cards">
    <div class="card"><h3>Asilbek Shodmonov</h3><p>Product, web app and business case. ERP project manager and business analyst at Technogym Uzbekistan.</p></div>
    <div class="card"><h3>Komron Akmalov</h3><p>Computer vision. AI and computer vision engineer at zehnmind.ai.</p></div>
    <div class="card"><h3>Ashraf Shermatov</h3><p>Project management, data and evaluation. BSc Computer Science, Westminster International University in Tashkent.</p></div>
  </div>
  <h2>Data sources and licences</h2>
  <div class="scroll"><table class="data"><thead><tr><th>Source</th><th>Use</th><th>Licence</th></tr></thead><tbody>
    <tr><td>Nexar Collision Prediction Dataset (Moura, Zhu, Zvitia, 2025), via Hugging Face</td><td>Test set (public test split, 667 clips), training split for the risk model, sample clips on this site</td><td>Nexar open data licence: free use with attribution, no resale of the dataset</td></tr>
    <tr><td>YOLO11n weights (Ultralytics), pre-trained on COCO</td><td>Object detector</td><td>AGPL-3.0</td></tr>
    <tr><td>COCO dataset (via the YOLO11n pre-training)</td><td>Detector classes</td><td>CC BY 4.0 (annotations)</td></tr>
    <tr><td>onnxruntime-web, onnxruntime-node (Microsoft)</td><td>In-browser and evaluation inference</td><td>MIT</td></tr>
    <tr><td>coi-serviceworker (Guido Zuidhof)</td><td>Enables multi-threaded WebAssembly on static hosting</td><td>MIT</td></tr>
    <tr><td>Road-safety statistics: Gazeta.uz, Kun.uz, Asian Transport Observatory, IIHS</td><td>Business case</td><td>Cited, public reporting</td></tr>
  </tbody></table></div>
  <p>No personal data is collected. Video from the camera is processed in memory on the device and never stored or uploaded.</p>
  <p>Code: <a href="https://github.com/trumen-com/road-vision">github.com/trumen-com/road-vision</a>. Our elimination-round work (traffic events from a fixed camera) is <a href="./elimination/">here</a>.</p>`;
}
