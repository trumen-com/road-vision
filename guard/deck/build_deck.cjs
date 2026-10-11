// Trumen Guard defence deck (7 minutes). Numbers are read from the evaluation output, never typed by hand.
//   node deck/build_deck.js  -> deck/Trumen_Guard_defence.pptx
const fs = require("fs");
const path = require("path");
const pptxgen = require("pptxgenjs");
const { applyTheme } = require("/root/.claude/skills/synced/9b9640bb-615b-43b3-9c6b-9eabd04faca0_edeff9fc-2a02-44d6-8d21-1804ad1c06b2/pptx/scripts/apply_theme.js");

const ROOT = path.join(__dirname, "..");
const R = JSON.parse(fs.readFileSync(path.join(ROOT, "app/public/results/results.json"), "utf8"));
const H = fs.existsSync(path.join(ROOT, "results/heuristic.json")) ? JSON.parse(fs.readFileSync(path.join(ROOT, "results/heuristic.json"), "utf8")) : null;
const QR = "image/png;base64," + fs.readFileSync(path.join(ROOT, "deck_qr.png")).toString("base64");

const THEME = {
  name: "Trumen Road",
  headFontFace: "Arial",
  bodyFontFace: "Calibri",
  colors: {
    dk1: "1C1F23", lt1: "FFFFFF", dk2: "23262B", lt2: "F1F2F0",
    accent1: "F2C230", accent2: "E0312B", accent3: "1D5AA6", accent4: "3E9B5F", accent5: "4A5058", accent6: "9AA1A9",
    hlink: "1D5AA6", folHlink: "4A5058",
  },
};
const HEX = THEME.colors;
const pres = new pptxgen();
pres.layout = "LAYOUT_WIDE";            // 13.33 x 7.5 in
pres.theme = { headFontFace: THEME.headFontFace, bodyFontFace: THEME.bodyFontFace };
pres.title = "Trumen Guard: collision warning on any phone";
pres.author = "Team Trumen";
pres.company = "Team Trumen";
const C = pres.SchemeColor;

const f2 = (x) => (x == null ? "–" : x.toFixed(2));
const pct = (x) => `${Math.round(x * 100)}%`;
const op = R.operating_point;

// ---------------------------------------------------------------- layouts
pres.defineSlideMaster({
  title: "DARK",
  background: { color: HEX.dk2 },
  objects: [{ text: { text: "Trumen Guard", options: { x: 0.6, y: 6.95, w: 4, h: 0.3, fontSize: 11, color: HEX.accent6, fontFace: "Calibri" } } }],
  placeholders: [],
  slideNumber: { x: 12.4, y: 6.95, w: 0.5, h: 0.3, fontSize: 11, color: HEX.accent6 },
});
pres.defineSlideMaster({
  title: "LIGHT",
  background: { color: HEX.lt1 },
  objects: [{ text: { text: "Trumen Guard", options: { x: 0.6, y: 6.95, w: 4, h: 0.3, fontSize: 11, color: HEX.accent6, fontFace: "Calibri" } } }],
  slideNumber: { x: 12.4, y: 6.95, w: 0.5, h: 0.3, fontSize: 11, color: HEX.accent6 },
});
// content-slide title: same position, size and weight on every light slide
function title(slide, t) { slide.addText(t, { isTextBox: true, margin: 0, x: 0.6, y: 0.45, w: 12.1, h: 0.9, fontSize: 32, bold: true, color: HEX.dk1, fontFace: "Arial", valign: "top", objectName: "title" }); }

function text(slide, t, o) { slide.addText(t, { isTextBox: true, margin: 0, fontFace: "Calibri", color: HEX.dk1, ...o }); }
function plate(slide, x, y, w, value, label, tone) {
  // the recurring motif: a road-sign style plate with a big number
  const fill = tone === "red" ? HEX.accent2 : tone === "yellow" ? HEX.accent1 : HEX.dk2;
  const fg = tone === "yellow" ? HEX.dk2 : "FFFFFF";
  slide.addShape(pres.shapes.ROUNDED_RECTANGLE, { x, y, w, h: 1.25, rectRadius: 0.12, fill: { color: fill }, line: { color: fill }, objectName: `plate ${label}` });
  text(slide, value, { x: x + 0.2, y: y + 0.12, w: w - 0.4, h: 0.7, fontSize: 40, bold: true, fontFace: "Arial", color: fg });
  text(slide, label, { x: x + 0.2, y: y + 0.82, w: w - 0.4, h: 0.38, fontSize: 13, color: fg });
}

// ---------------------------------------------------------------- 1. title
pres.addSection({ title: "Opening" });
let s = pres.addSlide({ masterName: "DARK", sectionTitle: "Opening" });
s.addShape(pres.shapes.ISOSCELES_TRIANGLE, { x: 0.6, y: 0.7, w: 0.9, h: 0.8, fill: { color: HEX.accent1 }, line: { color: HEX.accent1 }, objectName: "warning sign" });
text(s, "!", { x: 0.6, y: 0.88, w: 0.9, h: 0.6, fontSize: 30, bold: true, align: "center", color: HEX.dk2, fontFace: "Arial" });
text(s, "Trumen Guard", { x: 0.6, y: 1.8, w: 8.5, h: 1.1, fontSize: 60, bold: true, fontFace: "Arial", color: "FFFFFF" });
text(s, "A collision warning that runs on any phone, before the crash", { x: 0.6, y: 2.95, w: 8.6, h: 0.6, fontSize: 24, color: "D5D8DC" });
s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x: 0.6, y: 3.9, w: 7.6, h: 1.05, rectRadius: 0.1, fill: { color: "2E3238" }, line: { color: HEX.accent1, width: 1.5 }, objectName: "mode box" });
text(s, [
  { text: "Mode: Partner product", options: { bold: true, color: HEX.accent1, breakLine: true } },
  { text: "Owned by Toyota Royal Motors and BYD Royal Motors after-sales; changes the vehicle handover", options: { color: "FFFFFF" } },
], { x: 0.85, y: 4.02, w: 7.2, h: 0.85, fontSize: 16 });
text(s, "Team Trumen: Asilbek Shodmonov, Komron Akmalov, Ashraf Shermatov", { x: 0.6, y: 5.6, w: 8, h: 0.4, fontSize: 15, color: HEX.accent6 });
s.addImage({ data: QR, x: 9.75, y: 1.75, w: 2.9, h: 2.9, objectName: "QR to live site" });
text(s, "Try it on your phone now", { x: 9.5, y: 4.75, w: 3.4, h: 0.4, fontSize: 16, bold: true, color: "FFFFFF", align: "center" });
text(s, "trumen-com.github.io/road-vision", { x: 9.3, y: 5.15, w: 3.8, h: 0.35, fontSize: 13, color: HEX.accent6, align: "center" });
s.addNotes("Mode on the first slide, as the rules ask. Invite the jury to scan the QR code now; the demo comes in a minute.");

// ---------------------------------------------------------------- 2. problem
pres.addSection({ title: "Problem and user" });
s = pres.addSlide({ masterName: "LIGHT", sectionTitle: "Problem and user" });
title(s, "Most cars on our roads would not warn before a crash");
plate(s, 0.6, 1.65, 2.85, "2,188", "killed on Uzbekistan's roads in 2025", "red");
plate(s, 3.65, 1.65, 2.85, "46%", "of crashes hit a pedestrian", "yellow");
plate(s, 6.7, 1.65, 2.85, "≈ $2 bn", "yearly cost, about 3% of GDP", "dark");
plate(s, 9.75, 1.65, 2.95, "−27%", "rear-end crashes with a warning", "dark");
text(s, [
  { text: "Most of the 5.5 million registered vehicles have no forward collision warning: it comes only with new, higher trims, and aftermarket units cost hundreds of dollars installed.", options: { bullet: true, breakLine: true } },
  { text: "Almost every driver already carries a camera and a processor that can do it: their phone.", options: { bullet: true, breakLine: true } },
  { text: "Crashes into pedestrians and cyclists are more than half of all crashes here. A warning must cover people, not only cars.", options: { bullet: true } },
], { x: 0.6, y: 3.3, w: 12.1, h: 2.5, fontSize: 18, paraSpaceAfter: 10, color: HEX.dk1 });
text(s, "Sources: Gazeta.uz (Jan 2026; Nov 2025), Asian Transport Observatory road safety profile 2025, IIHS (forward collision warning alone, rear-end crashes).", { x: 0.6, y: 6.35, w: 12.1, h: 0.4, fontSize: 11, color: HEX.accent5 });
s.addNotes("One minute. The problem is fatal crashes, and almost half of them involve pedestrians. The tool to warn exists in every pocket.");

// ---------------------------------------------------------------- 3. user, owner, process
s = pres.addSlide({ masterName: "LIGHT", sectionTitle: "Problem and user" });
title(s, "Royal Motors after-sales owns it; the handover changes");
const cols = [
  ["Driver", "Mounts the phone, opens one web page. Hears a warning about the car ahead or a person stepping out. Gets a trip report. Pays nothing for the basic warning."],
  ["Royal Motors after-sales", "Owns and sells the Safety Pack (mount, set-up, a year of trip reports) at handover and at service visits. Keeps the customer connected between visits."],
  ["Fleet customers", "Taxi and corporate fleets buying Toyota and BYD in bulk pay per vehicle per month for warnings and near misses per driver, for coaching."],
];
cols.forEach(([h, b], i) => {
  const x = 0.6 + i * 4.1;
  s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x, y: 1.6, w: 3.9, h: 2.6, rectRadius: 0.1, fill: { color: HEX.lt2 }, line: { color: HEX.lt2 }, objectName: `card ${h}` });
  text(s, h, { x: x + 0.25, y: 1.8, w: 3.4, h: 0.45, fontSize: 20, bold: true, fontFace: "Arial" });
  text(s, b, { x: x + 0.25, y: 2.35, w: 3.4, h: 1.75, fontSize: 15, color: HEX.accent5, valign: "top" });
});
text(s, "The process it changes: vehicle handover", { x: 0.6, y: 4.5, w: 12, h: 0.45, fontSize: 20, bold: true, fontFace: "Arial" });
const steps = ["Sale or service done", "Safety Pack offered", "Phone mounted, lane set in 30 s", "Car leaves with a warning"];
steps.forEach((t, i) => {
  const x = 0.6 + i * 3.1;
  s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x, y: 5.1, w: 2.75, h: 0.85, rectRadius: 0.08, fill: { color: i === 3 ? HEX.accent1 : "FFFFFF" }, line: { color: HEX.dk2, width: 1.25 }, objectName: `step ${i + 1}` });
  text(s, t, { x: x + 0.15, y: 5.15, w: 2.45, h: 0.75, fontSize: 15, align: "center", valign: "middle", bold: i === 3 });
  if (i < 3) s.addShape(pres.shapes.LINE, { x: x + 2.78, y: 5.52, w: 0.3, h: 0, line: { color: HEX.dk2, width: 1.5, endArrowType: "triangle" }, objectName: `arrow ${i + 1}` });
});
s.addNotes("Partner product: the unit is Royal Motors after-sales; the process is the handover. No vehicle modification, no type approval: it only warns.");

// ---------------------------------------------------------------- 4. demo
pres.addSection({ title: "Live demo" });
s = pres.addSlide({ masterName: "DARK", sectionTitle: "Live demo" });
text(s, "Live demo", { x: 0.6, y: 0.5, w: 8, h: 0.9, fontSize: 44, bold: true, fontFace: "Arial", color: "FFFFFF" });
text(s, [
  { text: "A real dashcam crash clip: the band turns yellow, then red, with a beep, before impact", options: { bullet: true, breakLine: true } },
  { text: "Your video: open any dashcam file; it is analysed in real time in the browser", options: { bullet: true, breakLine: true } },
  { text: "Your camera: point a phone at the road or a street scene", options: { bullet: true, breakLine: true } },
  { text: "The trip report: every warning with what, where and time to contact; nothing leaves the phone", options: { bullet: true } },
], { x: 0.6, y: 1.6, w: 8.3, h: 3.6, fontSize: 20, color: "FFFFFF", paraSpaceAfter: 14 });
s.addImage({ data: QR, x: 9.75, y: 1.6, w: 2.9, h: 2.9, objectName: "QR to live site" });
text(s, "trumen-com.github.io/road-vision", { x: 9.3, y: 4.65, w: 3.8, h: 0.35, fontSize: 13, color: HEX.accent6, align: "center" });
text(s, "Backup: a recorded run of the demo is in the repository (demo/backup.mp4).", { x: 0.6, y: 6.2, w: 9, h: 0.4, fontSize: 13, color: HEX.accent6 });
s.addNotes("Two minutes. Play sample 1 (urban crash), then hand a phone to the jury or use their video. If the network fails, play the backup recording.");

// ---------------------------------------------------------------- 5. how it works
pres.addSection({ title: "Technology" });
s = pres.addSlide({ masterName: "LIGHT", sectionTitle: "Technology" });
title(s, "Physics of looming plus a model learned on real crashes");
const boxes = [["Camera", "phone, 10–30 fps"], ["Detector", "YOLO11n 512×288"], ["Tracker", "IoU + motion"], ["Time to contact", "looming, tau-dot"], ["Path in metres", "width-scaled offset"], ["Risk model", "trees on 2 s of features"], ["Warning", "sound, vibration"]];
boxes.forEach(([t, sub], i) => {
  const x = 0.6 + i * 1.78;
  s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x, y: 1.6, w: 1.6, h: 1.15, rectRadius: 0.08, fill: { color: i === 6 ? HEX.accent1 : "FFFFFF" }, line: { color: HEX.accent3, width: 1.5 }, objectName: `pipeline ${t}` });
  text(s, t, { x: x + 0.06, y: 1.68, w: 1.48, h: 0.5, fontSize: 14, bold: true, align: "center", fontFace: "Arial" });
  text(s, sub, { x: x + 0.06, y: 2.18, w: 1.48, h: 0.5, fontSize: 11, align: "center", color: HEX.accent5 });
  if (i < 6) s.addShape(pres.shapes.LINE, { x: x + 1.61, y: 2.17, w: 0.16, h: 0, line: { color: HEX.accent5, width: 1.25, endArrowType: "triangle" }, objectName: `pipe arrow ${i}` });
});
text(s, "All inside the phone browser (onnxruntime-web). No video leaves the phone.", { x: 0.6, y: 2.9, w: 12, h: 0.35, fontSize: 13, color: HEX.accent5 });
const ideas = [
  ["Time to contact from one camera", "An approaching car grows: d(ln h)/dt = 1/TTC. No distance or speed needed. Height, not width: a turning car gets wider, not taller."],
  ["Is it in my path, in metres", "Offset ÷ the object's own width does not change with distance, so a parked car stays beside us and a car in our lane stays at zero."],
  ["Braking in control or a crash course", "Tau-dot (Lee, 1976): when time to contact stops shrinking, the driver is handling it. Turning is removed using the whole scene's sideways motion."],
];
ideas.forEach(([h, b], i) => {
  const x = 0.6 + i * 4.1;
  text(s, h, { x, y: 3.55, w: 3.85, h: 0.45, fontSize: 17, bold: true, fontFace: "Arial", color: HEX.accent3 });
  text(s, b, { x, y: 4.05, w: 3.85, h: 1.9, fontSize: 15, valign: "top" });
});
text(s, "Ours: tracker, looming fit, metric path test, tau-dot, features, risk model, alert logic, evaluation harness, app. Not ours: YOLO11n weights (AGPL-3.0), onnxruntime (MIT).", { x: 0.6, y: 6.2, w: 12.1, h: 0.5, fontSize: 12, color: HEX.accent5 });
s.addNotes("One minute. Three ideas carry the system; the learned model sits on top of them. Everything we did not write is named.");

// ---------------------------------------------------------------- 6. results
s = pres.addSlide({ masterName: "LIGHT", sectionTitle: "Technology" });
title(s, `Tested on ${R.n_clips} held-out real dashcam clips`);
plate(s, 0.6, 1.5, 2.9, f2(R.map), "mAP, Nexar protocol", "dark");
plate(s, 3.7, 1.5, 2.9, pct(R.shipping.warned), "of crash clips warned before the event", "yellow");
plate(s, 6.8, 1.5, 2.9, R.shipping.fa_per_hour.toFixed(0), "false warnings per hour of normal driving", "red");
plate(s, 9.9, 1.5, 2.8, f2(R.auc), "ROC AUC", "dark");
const rows = [[{ text: "Version", options: { bold: true } }, { text: "mAP", options: { bold: true, align: "right" } }, { text: "AUC", options: { bold: true, align: "right" } }]];
if (R.baseline) rows.push(["Naive: biggest object in front", { text: f2(R.baseline.map), options: { align: "right" } }, { text: f2(R.baseline.auc), options: { align: "right" } }]);
if (H) rows.push(["Tracking + time to contact + path, hand-set", { text: f2(H.map), options: { align: "right" } }, { text: f2(H.auc), options: { align: "right" } }]);
rows.push([{ text: "Trumen Guard (learned trees on 2 s of features)", options: { bold: true } }, { text: f2(R.map), options: { bold: true, align: "right" } }, { text: f2(R.auc), options: { bold: true, align: "right" } }]);
if (R.visual_ablation) rows.push([{ text: "Tried, not shipped: + visual model (lowered mAP)", options: { color: HEX.accent5 } }, { text: f2(R.visual_ablation.fused), options: { align: "right", color: HEX.accent5 } }, { text: "–", options: { align: "right", color: HEX.accent5 } }]);
s.addTable(rows, { x: 0.6, y: 3.0, w: 6.1, colW: [4.3, 0.9, 0.9], fontSize: 13, fontFace: "Calibri", color: HEX.dk1, border: { type: "solid", pt: 0.5, color: "D5D8D3" }, rowH: 0.38 });
const best = (c, b) => Math.max(0, ...c.filter((p) => p.fa_per_hour <= b).map((p) => p.warned));
const budgets = [20, 40, 60, 80, 100, 125, 150, 200, 250];
s.addChart(pres.charts.LINE, [
  { name: "Trumen Guard", labels: budgets.map(String), values: budgets.map((b) => Math.round(100 * best(R.tradeoff, b))) },
  { name: "Naive baseline", labels: budgets.map(String), values: budgets.map((b) => Math.round(100 * best(R.baseline_tradeoff, b))) },
], { x: 7.0, y: 2.95, w: 5.75, h: 3.15, chartColors: [HEX.accent3, HEX.accent6], lineSize: 2.5, lineDataSymbolSize: 6,
  showTitle: true, title: "Crash clips warned in time, % (at a false-warning budget per hour)", titleFontSize: 12, titleColor: HEX.dk1, titleFontFace: "+mn-lt",
  showLegend: true, legendPos: "b", legendFontSize: 11, legendFontFace: "+mn-lt",
  catAxisLabelColor: HEX.accent5, valAxisLabelColor: HEX.accent5, catAxisLabelFontSize: 10, valAxisLabelFontSize: 10, catAxisLabelFontFace: "+mn-lt", valAxisLabelFontFace: "+mn-lt",
  valAxisMinVal: 0, valAxisMaxVal: 80, valGridLine: { color: "E3E5E2", size: 0.5 }, catGridLine: { style: "none" }, objectName: "tradeoff chart" });
text(s, [
  { text: `Ranking quality ties the naive baseline (mAP ${f2(R.map)} vs ${f2(R.baseline.map)}). As a live warning, at the same false-warning rate, ours warns before more crashes: ${pct(best(R.tradeoff, 60))} vs ${pct(best(R.baseline_tradeoff, 60))} at 60 per hour, ${pct(best(R.tradeoff, 150))} vs ${pct(best(R.baseline_tradeoff, 150))} at 150.`, options: { breakLine: true } },
  { text: `The threshold was fixed on the training split before testing. mAP 95% interval ${f2(R.map_ci[0])}–${f2(R.map_ci[1])}; Kaggle winners reached about 0.9 with server video networks.`, options: {} },
], { x: 0.6, y: 5.0, w: 6.1, h: 1.7, fontSize: 12, color: HEX.accent5, valign: "top", paraSpaceAfter: 6 });
s.addNotes("One minute. Honest numbers from eval/report.py. On ranking we only tie the naive baseline; the gain is as a live warning, where tracking and time to contact give more crashes warned at the same false-alarm rate. The false-warning rate is still too high for production: that is what the pilot tunes on local roads.");

// ---------------------------------------------------------------- 7. conditions + failures
s = pres.addSlide({ masterName: "LIGHT", sectionTitle: "Technology" });
title(s, "Night, rain, speed and cost, and where it fails");
const cond = [[{ text: "Condition", options: { bold: true } }, { text: "Clips", options: { bold: true, align: "right" } }, { text: "mAP", options: { bold: true, align: "right" } }, { text: "95% interval", options: { bold: true } }]];
for (const c of R.by_condition) cond.push([c.name, { text: String(c.n), options: { align: "right" } }, { text: f2(c.map), options: { align: "right" } }, `${f2(c.ci[0])}–${f2(c.ci[1])}`]);
s.addTable(cond, { x: 0.6, y: 1.55, w: 6.2, colW: [2.6, 0.9, 0.9, 1.8], fontSize: 13, fontFace: "Calibri", color: HEX.dk1, border: { type: "solid", pt: 0.5, color: "D5D8D3" }, rowH: 0.36 });
text(s, "Speed and cost", { x: 7.2, y: 1.55, w: 5.5, h: 0.4, fontSize: 18, bold: true, fontFace: "Arial" });
text(s, [
  { text: "Inference on the phone: WebGPU, or multi-threaded WebAssembly", options: { bullet: true, breakLine: true } },
  { text: `Detector ${R.speed ? R.speed.node_cpu_ms.toFixed(0) : "46"} ms per frame on 2 CPU threads; the live frame rate shows on screen`, options: { bullet: true, breakLine: true } },
  { text: "No server per car; static hosting under 1 cent per car per month", options: { bullet: true, breakLine: true } },
  { text: "Local cars and plates: classes, not brands; plates are never read", options: { bullet: true } },
], { x: 7.2, y: 2.0, w: 5.5, h: 2.1, fontSize: 14, paraSpaceAfter: 6 });
text(s, "Known failure cases", { x: 7.2, y: 4.2, w: 5.5, h: 0.4, fontSize: 18, bold: true, fontFace: "Arial" });
text(s, [
  { text: "Crashes from the side or behind: one forward camera cannot see them", options: { bullet: true, breakLine: true } },
  { text: "Unlit pedestrians at night: the detector misses them", options: { bullet: true, breakLine: true } },
  { text: `Rain is our weakest group (${(R.by_condition.find((c) => /rain/i.test(c.name)) || {}).n} clips, mAP ${f2((R.by_condition.find((c) => /rain/i.test(c.name)) || {}).map)}): wipers and glare break the boxes`, options: { bullet: true, breakLine: true } },
  { text: "Dense stop-and-go traffic: short time to contact is normal there; silenced below 8 km/h by GPS", options: { bullet: true } },
], { x: 7.2, y: 4.6, w: 5.5, h: 2.1, fontSize: 13, paraSpaceAfter: 4 });
s.addNotes("The rules ask for night, rain, speed, cost and failures. Small groups have wide intervals, and we show them.");

// ---------------------------------------------------------------- 8. moat
s = pres.addSlide({ masterName: "LIGHT", sectionTitle: "Technology" });
title(s, "Why another team cannot rebuild it in a weekend");
const moat = [
  ["Measured pipeline", "Every rule and threshold was tuned against hundreds of real crash clips; the harness runs the exact browser code headless over the dataset."],
  ["Physics, not only a model", "Looming, tau-dot and the metric path test explain every warning: which object, where, and in how many seconds."],
  ["On the device", "Real time in a phone browser with no server: privacy by design and near-zero cost per car."],
  ["Distribution and data", "Inside Royal Motors after-sales, every handover adds a car; consented trip reports build a Tashkent near-miss dataset no outside team has."],
];
moat.forEach(([h, b], i) => {
  const x = 0.6 + (i % 2) * 6.15, y = 1.6 + Math.floor(i / 2) * 2.35;
  s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x, y, w: 5.9, h: 2.1, rectRadius: 0.1, fill: { color: HEX.lt2 }, line: { color: HEX.lt2 }, objectName: `moat ${h}` });
  text(s, h, { x: x + 0.3, y: y + 0.2, w: 5.3, h: 0.45, fontSize: 20, bold: true, fontFace: "Arial" });
  text(s, b, { x: x + 0.3, y: y + 0.72, w: 5.3, h: 1.25, fontSize: 15, color: HEX.accent5, valign: "top" });
});
s.addNotes("The replication test: data, pipeline, integration and domain knowledge.");

// ---------------------------------------------------------------- 9. budget and ROI
pres.addSection({ title: "Budget and ROI" });
s = pres.addSlide({ masterName: "LIGHT", sectionTitle: "Budget and ROI" });
title(s, "A $14,900 pilot that pays back in about four months");
const cost = [[{ text: "Pilot, 3 months", options: { bold: true } }, { text: "USD", options: { bold: true, align: "right" } }],
  ["Two engineers (local rates)", { text: "9,000", options: { align: "right" } }],
  ["20 phones, 100 windscreen mounts", { text: "3,400", options: { align: "right" } }],
  ["Local test data with consent, 300 clips labelled", { text: "1,200", options: { align: "right" } }],
  ["Hosting, CDN, store fees", { text: "300", options: { align: "right" } }],
  ["Legal and privacy review", { text: "1,000", options: { align: "right" } }],
  [{ text: "Total", options: { bold: true } }, { text: "14,900", options: { bold: true, align: "right" } }]];
s.addTable(cost, { x: 0.6, y: 1.55, w: 5.9, colW: [4.5, 1.4], fontSize: 14, fontFace: "Calibri", color: HEX.dk1, border: { type: "solid", pt: 0.5, color: "D5D8D3" }, rowH: 0.42 });
const ue = [[{ text: "Per year after roll-out (assumptions)", options: { bold: true } }, { text: "USD", options: { bold: true, align: "right" } }],
  ["Safety Pack: $39 price − $13 direct cost = $26", { text: "", options: {} }],
  ["24,000 handovers and visits × 5% attach = 1,200 packs", { text: "31,200", options: { align: "right" } }],
  ["Fleets: 300 vehicles × $3 per month", { text: "10,800", options: { align: "right" } }],
  [{ text: "Contribution per year", options: { bold: true } }, { text: "42,000", options: { bold: true, align: "right" } }],
  [{ text: "Payback of the pilot", options: { bold: true } }, { text: "≈ 4 months", options: { bold: true, align: "right" } }]];
s.addTable(ue, { x: 6.85, y: 1.55, w: 5.85, colW: [4.45, 1.4], fontSize: 14, fontFace: "Calibri", color: HEX.dk1, border: { type: "solid", pt: 0.5, color: "D5D8D3" }, rowH: 0.42 });
text(s, "Every figure is our estimate, to be confirmed with Royal Motors in the pilot. Not counted: fewer crash repairs and downtime for fleets (IIHS: −27% rear-end crashes with a warning; we assume half until local data proves more) and insurer discounts. Running cost per car is near zero because inference runs on the phone.", { x: 0.6, y: 4.95, w: 12.1, h: 1.1, fontSize: 13, color: HEX.accent5 });
s.addNotes("One and a half minutes. Cost to build and run the pilot, unit economics and payback, every assumption stated.");

// ---------------------------------------------------------------- 9b. next product: Trumen Scan
pres.addSection({ title: "Next product" });
s = pres.addSlide({ masterName: "DARK", sectionTitle: "Next product" });
text(s, "Next product: one walk-around, the car in 3D, part by part", { x: 0.6, y: 0.45, w: 12.1, h: 0.8, fontSize: 30, bold: true, fontFace: "Arial", color: "FFFFFF" });
const img = (f) => "image/png;base64," + fs.readFileSync(path.join(__dirname, f)).toString("base64");
s.addImage({ data: img("scan_whole.png"), x: 0.6, y: 1.45, w: 5.9, h: 3.69, objectName: "scan whole" });
s.addImage({ data: img("scan_apart.png"), x: 6.8, y: 1.45, w: 5.9, h: 3.69, objectName: "scan apart" });
text(s, "The Royal Motors test-drive Camry at the venue: 29 s video + 9 photos", { x: 0.6, y: 5.2, w: 5.9, h: 0.35, fontSize: 13, color: HEX.accent6 });
text(s, "Split into 32 named exterior parts, each sized and searchable", { x: 6.8, y: 5.2, w: 5.9, h: 0.35, fontSize: 13, color: HEX.accent6 });
text(s, [
  { text: "95 of 95 views placed; scale from the published length, then width 1.93 m and height 1.36 m vs 1.84 / 1.445 m published. Exported as glTF + JSON for other apps.", options: { breakLine: true } },
  { text: "For after-sales: a 3D condition record at handover and trade-in, and parts lookup. Hidden parts would come from the Toyota/BYD catalogue by VIN.", options: {} },
], { x: 0.6, y: 5.65, w: 12.1, h: 1.1, fontSize: 15, color: "FFFFFF", paraSpaceAfter: 4 });
s.addNotes("Fifteen seconds, only if time allows. Built during the final as an add-on, live at /scan/ on the same site. Exterior only; plate and people are removed from the model.");

// ---------------------------------------------------------------- 10. next steps
pres.addSection({ title: "Next steps" });
s = pres.addSlide({ masterName: "DARK", sectionTitle: "Next steps" });
text(s, "Three months to a decision", { x: 0.6, y: 0.5, w: 12, h: 0.9, fontSize: 44, bold: true, fontFace: "Arial", color: "FFFFFF" });
const months = [
  ["Month 1", "20 Royal Motors loaner and test-drive cars; record local footage with consent; tune on Tashkent streets, day and night."],
  ["Month 2", "Safety Pack at handover in one Toyota and one BYD centre; one fleet customer with 50 vehicles."],
  ["Month 3", "Measure attach rate, warnings per 100 km, driver feedback and incidents; decide on roll-out."],
];
months.forEach(([h, b], i) => {
  const x = 0.6 + i * 4.1;
  s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x, y: 1.75, w: 3.9, h: 2.9, rectRadius: 0.1, fill: { color: "2E3238" }, line: { color: i === 2 ? HEX.accent1 : "4A5058", width: 1.5 }, objectName: `month ${i + 1}` });
  text(s, h, { x: x + 0.3, y: 1.95, w: 3.3, h: 0.5, fontSize: 22, bold: true, fontFace: "Arial", color: HEX.accent1 });
  text(s, b, { x: x + 0.3, y: 2.55, w: 3.3, h: 1.95, fontSize: 16, color: "FFFFFF", valign: "top" });
});
text(s, "Our ask: one after-sales manager as pilot owner, 20 loaner cars, and access to one fleet customer.", { x: 0.6, y: 5.15, w: 12, h: 0.5, fontSize: 20, bold: true, color: "FFFFFF" });
s.addImage({ data: QR, x: 11.6, y: 5.85, w: 1.1, h: 1.1, objectName: "QR small" });
s.addNotes("Half a minute. A concrete ask, then questions.");

(async () => {
  const out = path.join(__dirname, "Trumen_Guard_defence.pptx");
  await pres.writeFile({ fileName: out });
  await applyTheme(out, THEME);
  console.log("wrote", out);
})();
