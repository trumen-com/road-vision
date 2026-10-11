import { analyzeJoint, judgePanels, labToRgb, DEFAULTS } from "./engine.js";
import { PANELS, JOINTS, SHAPES, GLASS, JOINT_XY, jointFactor } from "./car.js";

const $ = (s, el = document) => el.querySelector(s);
const $$ = (s, el = document) => [...el.querySelectorAll(s)];
const NS = "http://www.w3.org/2000/svg";

// ---------------------------------------------------------------- state
const blankCar = (example = false) => ({
  id: "car-" + new Date().toISOString().slice(0, 16).replace(/[-:T]/g, "") + "-" + Math.random().toString(36).slice(2, 5),
  make: "", model: "", colour: "", note: "",
  example,
  gauge: {},      // panelId -> µm
  shots: {},      // jointId -> {img (ImageData), url, result, gapX, saved}
});
let car = blankCar();
let mode = "inspect";      // "inspect" | "collect"
let current = null;        // joint being reviewed
let cloud = null;          // {db, assets} when running inside Claude with storage granted

// ---------------------------------------------------------------- storage
// Inside Claude: photos go to the artifact's file store and labels to its database, so the team's data reaches us directly.
// Elsewhere (GitHub Pages): photos stay on the phone (IndexedDB) and are exported as one .zip.
async function initCloud() {
  if (!window.claude?.use) return;
  try {
    const [db, assets] = await Promise.all([window.claude.use("db"), window.claude.use("assets")]);
    if (db && assets) { cloud = { db, assets }; updateStoreNote(); refreshCollected(); }
  } catch { /* stay local */ }
}

const idb = (() => {
  let p;
  const open = () => p || (p = new Promise((res, rej) => {
    const r = indexedDB.open("paintcheck", 1);
    r.onupgradeneeded = () => { r.result.createObjectStore("shots", { keyPath: "key" }); r.result.createObjectStore("cars", { keyPath: "id" }); };
    r.onsuccess = () => res(r.result); r.onerror = () => rej(r.error);
  }));
  const tx = async (store, fn, mode = "readonly") => {
    const db = await open();
    return new Promise((res, rej) => { const t = db.transaction(store, mode); const out = fn(t.objectStore(store)); t.oncomplete = () => res(out?.result ?? out); t.onerror = () => rej(t.error); });
  };
  return {
    put: (store, v) => tx(store, (s) => s.put(v), "readwrite"),
    all: (store) => tx(store, (s) => s.getAll()),
    clear: async () => { await tx("shots", (s) => s.clear(), "readwrite"); await tx("cars", (s) => s.clear(), "readwrite"); },
  };
})();

function carRecord() {
  return { id: car.id, make: car.make, model: car.model, colour: car.colour, note: car.note, gauge: car.gauge, updatedAt: new Date().toISOString() };
}
async function saveCar() {
  if (mode !== "collect" || car.example) return;
  try {
    if (cloud) await cloud.db.doc("cars/" + car.id).set(carRecord());
    else await idb.put("cars", carRecord());
  } catch (e) { toast("Could not save the car details: " + (e.code || e.message)); }
}
async function saveShot(jid) {
  const s = car.shots[jid];
  if (mode !== "collect" || car.example || !s) return;
  const j = JOINTS.find((x) => x.id === jid);
  const blob = await s.jpeg;
  const row = { car: car.id, joint: jid, a: j.a, b: j.b, gapX: s.gapX ?? null, orientation: s.orientation, score: s.result.score, dE: s.result.dE, warnings: s.result.warnings, takenAt: new Date().toISOString() };
  try {
    if (cloud) {
      const up = await cloud.assets.upload(blob, { type: "image/jpeg" });
      await cloud.db.doc(`shots/${car.id}_${jid}`).set({ ...row, photo: up.id });
    } else {
      await idb.put("shots", { key: `${car.id}_${jid}`, ...row, blob });
    }
    s.saved = true;
    await saveCar();
    refreshCollected();
    toast("Saved " + j.id.toUpperCase());
  } catch (e) {
    toast(e.code === "quota_exceeded" ? "Storage is full. Export what you have first." : "Could not save the photo: " + (e.code || e.message));
  }
  renderJoints();
}
async function refreshCollected() {
  let cars = 0, shots = 0;
  try {
    if (cloud) { cars = (await cloud.db.collection("cars").get()).size; shots = (await cloud.db.collection("shots").get()).size; }
    else { cars = (await idb.all("cars")).length; shots = (await idb.all("shots")).length; }
  } catch { /* ignore */ }
  $("#collected").textContent = `${cars} car${cars === 1 ? "" : "s"}, ${shots} photo${shots === 1 ? "" : "s"} collected`;
  $("#export").hidden = !!cloud || shots === 0;
}
function updateStoreNote() {
  $("#storeNote").textContent = cloud
    ? "Photos and gauge readings are uploaded to the team's private store as you go."
    : "Photos and gauge readings stay on this phone. Tap Export when you finish and send the .zip to the team.";
}
async function exportZip() {
  if (!window.JSZip) return toast("The export library did not load. Check the connection and reload.");
  const zip = new JSZip();
  const cars = await idb.all("cars"), shots = await idb.all("shots");
  for (const s of shots) zip.file(`photos/${s.key}.jpg`, s.blob);
  zip.file("labels.json", JSON.stringify({ format: "trumen-paintcheck-1", cars, shots: shots.map(({ blob, ...r }) => ({ ...r, photo: `photos/${r.key}.jpg` })) }, null, 1));
  const blob = await zip.generateAsync({ type: "blob" });
  const a = document.createElement("a");
  a.href = URL.createObjectURL(blob); a.download = `paintcheck-${new Date().toISOString().slice(0, 10)}.zip`; a.click();
}

// ---------------------------------------------------------------- photos
async function loadPhoto(file) {
  const bmp = await createImageBitmap(file, { imageOrientation: "from-image" });
  const s = Math.min(1, 1600 / Math.max(bmp.width, bmp.height));
  const c = document.createElement("canvas");
  c.width = Math.round(bmp.width * s); c.height = Math.round(bmp.height * s);
  const g = c.getContext("2d", { willReadFrequently: true });
  g.drawImage(bmp, 0, 0, c.width, c.height);
  return { canvas: c, img: g.getImageData(0, 0, c.width, c.height), jpeg: new Promise((r) => c.toBlob(r, "image/jpeg", 0.95)) };
}

function analyse(jid) {
  const s = car.shots[jid], j = JOINTS.find((x) => x.id === jid);
  s.result = analyzeJoint(s.img, { orientation: s.orientation, gapX: s.gapX, factor: jointFactor(j) });
}

// ---------------------------------------------------------------- diagram
function statusClass(st) { return { original: "ok", check: "check", suspect: "suspect", repainted: "bad" }[st] || "none"; }
function renderDiagram(panelStatus) {
  const svg = $("#diagram");
  svg.replaceChildren();
  const el = (tag, attrs) => { const e = document.createElementNS(NS, tag); for (const k in attrs) e.setAttribute(k, attrs[k]); return e; };
  for (const g of GLASS) svg.append(el("path", { d: g, class: "glass" }));
  for (const [pid, d] of Object.entries(SHAPES)) {
    const st = panelStatus[pid]?.status;
    const p = el("path", { d, class: "panel " + statusClass(st) });
    const t = el("title", {}); t.textContent = PANELS[pid].name + (st ? ": " + st : ": not measured") + (car.gauge[pid] ? `, gauge ${car.gauge[pid]} µm` : "");
    p.append(t); svg.append(p);
  }
  for (const j of JOINTS) {
    const [x, y] = JOINT_XY[j.id], s = car.shots[j.id];
    const v = s?.result ? (s.result.reliable ? s.result.verdict : "retake") : "todo";
    const g = el("g", { class: "jmark " + v, tabindex: 0, role: "button", "aria-label": `Joint ${j.id.slice(1)}: ${j.where}` });
    g.append(el("circle", { cx: x, cy: y, r: 8.5 }));
    const tx = el("text", { x, y: y + 3.2, "text-anchor": "middle" }); tx.textContent = j.id.slice(1);
    g.append(tx);
    g.addEventListener("click", () => openJoint(j.id));
    g.addEventListener("keydown", (e) => { if (e.key === "Enter" || e.key === " ") { e.preventDefault(); openJoint(j.id); } });
    svg.append(g);
  }
  const lab = (x, y, t) => { const e = el("text", { x, y, class: "dlabel", "text-anchor": "middle" }); e.textContent = t; svg.append(e); };
  lab(110, -4, "FRONT"); lab(8, 220, "L"); lab(212, 220, "R");
}

// ---------------------------------------------------------------- joints list + report
const VERDICT_TEXT = { match: "Same paint", check: "Small difference", differs: "Different paint", retake: "Retake photo", todo: "Not taken" };
const WARN_TEXT = {
  gap_not_found: "No clear gap found. Put the gap in the middle of the photo, or drag the line onto it.",
  gap_near_edge: "The gap is near the edge, so one side has little paint to measure.",
  too_few_pixels: "Not enough clean paint on one side.",
  mixed_surface: "A sample box covers more than paint (trim, a badge, dirt or a sticker). Move closer or drag the line.",
  glare: "Strong glare. Step to the side so the sun or a lamp is not reflected on the gap.",
  too_dark: "Too dark to read the colour.",
  blurry: "The photo is blurry. Hold still and tap to focus on the gap.",
};
function panelStatuses() {
  return judgePanels(JOINTS.filter((j) => car.shots[j.id]?.result).map((j) => ({ ...j, factor: jointFactor(j), result: car.shots[j.id].result })));
}
function renderJoints() {
  const st = panelStatuses();
  renderDiagram(st);
  const list = $("#joints");
  list.replaceChildren();
  for (const j of JOINTS) {
    const s = car.shots[j.id];
    const v = s?.result ? (s.result.reliable ? s.result.verdict : "retake") : "todo";
    const li = document.createElement("li");
    li.className = "joint " + v;
    li.innerHTML = `<button class="jrow" type="button"><span class="jn">${j.id.slice(1)}</span><span class="jw"><b></b><small></small></span><span class="jv"></span></button>`;
    $("b", li).textContent = `${PANELS[j.a].name} / ${PANELS[j.b].name}`;
    $("small", li).textContent = j.where;
    $(".jv", li).textContent = VERDICT_TEXT[v] + (s?.result ? ` · ${(s.result.score / jointFactor(j)).toFixed(1)}` : "") + (s?.saved ? " · saved" : "");
    $("button", li).addEventListener("click", () => openJoint(j.id));
    list.append(li);
  }
  renderReport(st);
  renderGauge(st);
}
function renderReport(st) {
  const done = JOINTS.filter((j) => car.shots[j.id]?.result).length;
  const bad = Object.entries(st).filter(([, v]) => v.status === "repainted").map(([k]) => k);
  const sus = Object.entries(st).filter(([, v]) => v.status === "suspect").map(([k]) => k);
  const chk = Object.entries(st).filter(([, v]) => v.status === "check").map(([k]) => k);
  const retake = JOINTS.filter((j) => car.shots[j.id]?.result && !car.shots[j.id].result.reliable).length;
  const head = $("#verdict");
  head.className = "verdict " + (bad.length ? "bad" : sus.length ? "suspect" : done ? "ok" : "none");
  head.querySelector("b").textContent = !done ? "No joints photographed yet"
    : bad.length ? `${bad.length} panel${bad.length > 1 ? "s" : ""} likely repainted`
    : sus.length ? "Paint differs at some joints" : "No repaint found at the joints checked";
  head.querySelector("span").textContent = `${done} of ${JOINTS.length} joints checked` + (retake ? `, ${retake} need a retake` : "");
  const names = (arr) => arr.map((p) => PANELS[p].name).join(", ") || "none";
  $("#rBad").textContent = names(bad);
  $("#rSus").textContent = names(sus);
  $("#rChk").textContent = names(chk);
}
function reportText() {
  const st = panelStatuses();
  const lines = [`Paint check: ${[car.make, car.model, car.colour].filter(Boolean).join(" ") || "car"} (${new Date().toLocaleDateString()})`];
  lines.push($("#verdict b").textContent + ". " + $("#verdict span").textContent + ".");
  for (const [k, v] of Object.entries(st)) if (v.status !== "original") lines.push(`- ${PANELS[k].name}: ${v.status}`);
  lines.push("Joints:");
  for (const j of JOINTS) {
    const s = car.shots[j.id]; if (!s?.result) continue;
    lines.push(`  ${j.id.slice(1)}. ${PANELS[j.a].name} / ${PANELS[j.b].name}: ${VERDICT_TEXT[s.result.reliable ? s.result.verdict : "retake"]} (score ${(s.result.score / jointFactor(j)).toFixed(2)}, colour ΔE ${s.result.dE})`);
  }
  lines.push("Screening only: confirm flagged panels with a paint thickness gauge before pricing.");
  return lines.join("\n");
}
function renderGauge(st) {
  const box = $("#gauge");
  if (mode !== "collect") { box.hidden = true; return; }
  box.hidden = false;
  const grid = $("#gaugeGrid");
  if (!grid.children.length) {
    for (const [pid, p] of Object.entries(PANELS)) {
      const id = "g_" + pid;
      const wrap = document.createElement("label");
      wrap.className = "gcell"; wrap.htmlFor = id;
      wrap.innerHTML = `<span></span><input id="${id}" inputmode="numeric" pattern="[0-9]*" maxlength="4" placeholder="µm">`;
      $("span", wrap).textContent = p.name;
      const inp = $("input", wrap);
      inp.addEventListener("change", () => {
        const v = parseInt(inp.value, 10);
        if (Number.isFinite(v) && v > 0 && v < 5000) car.gauge[pid] = v; else delete car.gauge[pid];
        inp.classList.toggle("thick", (car.gauge[pid] || 0) > 200);
        saveCar(); renderDiagram(panelStatuses());
      });
      grid.append(wrap);
    }
  }
  for (const pid of Object.keys(PANELS)) { const i = $("#g_" + pid); if (document.activeElement !== i) i.value = car.gauge[pid] ?? ""; i.classList.toggle("thick", (car.gauge[pid] || 0) > 200); }
}

// ---------------------------------------------------------------- review sheet
function openJoint(jid) {
  current = jid;
  const j = JOINTS.find((x) => x.id === jid);
  $("#rvTitle").textContent = `Joint ${jid.slice(1)} · ${PANELS[j.a].name} / ${PANELS[j.b].name}`;
  $("#rvWhere").textContent = j.where;
  $("#sheet").hidden = false;
  document.body.classList.add("locked");
  drawReview();
  $("#sheet .close").focus();
}
function closeSheet() { $("#sheet").hidden = true; document.body.classList.remove("locked"); current = null; renderJoints(); }

function drawReview() {
  const s = car.shots[current];
  const has = !!s?.result;
  $("#rvEmpty").hidden = has; $("#rvBody").hidden = !has;
  $("#rvSave").hidden = !(has && mode === "collect" && !car.example);
  if (!has) return;
  const cv = $("#rvCanvas"), r = s.result, src = s.canvas;
  cv.width = src.width; cv.height = src.height;
  const g = cv.getContext("2d");
  g.drawImage(src, 0, 0);
  const W = cv.width, H = cv.height, vert = r.boxes.orientation !== "horizontal";
  const span = r.boxes.span;
  const rect = (x0, x1, col) => {
    g.lineWidth = Math.max(3, W / 300); g.strokeStyle = col; g.setLineDash([]);
    if (vert) g.strokeRect(x0 * W, span[0] * H, (x1 - x0) * W, (span[1] - span[0]) * H);
    else g.strokeRect(span[0] * W, x0 * H, (span[1] - span[0]) * W, (x1 - x0) * H);
  };
  rect(r.boxes.a[0], r.boxes.a[1], "#4FA3FF");
  rect(r.boxes.b[0], r.boxes.b[1], "#FFB020");
  g.setLineDash([W / 60, W / 90]); g.lineWidth = Math.max(3, W / 260); g.strokeStyle = r.gap.found ? "#FFFFFF" : "#FF5A4F";
  g.beginPath();
  if (vert) { g.moveTo(r.gap.x * W, 0); g.lineTo(r.gap.x * W, H); } else { g.moveTo(0, r.gap.x * H); g.lineTo(W, r.gap.x * H); }
  g.stroke();

  const v = r.reliable ? r.verdict : "retake";
  const j = JOINTS.find((x) => x.id === current);
  $("#rvVerdict").className = "rv-verdict " + v;
  $("#rvVerdict b").textContent = VERDICT_TEXT[v];
  $("#rvVerdict span").textContent = `score ${(r.score / jointFactor(j)).toFixed(2)} · same paint below ${DEFAULTS.checkAt}, different above ${DEFAULTS.flagAt}` + (jointFactor(j) > 1.5 ? " (hood or trunk edge: these panels face the light differently, so more difference is allowed)" : jointFactor(j) > 1 ? " (bumper joint: plastic sits at an angle to the body, so more difference is allowed)" : "");
  const [ca, cb] = r.sides.map((x) => `rgb(${labToRgb(x.lab).join(",")})`);
  $("#swA").style.background = ca; $("#swB").style.background = cb;
  $("#mColour").textContent = r.dE.toFixed(2);
  $("#mTexture").textContent = r.dTexture.toFixed(2);
  $("#mGloss").textContent = r.glossCompared ? r.dGloss.toFixed(2) : "no reflection on both sides";
  $("#mLight").textContent = r.dL.toFixed(1);
  const w = $("#rvWarn"); w.replaceChildren();
  for (const k of r.warnings) { const li = document.createElement("li"); li.textContent = WARN_TEXT[k] || k; w.append(li); }
  w.hidden = !r.warnings.length;
  $("#rvOrient").textContent = vert ? "Gap runs across the photo instead" : "Gap runs up and down instead";
  $("#rvAuto").hidden = s.gapX == null;
}

function canvasPointerToGap(e) {
  const s = car.shots[current]; if (!s?.result) return;
  const rc = $("#rvCanvas").getBoundingClientRect();
  const vert = s.orientation !== "horizontal";
  s.gapX = Math.min(0.97, Math.max(0.03, vert ? (e.clientX - rc.left) / rc.width : (e.clientY - rc.top) / rc.height));
  analyse(current); drawReview();
  if (s.saved) { s.saved = false; }
}

// ---------------------------------------------------------------- example car (simulated photos)
function simulatedJoint(c1, c2, { seed = 1, gapAt = 0.5, noise = [3, 3], W = 450, H = 600 } = {}) {
  let st = seed; const rnd = () => ((st = (st * 16807) % 2147483647) / 2147483647);
  const n = () => { let u = 0; for (let i = 0; i < 6; i++) u += rnd(); return u - 3; };
  const c = document.createElement("canvas"); c.width = W; c.height = H;
  const g = c.getContext("2d"); const im = g.createImageData(W, H); const d = im.data; const gx = Math.round(W * gapAt);
  const skyX = 0.2 + 0.6 * rnd();
  for (let y = 0; y < H; y++) for (let x = 0; x < W; x++) {
    const left = x < gx, col = left ? c1 : c2, nz = left ? noise[0] : noise[1];
    let k = 0.8 + 0.3 * (y / H) - 0.12 * (x / W);
    if (Math.abs(x - gx) < 4) k *= 0.22;
    const sky = Math.exp(-(((x - W * skyX) / 50) ** 2 + ((y - H * 0.35) / 260) ** 2));
    const i = (y * W + x) * 4;
    for (let ch = 0; ch < 3; ch++) d[i + ch] = Math.min(255, col[ch] * k + 190 * sky + nz * n());
    d[i + 3] = 255;
  }
  g.putImageData(im, 0, 0);
  return { canvas: c, img: im, jpeg: Promise.resolve(null) };
}
function loadExample() {
  car = blankCar(true);
  Object.assign(car, { make: "Chevrolet", model: "Cobalt", colour: "Silver" });
  const silver = [166, 169, 172], repaint = [161, 168, 177];
  const repainted = new Set(["lf_door"]);
  JOINTS.forEach((j, i) => {
    const a = repainted.has(j.a) ? repaint : silver, b = repainted.has(j.b) ? repaint : silver;
    const ph = simulatedJoint(a, b, { seed: 7 + i * 13, gapAt: 0.42 + 0.16 * ((i * 37) % 10) / 10, noise: [3, repainted.has(j.b) ? 6 : 3] });
    car.shots[j.id] = { ...ph, orientation: "vertical", gapX: null };
    analyse(j.id);
  });
  syncCarFields();
  $("#exampleBanner").hidden = false;
  renderJoints();
}
function newCar() {
  car = blankCar(false);
  syncCarFields();
  $("#exampleBanner").hidden = true;
  $("#gaugeGrid").replaceChildren();
  renderJoints();
}
function syncCarFields() {
  for (const k of ["make", "model", "colour"]) $("#c_" + k).value = car[k];
}

// ---------------------------------------------------------------- ui
let toastT;
function toast(msg) { const t = $("#toast"); t.textContent = msg; t.hidden = false; clearTimeout(toastT); toastT = setTimeout(() => (t.hidden = true), 3200); }

function setMode(m) {
  mode = m;
  $$(".tab").forEach((b) => b.setAttribute("aria-selected", String(b.dataset.mode === m)));
  document.body.dataset.mode = m;
  if (m === "collect" && car.example) newCar();
  renderJoints();
}

function bind() {
  $$(".tab").forEach((b) => b.addEventListener("click", () => setMode(b.dataset.mode)));
  for (const k of ["make", "model", "colour"]) $("#c_" + k).addEventListener("change", (e) => { car[k] = e.target.value.trim(); saveCar(); });
  $("#newCar").addEventListener("click", newCar);
  $("#loadExample").addEventListener("click", () => { if (mode === "collect") setMode("inspect"); loadExample(); });
  $("#sheet .close").addEventListener("click", closeSheet);
  $("#sheet").addEventListener("click", (e) => { if (e.target.id === "sheet") closeSheet(); });
  document.addEventListener("keydown", (e) => { if (e.key === "Escape" && !$("#sheet").hidden) closeSheet(); });
  $("#rvFile").addEventListener("change", async (e) => {
    const f = e.target.files[0]; e.target.value = "";
    if (!f || !current) return;
    try {
      if (car.example) { car = blankCar(false); syncCarFields(); $("#exampleBanner").hidden = true; }
      const ph = await loadPhoto(f);
      car.shots[current] = { ...ph, orientation: car.shots[current]?.orientation || "vertical", gapX: null, saved: false };
      analyse(current); drawReview();
      if (mode === "collect") saveShot(current);
    } catch (err) { toast("Could not read that photo. Try another one."); }
  });
  $("#rvCanvas").addEventListener("click", canvasPointerToGap);
  $("#rvOrient").addEventListener("click", () => { const s = car.shots[current]; if (!s) return; s.orientation = s.orientation === "horizontal" ? "vertical" : "horizontal"; s.gapX = null; analyse(current); drawReview(); });
  $("#rvAuto").addEventListener("click", () => { const s = car.shots[current]; if (!s) return; s.gapX = null; analyse(current); drawReview(); });
  $("#rvSave").addEventListener("click", () => saveShot(current));
  $("#rvNext").addEventListener("click", () => { const i = JOINTS.findIndex((j) => j.id === current); openJoint(JOINTS[(i + 1) % JOINTS.length].id); });
  $("#copyReport").addEventListener("click", async () => {
    const t = reportText();
    try { await navigator.clipboard.writeText(t); toast("Report copied"); }
    catch { const pre = $("#reportText"); pre.hidden = false; pre.textContent = t; const r = document.createRange(); r.selectNodeContents(pre); getSelection().removeAllRanges(); getSelection().addRange(r); toast("Select the text below and copy it"); }
  });
  $("#export").addEventListener("click", exportZip);
}

bind();
updateStoreNote();
loadExample();
setMode("inspect");
refreshCollected();
initCloud();
