import "./style.css";
import { CLASS_NAME, GuardState, IN_H, IN_W } from "../../engine/types";
import { corridorAt } from "../../engine/threat";
import { cropRect, createRuntime, Runtime } from "./runtime";
import { renderPages } from "./pages";

const BASE = import.meta.env.BASE_URL;
const $ = <T extends HTMLElement>(id: string) => document.getElementById(id) as T;

// ---------------------------------------------------------------- routing
function route() {
  const page = (location.hash.replace("#", "") || "drive").split("/")[0];
  document.querySelectorAll<HTMLElement>("[data-page]").forEach((el) => (el.hidden = el.dataset.page !== page));
  document.querySelectorAll<HTMLAnchorElement>("[data-nav]").forEach((a) => {
    if (a.dataset.nav === page) a.setAttribute("aria-current", "page"); else a.removeAttribute("aria-current");
  });
  if (page !== "drive") window.scrollTo(0, 0);
}
window.addEventListener("hashchange", route);
route();
renderPages(BASE);

// ---------------------------------------------------------------- state
const video = $<HTMLVideoElement>("video");
const overlay = $<HTMLCanvasElement>("overlay");
const octx = overlay.getContext("2d")!;
const band = $("band"), bandWord = $("bandWord"), bandDetail = $("bandDetail");
let rt: Runtime | null = null;
let running = false, busy = false, source: "sample" | "camera" | "file" | null = null;
let last: GuardState | null = null;
let lastFrameT = 0, fpsEma = 0, calibrating = false;
let gpsSpeed: number | null = null, geoWatch: number | null = null;
const history: { t: number; r: number; l: number }[] = [];
const events: { at: string; t: number; level: number; what: string; ttc: number }[] = [];
let lastLevel = 0, clock0 = 0;

let pendingCorridor: { vpX: number; vpY: number; bottomHalfWidth: number } | undefined;
try { const c = JSON.parse(localStorage.getItem("guard.corridor") || "null"); if (c) pendingCorridor = c; } catch { /* storage unavailable */ }

// ---------------------------------------------------------------- samples
interface SampleClip { file: string; title: string; note: string }
async function loadSamples() {
  const box = $("samples");
  try {
    const list = (await (await fetch(`${BASE}samples/samples.json`)).json()) as SampleClip[];
    box.innerHTML = "";
    for (const s of list) {
      const b = document.createElement("button");
      b.className = "sample";
      b.innerHTML = `${s.title}<small>${s.note}</small>`;
      const mp4 = video.canPlayType('video/mp4; codecs="avc1.42E01E"') !== "";
      const file = mp4 ? s.file : s.file.replace(/\.mp4$/, ".webm");
      b.onclick = () => start("sample", `${BASE}samples/${file}`);
      box.appendChild(b);
    }
  } catch { box.textContent = "Sample clips are not available. Use your camera or open a video."; }
}
loadSamples();

// ---------------------------------------------------------------- runtime
async function ensureRuntime(): Promise<Runtime> {
  if (rt) return rt;
  const L = $("loading"); L.hidden = false;
  rt = await createRuntime(BASE, (s) => (L.textContent = s));
  if (pendingCorridor) Object.assign(rt.guard.corridor, pendingCorridor);
  $("backend").textContent = rt.backend === "webgpu" ? "GPU" : "CPU";
  $("modelInfo").textContent = rt.modelInfo;
  L.hidden = true;
  return rt;
}

async function start(kind: "sample" | "camera" | "file", src?: string | File) {
  stop();
  $("empty").hidden = true;
  setBand(0, "Starting…", "Loading the detector");
  try {
    const r = await ensureRuntime();
    r.guard.reset();
    r.visual?.reset();
    history.length = 0; lastLevel = 0; clock0 = performance.now();
    if (kind === "camera") {
      const stream = await navigator.mediaDevices.getUserMedia({ video: { facingMode: { ideal: "environment" }, width: { ideal: 1280 }, height: { ideal: 720 } }, audio: false });
      video.srcObject = stream;
      startGps();
    } else {
      video.srcObject = null;
      video.src = typeof src === "string" ? src : URL.createObjectURL(src as File);
      video.loop = false;
    }
    source = kind;
    await video.play();
    running = true;
    $("stage").classList.add("live");
    $("btnStop").hidden = false;
    setBand(0, "Watching the road", kind === "camera" ? "Warnings start above 8 km/h when GPS is available" : "Analysing the video in real time");
    loop();
  } catch (e) {
    $("empty").hidden = false;
    setBand(0, "Could not start", explain(e));
  }
}

function explain(e: unknown): string {
  const n = (e as { name?: string })?.name;
  if (n === "NotAllowedError") return "Camera access was blocked. Allow the camera for this site in the browser settings, then try again.";
  if (n === "NotFoundError") return "No camera was found on this device. Open a video instead.";
  if (n === "NotSupportedError") return "This video format cannot be played here. Try an MP4 (H.264) file.";
  return `Something stopped the start-up: ${String((e as Error)?.message ?? e)}`;
}

function stop() {
  running = false;
  $("stage").classList.remove("live");
  video.pause();
  const s = video.srcObject as MediaStream | null;
  s?.getTracks().forEach((t) => t.stop());
  video.srcObject = null;
  if (geoWatch !== null) { navigator.geolocation.clearWatch(geoWatch); geoWatch = null; gpsSpeed = null; }
  $("btnStop").hidden = true;
  source = null;
}

function startGps() {
  if (!("geolocation" in navigator)) return;
  geoWatch = navigator.geolocation.watchPosition((p) => {
    gpsSpeed = p.coords.speed != null ? p.coords.speed * 3.6 : null;
    $("speed").textContent = gpsSpeed != null ? gpsSpeed.toFixed(0) : "–";
  }, () => { /* no GPS: warnings are not speed-gated */ }, { enableHighAccuracy: true, maximumAge: 1000 });
}

// ---------------------------------------------------------------- main loop
function loop() {
  if (!running || !rt) return;
  const r = rt;
  if (video.ended) { finish(); return; }
  if (!busy && video.readyState >= 2) {
    busy = true;
    const c = cropRect(video.videoWidth, video.videoHeight);
    r.ctx.drawImage(video, c.sx, c.sy, c.sw, c.sh, 0, 0, IN_W, IN_H);
    const px = r.ctx.getImageData(0, 0, IN_W, IN_H).data;
    const t = source === "camera" ? (performance.now() - clock0) / 1000 : video.currentTime;
    let vpx: Uint8ClampedArray | null = null;
    if (r.visual) { r.vctx.drawImage(video, c.sx, c.sy, c.sw, c.sh, 0, 0, r.vframe.width, r.vframe.height); vpx = r.vctx.getImageData(0, 0, r.vframe.width, r.vframe.height).data; }
    const vis = r.visual && vpx ? r.visual.step(vpx, 4, t).then((v) => { if (v != null) r.guard.visualRisk = v; }) : Promise.resolve();
    vis.then(() => r.guard.step(px, 4, t)).then((st) => {
      // speed gate: stationary or crawling (queues) => no audible warnings
      if (source === "camera" && gpsSpeed != null && gpsSpeed < 8) st.level = 0;
      last = st;
      const now = performance.now();
      if (lastFrameT) fpsEma = fpsEma ? fpsEma * 0.85 + 0.15 * (1000 / (now - lastFrameT)) : 1000 / (now - lastFrameT);
      lastFrameT = now;
      onState(st);
      busy = false;
    }).catch(() => { busy = false; });
  }
  draw();
  requestAnimationFrame(loop);
}

function finish() {
  running = false;
  setBand(lastLevel as 0 | 1 | 2, lastLevel ? bandWord.textContent || "" : "Clip finished", "Replay it, try another sample or use your camera");
  $("btnStop").hidden = true;
  $("empty").hidden = false;
}

// ---------------------------------------------------------------- alerts
let audio: AudioContext | null = null;
function beep(level: number) {
  if (!(document.getElementById("sound") as HTMLInputElement).checked) return;
  try {
    audio ??= new AudioContext();
    const n = level === 2 ? 3 : 1;
    for (let i = 0; i < n; i++) {
      const o = audio.createOscillator(), g = audio.createGain();
      o.frequency.value = level === 2 ? 1250 : 880;
      o.type = "square";
      const t0 = audio.currentTime + i * 0.16;
      g.gain.setValueAtTime(0.0001, t0); g.gain.exponentialRampToValueAtTime(0.35, t0 + 0.01); g.gain.exponentialRampToValueAtTime(0.0001, t0 + 0.12);
      o.connect(g).connect(audio.destination); o.start(t0); o.stop(t0 + 0.13);
    }
  } catch { /* audio blocked until the first tap */ }
  navigator.vibrate?.(level === 2 ? [120, 60, 120, 60, 120] : [80]);
}

function describe(st: GuardState): string {
  const top = st.top;
  if (!top) return "";
  const what = CLASS_NAME[top.cls] ?? "object";
  const where = top.inPathNow ? "ahead in your lane" : top.inPathSoon ? "moving into your path" : "nearby";
  const ttc = Number.isFinite(top.ttc) ? `, contact in ${top.ttc.toFixed(1)} s` : "";
  return `${what[0].toUpperCase()}${what.slice(1)} ${where}${ttc}`;
}

function setBand(level: 0 | 1 | 2, word: string, detail: string) {
  band.dataset.level = String(level);
  bandWord.textContent = word;
  bandDetail.textContent = detail;
}

function onState(st: GuardState) {
  history.push({ t: st.t, r: st.risk, l: st.level });
  while (history.length && st.t - history[0].t > 20) history.shift();
  if (history.length && st.t < history[history.length - 1].t - 1) history.length = 0;
  $("risk").textContent = st.risk.toFixed(2);
  $("ttc").textContent = st.top && Number.isFinite(st.top.ttc) && (st.top.inPathNow || st.top.inPathSoon) ? st.top.ttc.toFixed(1) : "–";
  $("fps").textContent = fpsEma ? fpsEma.toFixed(0) : "–";
  if (st.level !== lastLevel) {
    if (st.level === 2) setBand(2, "Brake", describe(st));
    else if (st.level === 1) setBand(1, "Caution", describe(st));
    else setBand(0, "Watching the road", "No threat in your path");
    if (st.level > lastLevel) {
      beep(st.level);
      const at = source === "camera" ? new Date().toLocaleTimeString() : `${st.t.toFixed(1)} s into the video`;
      events.unshift({ at, t: st.t, level: st.level, what: describe(st), ttc: st.top?.ttc ?? Infinity });
      renderEvents();
    }
    lastLevel = st.level;
  } else if (st.level > 0) bandDetail.textContent = describe(st);
  drawSpark();
}

function renderEvents() {
  const ol = $("events");
  ol.innerHTML = events.slice(0, 50).map((e) => `<li class="ev-l${e.level}"><b>${e.level === 2 ? "Brake warning" : "Caution"}</b>, ${e.at}: ${e.what}</li>`).join("");
  const n2 = events.filter((e) => e.level === 2).length, n1 = events.length - n2;
  $("tripSummary").textContent = `${n2} brake warning${n2 === 1 ? "" : "s"} and ${n1} caution${n1 === 1 ? "" : "s"} this session. Nothing leaves this phone.`;
  $("btnExport").hidden = events.length === 0;
}

$("btnExport").onclick = () => {
  const rows = ["level,when,description,ttc_s", ...events.map((e) => `${e.level === 2 ? "brake" : "caution"},"${e.at}","${e.what}",${Number.isFinite(e.ttc) ? e.ttc.toFixed(2) : ""}`)];
  const a = document.createElement("a");
  a.href = URL.createObjectURL(new Blob([rows.join("\n")], { type: "text/csv" }));
  a.download = `trumen-guard-trip-${new Date().toISOString().slice(0, 16).replace(/[:T]/g, "-")}.csv`;
  a.click();
};

// ---------------------------------------------------------------- drawing
function viewMap() {
  // model space (IN_W x IN_H) -> overlay pixels, honouring the centre crop and object-fit: cover
  const W = overlay.clientWidth, H = overlay.clientHeight, dpr = window.devicePixelRatio || 1;
  if (overlay.width !== Math.round(W * dpr) || overlay.height !== Math.round(H * dpr)) { overlay.width = Math.round(W * dpr); overlay.height = Math.round(H * dpr); }
  const vw = video.videoWidth || 16, vh = video.videoHeight || 9;
  const sc = Math.max(W / vw, H / vh), ox = (W - vw * sc) / 2, oy = (H - vh * sc) / 2;
  const c = cropRect(vw, vh);
  const kx = c.sw / IN_W, ky = c.sh / IN_H;
  return { dpr, map: (x: number, y: number): [number, number] => [ox + (c.sx + x * kx) * sc, oy + (c.sy + y * ky) * sc] };
}

function draw() {
  const { dpr, map } = viewMap();
  octx.setTransform(dpr, 0, 0, dpr, 0, 0);
  octx.clearRect(0, 0, overlay.width, overlay.height);
  if (!rt) return;
  const cor = rt.guard.corridor;
  // ego corridor
  const yTop = cor.vpY * IN_H + (IN_H - cor.vpY * IN_H) * 0.12;
  const a = corridorAt(cor, yTop), b = corridorAt(cor, IN_H);
  const pts = [map(a.cx - a.hw, yTop), map(a.cx + a.hw, yTop), map(b.cx + b.hw, IN_H), map(b.cx - b.hw, IN_H)];
  octx.beginPath(); pts.forEach(([x, y], i) => (i ? octx.lineTo(x, y) : octx.moveTo(x, y))); octx.closePath();
  const lvl = last?.level ?? 0;
  octx.fillStyle = lvl === 2 ? "rgba(224,49,43,.22)" : lvl === 1 ? "rgba(242,194,48,.18)" : "rgba(255,255,255,.06)";
  octx.fill();
  octx.strokeStyle = lvl === 2 ? "#E0312B" : lvl === 1 ? "#F2C230" : "rgba(255,255,255,.35)";
  octx.lineWidth = 2; octx.stroke();
  if (calibrating) {
    const [vx, vy] = map(cor.vpX * IN_W, cor.vpY * IN_H);
    octx.fillStyle = "#F2C230"; octx.beginPath(); octx.arc(vx, vy, 12, 0, Math.PI * 2); octx.fill();
    octx.font = "600 15px 'Barlow Semi Condensed', sans-serif"; octx.fillText("Drag to where your lane meets the horizon", vx + 18, vy + 5);
  }
  if (!last || !(document.getElementById("showBoxes") as HTMLInputElement).checked) return;
  for (const th of last.threats) {
    const [x1, y1] = map(th.box[0], th.box[1]), [x2, y2] = map(th.box[2], th.box[3]);
    const hot = th.risk >= 0.5, path = th.inPathNow || th.inPathSoon;
    octx.strokeStyle = hot ? (th.risk >= 0.75 ? "#E0312B" : "#F2C230") : path ? "rgba(255,255,255,.85)" : "rgba(255,255,255,.35)";
    octx.lineWidth = hot ? 3 : 1.5;
    octx.strokeRect(x1, y1, x2 - x1, y2 - y1);
    if (path && Number.isFinite(th.ttc) && th.ttc < 6) {
      octx.font = "700 15px 'Barlow Semi Condensed', sans-serif";
      const label = `${th.ttc.toFixed(1)} s`;
      const w = octx.measureText(label).width + 8;
      octx.fillStyle = octx.strokeStyle as string; octx.fillRect(x1, y1 - 20, w, 20);
      octx.fillStyle = hot && th.risk < 0.75 ? "#23262B" : "#fff"; octx.fillText(label, x1 + 4, y1 - 5);
    }
  }
}

function drawSpark() {
  const c = $<HTMLCanvasElement>("spark"), dpr = window.devicePixelRatio || 1;
  const W = c.clientWidth, H = 90;
  c.width = W * dpr; c.height = H * dpr;
  const g = c.getContext("2d")!; g.setTransform(dpr, 0, 0, dpr, 0, 0);
  g.clearRect(0, 0, W, H);
  g.strokeStyle = "rgba(255,255,255,.12)"; g.lineWidth = 1;
  for (const v of [0.5, 0.75]) { g.beginPath(); g.moveTo(0, H - v * H); g.lineTo(W, H - v * H); g.stroke(); }
  if (history.length < 2) return;
  const t1 = history[history.length - 1].t, t0 = t1 - 20;
  g.beginPath();
  history.forEach((h, i) => { const x = ((h.t - t0) / 20) * W, y = H - h.r * (H - 4) - 2; i ? g.lineTo(x, y) : g.moveTo(x, y); });
  g.strokeStyle = "#F2C230"; g.lineWidth = 2; g.stroke();
}

// ---------------------------------------------------------------- calibration (drag the vanishing point)
$("btnCalib").onclick = async () => { await ensureRuntime(); calibrating = !calibrating; $("btnCalib").textContent = calibrating ? "Done" : "Adjust lane"; draw(); };
overlay.addEventListener("pointerdown", (e) => { if (calibrating) moveVp(e); });
overlay.addEventListener("pointermove", (e) => { if (calibrating && e.buttons) moveVp(e); });
function moveVp(e: PointerEvent) {
  if (!rt) return;
  const r = overlay.getBoundingClientRect();
  const W = r.width, H = r.height, vw = video.videoWidth || 16, vh = video.videoHeight || 9;
  const sc = Math.max(W / vw, H / vh), ox = (W - vw * sc) / 2, oy = (H - vh * sc) / 2;
  const c = cropRect(vw, vh);
  const x = ((e.clientX - r.left - ox) / sc - c.sx) / c.sw, y = ((e.clientY - r.top - oy) / sc - c.sy) / c.sh;
  rt.guard.corridor.vpX = Math.min(0.8, Math.max(0.2, x));
  rt.guard.corridor.vpY = Math.min(0.7, Math.max(0.2, y));
  try { localStorage.setItem("guard.corridor", JSON.stringify(rt.guard.corridor)); } catch { /* not persisted */ }
  draw();
}

// ---------------------------------------------------------------- controls
$("btnCamera").onclick = () => start("camera");
$<HTMLInputElement>("file").onchange = (e) => { const f = (e.target as HTMLInputElement).files?.[0]; if (f) start("file", f); };
$("btnSamples").onclick = () => { stop(); $("empty").hidden = false; setBand(0, "Watching the road", "Choose a sample clip, your camera or your own video"); };
$("btnStop").onclick = () => { stop(); setBand(0, "Stopped", "Choose a sample clip, your camera or your own video"); $("empty").hidden = false; };
window.addEventListener("resize", draw);
