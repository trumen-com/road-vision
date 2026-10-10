// Trumen Paint Check: paint-match engine.
// Compares the paint on the two sides of one panel gap, in one photo, under the same light.
// Works on an ImageData-like object {data: Uint8ClampedArray RGBA, width, height}; runs in the browser and in Node.

export const DEFAULTS = {
  maxSide: 720,          // analysis resolution (long side, px)
  searchBand: 0.22,      // look for the gap within ±22% of the guide line (photos from the phone camera are rarely centred)
  margin: 0.035,         // keep this far from the gap (fraction of width) to skip the gap shadow and edge curvature
  sampleWidth: 0.2,      // width of each sample box (fraction)
  sampleSpan: [0.18, 0.82], // along the gap
  diffuse: [0.15, 0.6],  // luminance quantiles taken as diffuse paint (below: shadow and dirt; above: reflections)
  // joint verdict thresholds on the combined score (tuned on collected data; see eval/)
  checkAt: 1.0,
  flagAt: 1.6,
  // feature weights in the combined score
  wTexture: 1.2,
  wGloss: 0.8,
  wLight: 0,             // brightness difference: off by default (mostly panel angle); weight set from collected data
};

// ---------------------------------------------------------------- colour
const LIN = new Float32Array(256);
for (let i = 0; i < 256; i++) {
  const c = i / 255;
  LIN[i] = c <= 0.04045 ? c / 12.92 : Math.pow((c + 0.055) / 1.055, 2.4);
}
const fLab = (t) => (t > 216 / 24389 ? Math.cbrt(t) : (24389 / 27 * t + 16) / 116);

export function rgbToLab(r, g, b) {
  const R = LIN[r], G = LIN[g], B = LIN[b];
  const X = (0.4124564 * R + 0.3575761 * G + 0.1804375 * B) / 0.95047;
  const Y = 0.2126729 * R + 0.7151522 * G + 0.072175 * B;
  const Z = (0.0193339 * R + 0.119192 * G + 0.9503041 * B) / 1.08883;
  const fx = fLab(X), fy = fLab(Y), fz = fLab(Z);
  return [116 * fy - 16, 500 * (fx - fy), 200 * (fy - fz)];
}

// CIEDE2000 (Sharma, Wu, Dalal 2005) with a lightness weight kL.
export function deltaE2000(l1, l2, kL = 1, kC = 1, kH = 1) {
  const [L1, a1, b1] = l1, [L2, a2, b2] = l2;
  const deg = Math.PI / 180;
  const C1 = Math.hypot(a1, b1), C2 = Math.hypot(a2, b2);
  const Cm = (C1 + C2) / 2, Cm7 = Math.pow(Cm, 7);
  const G = 0.5 * (1 - Math.sqrt(Cm7 / (Cm7 + Math.pow(25, 7))));
  const a1p = (1 + G) * a1, a2p = (1 + G) * a2;
  const C1p = Math.hypot(a1p, b1), C2p = Math.hypot(a2p, b2);
  const hp = (a, b) => { if (a === 0 && b === 0) return 0; const h = Math.atan2(b, a) / deg; return h < 0 ? h + 360 : h; };
  const h1p = hp(a1p, b1), h2p = hp(a2p, b2);
  const dLp = L2 - L1, dCp = C2p - C1p;
  let dhp = 0;
  if (C1p * C2p !== 0) { dhp = h2p - h1p; if (dhp > 180) dhp -= 360; else if (dhp < -180) dhp += 360; }
  const dHp = 2 * Math.sqrt(C1p * C2p) * Math.sin((dhp / 2) * deg);
  const Lmp = (L1 + L2) / 2, Cmp = (C1p + C2p) / 2;
  let hmp = h1p + h2p;
  if (C1p * C2p !== 0) { if (Math.abs(h1p - h2p) > 180) hmp += h1p + h2p < 360 ? 360 : -360; hmp /= 2; }
  const T = 1 - 0.17 * Math.cos((hmp - 30) * deg) + 0.24 * Math.cos(2 * hmp * deg)
    + 0.32 * Math.cos((3 * hmp + 6) * deg) - 0.2 * Math.cos((4 * hmp - 63) * deg);
  const dTheta = 30 * Math.exp(-Math.pow((hmp - 275) / 25, 2));
  const Cmp7 = Math.pow(Cmp, 7);
  const RC = 2 * Math.sqrt(Cmp7 / (Cmp7 + Math.pow(25, 7)));
  const SL = 1 + (0.015 * Math.pow(Lmp - 50, 2)) / Math.sqrt(20 + Math.pow(Lmp - 50, 2));
  const SC = 1 + 0.045 * Cmp, SH = 1 + 0.015 * Cmp * T;
  const RT = -Math.sin(2 * dTheta * deg) * RC;
  const x = dLp / (kL * SL), y = dCp / (kC * SC), z = dHp / (kH * SH);
  return Math.sqrt(x * x + y * y + z * z + RT * y * z);
}

// ---------------------------------------------------------------- image helpers
export function downscale(img, maxSide) {
  const { width: W, height: H, data } = img;
  const s = Math.min(1, maxSide / Math.max(W, H));
  if (s === 1) return img;
  const w = Math.max(1, Math.round(W * s)), h = Math.max(1, Math.round(H * s));
  const out = new Uint8ClampedArray(w * h * 4);
  const fx = W / w, fy = H / h;
  for (let y = 0; y < h; y++) {
    const y0 = Math.floor(y * fy), y1 = Math.max(y0 + 1, Math.floor((y + 1) * fy));
    for (let x = 0; x < w; x++) {
      const x0 = Math.floor(x * fx), x1 = Math.max(x0 + 1, Math.floor((x + 1) * fx));
      let r = 0, g = 0, b = 0, n = 0;
      for (let yy = y0; yy < y1; yy++) for (let xx = x0; xx < x1; xx++) {
        const i = (yy * W + xx) * 4; r += data[i]; g += data[i + 1]; b += data[i + 2]; n++;
      }
      const o = (y * w + x) * 4; out[o] = r / n; out[o + 1] = g / n; out[o + 2] = b / n; out[o + 3] = 255;
    }
  }
  return { data: out, width: w, height: h };
}

// Transpose so that the gap always runs vertically in our working image.
function transpose(img) {
  const { width: W, height: H, data } = img;
  const out = new Uint8ClampedArray(W * H * 4);
  for (let y = 0; y < H; y++) for (let x = 0; x < W; x++) {
    const i = (y * W + x) * 4, o = (x * H + y) * 4;
    out[o] = data[i]; out[o + 1] = data[i + 1]; out[o + 2] = data[i + 2]; out[o + 3] = 255;
  }
  return { data: out, width: H, height: W };
}

function luma(img) {
  const { width: W, height: H, data } = img;
  const L = new Float32Array(W * H);
  for (let i = 0, j = 0; j < L.length; i += 4, j++) L[j] = 0.299 * data[i] + 0.587 * data[i + 1] + 0.114 * data[i + 2];
  return L;
}

function boxBlur(src, W, H, r) {
  const tmp = new Float32Array(W * H), out = new Float32Array(W * H);
  for (let y = 0; y < H; y++) {
    let s = 0; const row = y * W;
    for (let x = -r; x <= r; x++) s += src[row + Math.min(W - 1, Math.max(0, x))];
    for (let x = 0; x < W; x++) {
      tmp[row + x] = s / (2 * r + 1);
      s += src[row + Math.min(W - 1, x + r + 1)] - src[row + Math.max(0, x - r)];
    }
  }
  for (let x = 0; x < W; x++) {
    let s = 0;
    for (let y = -r; y <= r; y++) s += tmp[Math.min(H - 1, Math.max(0, y)) * W + x];
    for (let y = 0; y < H; y++) {
      out[y * W + x] = s / (2 * r + 1);
      s += tmp[Math.min(H - 1, y + r + 1) * W + x] - tmp[Math.max(0, y - r) * W + x];
    }
  }
  return out;
}

const quantile = (arr, q) => {
  if (!arr.length) return NaN;
  const a = Float64Array.from(arr).sort();
  const p = (a.length - 1) * q, i = Math.floor(p), f = p - i;
  return i + 1 < a.length ? a[i] * (1 - f) + a[i + 1] * f : a[i];
};
const median = (arr) => quantile(arr, 0.5);

// ---------------------------------------------------------------- gap finding
// The gap between panels shows up as a thin dark line (shadow) with strong across-gap gradient.
export function findGap(img, guide, opt = DEFAULTS) {
  const { width: W, height: H } = img;
  const L = luma(img);
  const y0 = Math.floor(H * opt.sampleSpan[0]), y1 = Math.ceil(H * opt.sampleSpan[1]);
  const lo = Math.max(2, Math.floor(W * (guide - opt.searchBand))), hi = Math.min(W - 3, Math.ceil(W * (guide + opt.searchBand)));
  const prof = new Float32Array(W), grad = new Float32Array(W);
  for (let x = 0; x < W; x++) {
    let s = 0, g = 0;
    for (let y = y0; y < y1; y++) {
      const v = L[y * W + x]; s += v;
      if (x > 0 && x < W - 1) g += Math.abs(L[y * W + x + 1] - L[y * W + x - 1]);
    }
    prof[x] = s / (y1 - y0); grad[x] = g / (y1 - y0);
  }
  // darkness relative to a wider neighbourhood: a gap is darker than both sides
  const k = Math.max(4, Math.round(W * 0.02));
  let best = -Infinity, bx = Math.round(W * guide), second = -Infinity;
  const scores = [];
  for (let x = lo; x <= hi; x++) {
    let l = 0, r = 0, n = 0;
    for (let d = 2; d <= k; d++) { if (x - d >= 0) l += prof[x - d]; if (x + d < W) r += prof[x + d]; n++; }
    const dark = Math.min(l, r) / n - prof[x];        // how much darker than the darker side
    const s = Math.max(0, dark) + 0.5 * grad[x];
    scores.push([x, s]);
    if (s > best) { second = best; best = s; bx = x; }
  }
  // contrast of the peak against the band's median: low means we did not find a clear gap
  const med = median(scores.map((v) => v[1]));
  const strength = med > 0 ? best / med : best > 0 ? 10 : 0;
  return { x: bx, strength, found: strength >= 1.8 };
}

// ---------------------------------------------------------------- one side of the gap
// Light on a car body = paint colour x how much light falls on that spot + white-ish reflection added on top.
// So we take the darker, reflection-free part of each box (diffuse paint) and compare colour at equal brightness.
// Brightness differences between the two panels come mostly from their angle to the light, not from the paint.
function sampleRegion(img, x0, x1, y0, y1, Lfull, maps, opt) {
  const { width: W, data } = img;
  const lum = [];
  for (let y = y0; y < y1; y++) for (let x = x0; x < x1; x++) lum.push(Lfull[y * W + x]);
  const q = (v) => quantile(lum, v);
  const dLo = q(opt.diffuse[0]), dHi = q(opt.diffuse[1]), specQ = q(0.9);
  let clipped = 0, total = 0;
  const R = [], G = [], B = [], hf = [], spec = [], as = [], bs = [];
  for (let y = y0; y < y1; y++) for (let x = x0; x < x1; x++) {
    const i = (y * W + x) * 4, v = Lfull[y * W + x];
    total++;
    if (data[i] > 250 || data[i + 1] > 250 || data[i + 2] > 250) { clipped++; continue; }
    if (v >= specQ) spec.push(maps.grad[y * W + x]);
    if (v < dLo || v > dHi) continue;
    R.push(LIN[data[i]]); G.push(LIN[data[i + 1]]); B.push(LIN[data[i + 2]]);
    hf.push(Math.abs(maps.detail[y * W + x]));
    const l = rgbToLab(data[i], data[i + 1], data[i + 2]); as.push(l[1]); bs.push(l[2]);
  }
  const rgb = [median(R), median(G), median(B)];
  return {
    rgb,                                  // linear RGB of the diffuse paint
    rawLab: rgbToLab(...rgb.map(toByte)), // as photographed (brightness included)
    n: R.length,
    // spread of colour inside the box: high means the box covers more than one surface (trim, badge, dirt, a sticker)
    spread: Math.hypot(quantile(as, 0.75) - quantile(as, 0.25), quantile(bs, 0.75) - quantile(bs, 0.25)),
    texture: median(hf),                  // fine-grain roughness of the paint (orange peel, sanding marks)
    gloss: spec.length ? median(spec) : 0, // sharpness at reflections: crisp on factory clear coat, often softer on repaints
    hasReflection: specQ > 1.3 * q(0.5) && specQ - q(0.5) > 25, // a real reflection, not just the bright end of noise
    clipped: clipped / total,
  };
}
const toByte = (c) => { const v = c <= 0.0031308 ? 12.92 * c : 1.055 * Math.pow(c, 1 / 2.4) - 0.055; return Math.max(0, Math.min(255, Math.round(v * 255))); };
const Yof = (rgb) => 0.2126729 * rgb[0] + 0.7151522 * rgb[1] + 0.072175 * rgb[2];
function labAtY(rgb, Y) {
  const k = Y / Math.max(1e-6, Yof(rgb));
  const [r, g, b] = rgb.map((c) => c * k);
  // exact float conversion (no byte rounding)
  const X = (0.4124564 * r + 0.3575761 * g + 0.1804375 * b) / 0.95047;
  const YY = 0.2126729 * r + 0.7151522 * g + 0.072175 * b;
  const Z = (0.0193339 * r + 0.119192 * g + 0.9503041 * b) / 1.08883;
  const fx = fLab(X), fy = fLab(YY), fz = fLab(Z);
  return [116 * fy - 16, 500 * (fx - fy), 200 * (fy - fz)];
}

// ---------------------------------------------------------------- one joint
// orientation: "vertical" (gap runs up-down in the photo) or "horizontal"
// guide: where the user aligned the guide line, 0..1 across the gap direction
export function analyzeJoint(imgIn, { orientation = "vertical", guide = 0.5, gapX = null, factor = 1 } = {}, opt = DEFAULTS) {
  let img = downscale(imgIn, opt.maxSide);
  if (orientation === "horizontal") img = transpose(img);
  const { width: W, height: H } = img;
  const gap = gapX != null ? { x: Math.round(gapX * W), strength: 99, found: true, manual: true } : findGap(img, guide, opt);
  const L = luma(img);
  const blur = boxBlur(L, W, H, 2);
  const detail = new Float32Array(W * H), grad = new Float32Array(W * H);
  for (let y = 1; y < H - 1; y++) for (let x = 1; x < W - 1; x++) {
    const i = y * W + x;
    detail[i] = L[i] - blur[i];
    grad[i] = Math.hypot(L[i + 1] - L[i - 1], L[i + W] - L[i - W]);
  }
  const m = Math.round(W * opt.margin), sw = Math.round(W * opt.sampleWidth);
  const y0 = Math.floor(H * opt.sampleSpan[0]), y1 = Math.ceil(H * opt.sampleSpan[1]);
  const lb = [Math.max(1, gap.x - m - sw), Math.max(2, gap.x - m)];
  const rb = [Math.min(W - 2, gap.x + m), Math.min(W - 1, gap.x + m + sw)];
  const maps = { detail, grad };
  const A = sampleRegion(img, lb[0], lb[1], y0, y1, L, maps, opt);
  const B = sampleRegion(img, rb[0], rb[1], y0, y1, L, maps, opt);
  // colour difference at equal brightness (shading removed), plus the brightness difference as a separate feature
  const Ym = (Yof(A.rgb) + Yof(B.rgb)) / 2;
  A.lab = labAtY(A.rgb, Ym); B.lab = labAtY(B.rgb, Ym);
  const dE = deltaE2000(A.lab, B.lab);
  const dL = Math.abs(A.rawLab[0] - B.rawLab[0]);
  const ratio = (p, q) => (p > 0 && q > 0 ? Math.abs(Math.log(p / q)) : 0);
  const dTexture = ratio(A.texture, B.texture);
  // reflection sharpness is only comparable when both sides actually show a reflection
  const glossOk = A.hasReflection && B.hasReflection;
  const dGloss = glossOk ? ratio(A.gloss, B.gloss) : 0;
  const score = dE + opt.wTexture * dTexture + opt.wGloss * dGloss + opt.wLight * dL;

  // photo quality checks: the result is only as good as the shot
  const warnings = [];
  let sharp = 0, cnt = 0;
  for (let y = y0; y < y1; y += 2) for (let x = 2; x < W - 2; x += 2) { sharp += Math.abs(detail[y * W + x]); cnt++; }
  sharp /= cnt;
  if (!gap.found) warnings.push("gap_not_found");
  if (lb[1] - lb[0] < sw * 0.6 || rb[1] - rb[0] < sw * 0.6) warnings.push("gap_near_edge");
  if (A.n < 200 || B.n < 200) warnings.push("too_few_pixels");
  if (Math.max(A.spread, B.spread) > 8) warnings.push("mixed_surface");
  if (Math.max(A.clipped, B.clipped) > 0.5) warnings.push("glare"); // clipped pixels are skipped anyway; only block when most of a box is blown out
  if (Math.min(A.rawLab[0], B.rawLab[0]) < 8) warnings.push("too_dark");
  if (sharp < 0.25) warnings.push("blurry");

  const eff = score / factor;
  const verdict = eff >= opt.flagAt ? "differs" : eff >= opt.checkAt ? "check" : "match";
  const toFrac = (v) => v / W;
  return {
    verdict,
    score: +score.toFixed(3),
    dE: +dE.toFixed(3),
    dTexture: +dTexture.toFixed(3),
    dGloss: +dGloss.toFixed(3),
    dL: +dL.toFixed(2),
    glossCompared: glossOk,
    sides: [A, B].map((s) => ({ lab: s.lab.map((v) => +v.toFixed(2)), rawLab: s.rawLab.map((v) => +v.toFixed(2)), texture: +s.texture.toFixed(3), gloss: +s.gloss.toFixed(2), spread: +s.spread.toFixed(2), n: s.n })),
    gap: { x: toFrac(gap.x), strength: +gap.strength.toFixed(2), found: gap.found, manual: !!gap.manual },
    boxes: { a: lb.map(toFrac), b: rb.map(toFrac), span: opt.sampleSpan, orientation },
    warnings,
    reliable: !warnings.some((w) => ["gap_not_found", "too_few_pixels", "mixed_surface", "glare", "too_dark", "blurry"].includes(w)),
  };
}

// ---------------------------------------------------------------- whole car
// joints: [{a: panelId, b: panelId, result}] → per-panel status.
// A panel is called repainted when it differs from every neighbour we measured it against
// while at least one of those neighbours matches somebody else ("the odd one out").
export function judgePanels(joints, opt = DEFAULTS) {
  const byPanel = new Map();
  const add = (p, j) => { if (!byPanel.has(p)) byPanel.set(p, []); byPanel.get(p).push(j); };
  for (const j of joints) if (j.result && j.result.reliable) { add(j.a, j); add(j.b, j); }
  // j.factor > 1 for joints that naturally differ a little (plastic bumper against metal)
  const eff = (j) => j.result.score / (j.factor || 1);
  const differs = (j) => eff(j) >= opt.flagAt;
  const checks = (j) => eff(j) >= opt.checkAt;
  const out = {};
  for (const [p, js] of byPanel) {
    const allDiffer = js.every(differs), anyDiffer = js.some(differs), anyCheck = js.some(checks);
    // neighbours that agree with someone else are anchors of "original paint"
    const anchored = js.some((j) => {
      const other = j.a === p ? j.b : j.a;
      return (byPanel.get(other) || []).some((k) => k !== j && !checks(k));
    });
    let status = "original";
    if (allDiffer && (js.length >= 2 || anchored)) status = "repainted";
    else if (anyDiffer) status = "suspect";
    else if (anyCheck) status = "check";
    out[p] = { status, joints: js.length, worst: +Math.max(...js.map(eff)).toFixed(2) };
  }
  return out;
}

// Lab (D65) back to sRGB bytes, for showing the measured paint as swatches.
export function labToRgb([L, a, b]) {
  const fy = (L + 16) / 116, fx = fy + a / 500, fz = fy - b / 200;
  const inv = (t) => (t ** 3 > 216 / 24389 ? t ** 3 : (116 * t - 16) / (24389 / 27));
  const X = inv(fx) * 0.95047, Y = inv(fy), Z = inv(fz) * 1.08883;
  const r = 3.2404542 * X - 1.5371385 * Y - 0.4985314 * Z;
  const g = -0.969266 * X + 1.8760108 * Y + 0.041556 * Z;
  const bb = 0.0556434 * X - 0.2040259 * Y + 1.0572252 * Z;
  return [r, g, bb].map(toByte);
}
