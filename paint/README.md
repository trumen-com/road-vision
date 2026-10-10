# Trumen Paint Check

Repaint screening for trade-in cars with a phone camera. The inspector photographs each of the 14 gaps between body panels.
The app compares the paint on the two sides of each gap **in the same photo, under the same light**, and marks panels
that differ from all their neighbours as likely repainted.

Live: `https://trumen-com.github.io/road-vision/paint/` (static, runs fully in the browser; photos never leave the phone
unless you export them).

## How it works (`engine.js`)

1. **Find the gap.** Near the guide line, the gap is the thin dark line with the strongest across-gap gradient. If it
   can't find a clear one, it asks the user to tap the gap.
2. **Sample both sides.** For each side it takes a box next to the gap (outside the gap shadow). It drops blown-out
   pixels, reflections (top 40% of brightness) and shadow and dirt (bottom 15%), keeping the diffuse paint.
3. **Compare.**
   - **Colour:** CIEDE2000 between the two sides at equal brightness. Two panels face the light at different angles,
     so a brightness difference alone is not evidence.
   - **Texture:** log-ratio of fine-grain roughness (orange peel, sanding marks).
   - **Gloss:** log-ratio of edge sharpness inside reflections, only when both sides show a reflection.
4. **Score.** `score = ΔE + 1.2·texture + 0.8·gloss`. Joints against a plastic bumper are divided by 1.6, because bumpers
   are painted separately at the factory.
5. **Panels.** A panel is "likely repainted" when it differs from every neighbour it was measured against, and either it
   has two or more measured joints or a neighbour is anchored to original paint elsewhere.
6. **Photo checks.** No clear gap, too few clean paint pixels, a box covering trim or a sticker, glare over most of a
   box, too dark, or blurry → "Retake photo". These joints are not used.

Thresholds (`checkAt`, `flagAt`) and weights are placeholders until they are tuned on real cars with gauge readings
(below).

## Data collection and evaluation

- In the app, the **Collect data** tab stores each joint photo with the make, model, colour and the paint-thickness gauge
  reading of every panel. Inside Claude it uploads to the artifact's private store; on the public site it keeps the data
  on the phone and exports one `.zip` (`labels.json` + `photos/`).
- Ground truth: a panel reading **over 200 µm** counts as repainted. Factory paint is usually 80–150 µm.
- `node eval/run.mjs <unzipped export> --out results.json` reports:
  - joint-level ROC AUC with a 95% bootstrap CI, resampling whole cars;
  - AUC per feature;
  - caught / false flags at the shipped threshold, and with the threshold chosen leaving one car out;
  - panel-level detection;
  - the retake rate.
- `node eval/make_synthetic.mjs <dir>` makes a synthetic set **only to test the pipeline**. Its numbers are not results.
- `node eval/selftest.mjs` checks CIEDE2000 against the Sharma et al. reference pairs and runs synthetic joints.

## Limits

- Screening, not a verdict: confirm flagged panels with a paint gauge before pricing.
- A well-blended professional repaint can match in colour and texture; a full respray of the whole car has no "odd one
  out".
- Hard cases are metallic and pearl paints, very dark colours, and dirty or wet cars.
