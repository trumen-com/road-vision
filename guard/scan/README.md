# Trumen Scan (add-on built during the final)

A phone walk-around of a car becomes a 3D model of **that exact car's exterior**, split into named parts you can take
apart, measure and search for. Live: **https://trumen-com.github.io/road-vision/scan/**

Sample: a Royal Motors test-drive Toyota Camry (XV80) photographed at the hackathon venue: a 29 s video (3 fps → 86
frames) plus 9 photos. Raw photos are **not** in the repository (number plate and bystanders are visible in them).

| Step | What | Tool |
|---|---|---|
| Poses | 95 of 95 views placed, 41,706 sparse points, 0.69 px mean reprojection error; the 86 video frames share one camera | COLMAP via pycolmap (BSD-3) |
| Parts | 33-class side-aware part masks per view | `mitbersh/car-parts-segmentation` (YOLO26-s, Hugging Face; trained on the HITL car parts dataset; licence not stated by the author: demo use only until cleared) |
| Depth | Monocular depth per view, aligned to that view's SfM points with a scale, offset and tilt term (median error 0.8% of depth) | Depth Anything V2 Small (Apache-2.0) |
| Fusion | Each point must agree (within 1.5% depth, same part) with 2 of its 6 nearest views; 2 cm voxels seen in ≥3 views; majority part vote; holes filled from the unchecked depth only where nothing else is within 6 cm | own code |
| Axes and scale | Up from a plane fitted to the ground around the car, forward from the car's left-right symmetry; scale from the published length 4,915 mm | own code |
| Output | `camry_parts.glb` (one node per part), `parts.json` (name, side, size, centre, search links), viewer | trimesh, own code |

Privacy: only pixels inside car-part masks are used, so people around the car never enter the model, and the
number-plate class is dropped.

## Tuning (11 October)

`pipeline/tune.py` scores a build against three published figures the scale does not use, plus a figure that needs
no reference at all: how thick a flat panel comes out (RMS distance to a local plane in 15 cm patches).

| | Wheelbase (2.825 m) | Body width (1.84 m) | Height (1.445 m) | Panel thickness | Points |
|---|---|---|---|---|---|
| Before tuning | +13 mm | +75 mm | -15 mm | 15 mm | 187,473 |
| After tuning | +2 mm | +16 mm | +4 mm | 10 mm | 116,155 |

What mattered, in order: a tilt term in the per-view depth alignment (thickness 14 → 9.5 mm, width error halved), axes
from the ground plane and symmetry (removed a 4 cm left-right offset and most of the height error), and the neighbour-view
depth check. Sharing one camera for the video frames changed the numbers by a few millimetres, which is within noise.
Width is the 95th–5th percentile of the side panels; the 99th–1st gives 1.870 m. These figures were used
while tuning on this one car, so they are optimistic for a new car; panel thickness is the honest quality figure.
Side views before and after: `before_after.png`.

**Limits.** Exterior only: hidden parts cannot be seen by a camera; in a product they come from the OEM parts catalogue
by VIN (Royal Motors access). The vehicle model is set by hand from the badge in this demo; part searches are text
queries, not verified part numbers. Glossy panels give noisier depth than textured ones.

Run: `scan/pipeline/run.sh` (see the header for inputs and dependencies).
