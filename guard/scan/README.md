# Trumen Scan (add-on built during the final)

A phone walk-around of a car becomes a 3D model of **that exact car's exterior**, split into named parts you can take
apart, measure and search for. Live: **https://trumen-com.github.io/road-vision/scan/**

Sample: a Royal Motors test-drive Toyota Camry (XV80) photographed at the hackathon venue: a 29 s video (3 fps → 86
frames) plus 9 photos. Raw photos are **not** in the repository (number plate and bystanders are visible in them).

| Step | What | Tool |
|---|---|---|
| Poses | 95 of 95 views placed, 39,082 sparse points, 0.70 px mean reprojection error | COLMAP via pycolmap (BSD-3) |
| Parts | 33-class side-aware part masks per view | `mitbersh/car-parts-segmentation` (YOLO26-s, Hugging Face; trained on the HITL car parts dataset; licence not stated by the author: demo use only until cleared) |
| Depth | Monocular depth per view, aligned to that view's SfM points (median error 1.3% of depth) | Depth Anything V2 Small (Apache-2.0) |
| Fusion | 2 cm voxels kept only when seen in ≥3 views; majority part vote; largest connected piece per part | own code |
| Scale | Set from the published length 4,915 mm; check: width 1.93 m (1.84 m published, ours includes mirrors), height 1.36 m (1.445 m) | own code |
| Output | `camry_parts.glb` (one node per part), `parts.json` (name, side, size, centre, search links), viewer | trimesh, own code |

Privacy: only pixels inside car-part masks are used, so people around the car never enter the model, and the
number-plate class is dropped.

**Limits.** Exterior only: hidden parts cannot be seen by a camera; in a product they come from the OEM parts catalogue
by VIN (Royal Motors access). The vehicle model is set by hand from the badge in this demo; part searches are text
queries, not verified part numbers. Glossy panels give noisier depth than textured ones.

Run: `scan/pipeline/run.sh` (see the header for inputs and dependencies).
