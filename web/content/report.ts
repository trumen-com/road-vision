// Technical report content (one page). Update the numbers after running tools/tune.py on the dev labels.
import ablationRows from "./ablations.json";

export const WORKED = [
  { title: "Relative speed units", text: "Measuring speed in object sizes per second made one set of thresholds work from the bottom of the frame to the horizon, with no camera calibration." },
  { title: "Learned flow field for direction", text: "Legal direction per image cell is learned from the vehicles themselves. Wrong-way and congestion work even where the scene map has no lanes drawn." },
  { title: "Impact signature instead of box overlap", text: "Boxes overlap all the time in a perspective view. Requiring a real collapse of speed (within ~0.8 s), or a stationary vehicle being shoved, plus traffic around still flowing, removed almost all crash false alarms on dense footage." },
  { title: "Frame-border and occlusion guards", text: "Boxes cut by the frame edge or by an overpass make a car look like it stops. Dropping border samples and checking that the box area is stable around a speed drop removed most false 'hard braking' events." },
  { title: "Causal risk shaped for the metric", text: "Graded low scores keep the ranking (AP) informative; only a genuine collision course crosses 0.5, so alarms stay rare; a 1.2 s half-life keeps one danger episode as one alarm." },
];

export const DIDNT = [
  { title: "Swerve detection", text: "Per-sample yaw rate from boxes is too noisy at a distance; swerves produced false near misses in stop-and-go traffic. Disabled; near misses rely on hard braking with a road user ahead." },
  { title: "Heading-based U-turns", text: "Unwrapping noisy headings added up to phantom 180° turns for slow vehicles. Replaced by net-displacement directions, a minimum turn time and a lateral offset check." },
  { title: "Fire / smoke", text: "Colour heuristics fire on brake lights and signals. Without a trustworthy detector we do not predict this class." },
];

export const NEXT = [
  "Fine-tune the detector on frames from the samples (auto-labelled with a larger model, then corrected) to recover small, far vehicles.",
  "Train a light clip classifier on public crash datasets (CCD, DoTA) to re-score accident candidates.",
  "A ground-plane homography from lane markings, to report speeds in km/h and distances in metres.",
  "Per-lane signal mapping learned from which approach stops when each light turns red.",
];

export const FAILURES = [
  { title: "Dense stop-and-go queues", text: "Cars touching in the image while braking look like the start of a crash.", fix: "An accident must show an abrupt speed collapse or a shove, and the traffic around it must still be moving." },
  { title: "Vehicles leaving under an overpass or at the frame edge", text: "The box shrinks or sticks to the border, so the tracker sees a car decelerate to zero.", fix: "Border-touching samples are dropped; speed drops only count when the box area is stable." },
  { title: "Duplicate boxes (car + truck on one vehicle)", text: "Two detections of one object produce a perfect 'contact'.", fix: "Class-agnostic NMS, duplicate suppression in the tracker, and pairs with IoU > 0.6 are ignored." },
  { title: "People at the kerb counted as jaywalking", text: "Pedestrians waiting at the kerb, walking along the road edge or on refuge islands had their feet just inside the drawn carriageway.", fix: "A jaywalker must stand with road all around them (0.8 body sizes) and move ACROSS the traffic direction by 1.5 body sizes, away from any zebra." },
  { title: "Pedestrians waiting at a crossing's end", text: "Someone standing at the kerb end of a zebra while a car passed was read as a car failing to yield.", fix: "The pedestrian must be out on the road part of the crossing, and the car must pass within one car width of them." },
  { title: "A car hidden behind a bus", text: "A car pulling up behind a stopped bus vanished from tracking for a second; its speed 'collapsed' across the gap and looked like a crash.", fix: "An impact only counts when the track is continuous around the contact (no occlusion gap)." },
  { title: "Queues and turning traffic as stopped vehicles", text: "Cars waiting in a long queue, or in the junction for a gap to turn, stood still for more than 10 s.", fix: "A stop is ignored when most vehicles around are stopped too, or when it is inside the junction for under 40 s." },
  { title: "Far, small objects", text: "Below ~20 px, box jitter dominates the motion and headings flip.", fix: "Direction, turn and collision rules ignore objects smaller than 20–25 px." },
];

export interface AblationRow { name: string; score_a: number; micro_f1_05: number; agnostic_f1_05: number; wall_per_video_sec: number }

// written by tools/ablate.py (Part A metrics on our dev labels of the samples)
export const ABLATIONS: AblationRow[] = ablationRows as AblationRow[];

// Visual spot check after tuning (not a benchmark): raw rule hits over the four samples, before and after,
// and how many of a random sample of hits looked like the real event to us on the video.
export const SPOTCHECK = [
  { cls: "jaywalking", before: 95, after: 19, looked_real_before: "3 of 16", looked_real_after: "about 9 of 12" },
  { cls: "failure_to_yield", before: 119, after: 43, looked_real_before: "4 of 16", looked_real_after: "about 9 of 20" },
  { cls: "stopped_vehicle", before: 20, after: 2, looked_real_before: "few (mostly queues)", looked_real_after: "2 of 2" },
  { cls: "near_miss", before: 7, after: 2, looked_real_before: "0 of 7", looked_real_after: "unclear (far lanes)" },
  { cls: "accident", before: 1, after: 0, looked_real_before: "0 of 1 (car behind a bus)", looked_real_after: "no crashes in the samples" },
];
