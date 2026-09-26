// EDA findings from the four sample videos (C3896, C3897, C3902, C3905).

export const FINDINGS = [
  { title: "4K, 29.97 fps, four times of day",
    text: "All samples are 3840×2160 H.264 at 29.97 fps: three clips of about 5.5 minutes and one of 2 minutes, filmed in full daylight, golden hour and dusk. Decoding 4K dominates the cost, not detection.",
    impact: "Detection runs on every 2nd frame at 960 px; decoding of the skipped frames is kept cheap (grab without retrieve)." },
  { title: "The camera moves slightly between recordings",
    text: "Registering each clip to a reference frame gives shifts of up to ~60 px and ~1° of rotation (at 1080p). A fixed scene map would miss the crosswalks by half their width.",
    impact: "Every video is aligned to the reference frame with SIFT + a RANSAC similarity transform; the scene map moves with it." },
  { title: "One signal head is readable, and daylight washes it out",
    text: "Only the pole head at the end of the median faces the camera. It controls the queue that faces the camera: the queue discharges exactly when it turns green. In sunlight the lit lamp is barely brighter than the unlit ones.",
    impact: "Lamp state is found by clustering the red-minus-green colour of the brightest lamp pixels over the whole video, not by fixed thresholds." },
  { title: "A busy pedestrian scene",
    text: "Three zebra crossings, two refuge islands and a bus stop generate constant pedestrian traffic; many people cross next to the markings or walk between queued cars.",
    impact: "Jaywalking needs a hand-drawn carriageway and crosswalk map. A jaywalker must be well out on the road (not at a kerb or island edge) and move across the traffic direction; riders are filtered out." },
  { title: "Long stationary periods that are not events",
    text: "Cars wait tens of seconds at the stop line, buses dwell at the stop, and cars park along the left curb.",
    impact: "Stopped-vehicle ignores the stop-line queue, the bus stop and curb parking; only vehicles stopped elsewhere on the carriageway for 10 s or more count." },
];
