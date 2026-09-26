// The 14 official event classes: how we detect each one, and how it is drawn.
// Color encodes the event family (identity comes from the row label, never color alone).

export type Family = "collision" | "violation" | "pedestrian" | "flow";

export const FAMILY_COLOR: Record<Family, string> = {
  collision: "var(--fam-collision)",
  violation: "var(--fam-violation)",
  pedestrian: "var(--fam-pedestrian)",
  flow: "var(--fam-flow)",
};

export const FAMILY_LABEL: Record<Family, string> = {
  collision: "Collision risk",
  violation: "Traffic violation",
  pedestrian: "Pedestrian safety",
  flow: "Flow & obstruction",
};

export interface ClassInfo {
  id: string;
  name: string;
  family: Family;
  rule: string;
  learned: string;
  needs: string;
}

export const CLASSES: ClassInfo[] = [
  { id: "accident", name: "Accident", family: "collision", learned: "YOLO11 detections + ByteTrack tracks",
    rule: "Two road users' ground footprints touch after at least 1 s of history, and one of them shows an impact signature: it loses most of its speed within a second, or a stationary one is shoved. Both then come to rest. Single-vehicle crashes: an abrupt stop from high speed with no queue ahead.",
    needs: "Tracks only" },
  { id: "near_miss", name: "Near miss", family: "collision", learned: "Tracks",
    rule: "Hard braking (≥60% of speed and ≥1.2 sizes/s lost within 0.8 s, box shape stable) while a road user ahead in the same lane is under 1.5 s time-to-collision, outside stop-and-go queues, with no contact afterwards.",
    needs: "Tracks only" },
  { id: "red_light", name: "Red-light running", family: "violation", learned: "Tracks + traffic-light colour reader",
    rule: "The vehicle's front crosses the stop line in the legal direction while its signal head reads red; the event lasts until it leaves the intersection.",
    needs: "Stop line + signal ROI" },
  { id: "wrong_way", name: "Wrong-way driving", family: "violation", learned: "Flow field learned from all trajectories",
    rule: "The vehicle's net motion over ≥1.5 s points against the legal direction at its location (lane map, or the learned dominant flow where the map is silent).",
    needs: "Lane map or learned flow" },
  { id: "illegal_u_turn", name: "Illegal U-turn", family: "violation", learned: "Tracks",
    rule: "Entry and exit travel directions (net displacement over 1 s windows) differ by ≥150°, outside any zone where U-turns are allowed.",
    needs: "Tracks (+ optional allowed zones)" },
  { id: "stopped_vehicle", name: "Stopped vehicle", family: "flow", learned: "Tracks",
    rule: "A vehicle is stationary on the carriageway for ≥10 s. Stationary runs at the same spot are merged across track IDs; queues (a stopped car right ahead, or a non-green signal at the stop line) are excluded.",
    needs: "Carriageway (drawn or learned)" },
  { id: "jaywalking", name: "Jaywalking", family: "pedestrian", learned: "Person detections",
    rule: "A pedestrian's feet are on the carriageway outside every crosswalk for ≥1 s. Riders of bikes and motorbikes, and people inside vehicles, are filtered out.",
    needs: "Carriageway + crosswalks" },
  { id: "failure_to_yield", name: "Failure to yield", family: "pedestrian", learned: "Tracks",
    rule: "A moving vehicle is inside a crosswalk polygon while a pedestrian is on it or stepping onto it.",
    needs: "Crosswalk polygons" },
  { id: "illegal_turn", name: "Illegal turn", family: "violation", learned: "Tracks",
    rule: "Entry gate → exit gate is not in the list of allowed movements; the event spans the turn.",
    needs: "Gates + allowed movements" },
  { id: "solid_line_crossing", name: "Solid-line crossing", family: "violation", learned: "Tracks",
    rule: "The vehicle's bottom-centre changes side of a solid marking and stays there; it starts when the first wheel corner crosses and ends when both are across.",
    needs: "Solid-line polylines" },
  { id: "stop_line", name: "Stop-line violation", family: "violation", learned: "Tracks + signal reader",
    rule: "The vehicle stops with its front past the stop line during red without entering the intersection; ends when the signal turns green.",
    needs: "Stop line + signal ROI" },
  { id: "congestion", name: "Congestion", family: "flow", learned: "Tracks + learned direction groups",
    rule: "In one direction of travel, ≥4 vehicles are present and ≥75% of them crawl below 0.6 sizes/s for ≥15 s (enter/exit hysteresis). Plain red-light queues that release on green are excluded.",
    needs: "Learned flow (automatic)" },
  { id: "road_obstacle", name: "Road obstacle", family: "flow", learned: "COCO animal classes",
    rule: "An animal is on the carriageway for ≥1.5 s.",
    needs: "Carriageway" },
  { id: "fire_smoke", name: "Fire / smoke", family: "collision", learned: "—",
    rule: "Not predicted: we have no detector we trust, and predicting an absent class adds a zero to the macro F1.",
    needs: "—" },
];

export const CLASS_BY_ID: Record<string, ClassInfo> = Object.fromEntries(CLASSES.map((c) => [c.id, c]));

export function classColor(id: string): string {
  return FAMILY_COLOR[CLASS_BY_ID[id]?.family ?? "flow"];
}

export function className(id: string): string {
  return CLASS_BY_ID[id]?.name ?? id;
}
