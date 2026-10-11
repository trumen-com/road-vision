// Shared types. All geometry is in model-input pixels (the frame resized/letterboxed to IN_W x IN_H).

export const IN_W = 512;
export const IN_H = 288;

// COCO ids we care about
export const PERSON = 0, BICYCLE = 1, CAR = 2, MOTORCYCLE = 3, BUS = 5, TRUCK = 7;
export const ROAD_USERS = new Set([PERSON, BICYCLE, CAR, MOTORCYCLE, BUS, TRUCK, 16, 17, 18, 19]); // + dog, horse, sheep, cow
export const VULNERABLE = new Set([PERSON, BICYCLE, MOTORCYCLE, 16, 17, 18, 19]);
export const CLASS_NAME: Record<number, string> = {
  0: "person", 1: "bicycle", 2: "car", 3: "motorcycle", 5: "bus", 7: "truck", 16: "dog", 17: "horse", 18: "sheep", 19: "cow",
};

export interface Detection { x1: number; y1: number; x2: number; y2: number; score: number; cls: number }

export interface Sample { t: number; x1: number; y1: number; x2: number; y2: number; score: number; cut: number }
// cut bit flags: 1 = left/right image border, 2 = bottom border, 4 = top border

/** Ego-path corridor: a trapezoid from the bottom of the frame to the vanishing point (normalised 0..1). */
export interface Corridor { vpX: number; vpY: number; bottomHalfWidth: number }
export const DEFAULT_CORRIDOR: Corridor = { vpX: 0.5, vpY: 0.42, bottomHalfWidth: 0.36 };

export interface TrackThreat {
  id: number; cls: number; box: [number, number, number, number];
  ttc: number;            // seconds, Infinity if not closing
  inPathNow: boolean;     // overlaps the ego corridor now
  inPathSoon: boolean;    // predicted to be in the corridor when contact would happen (cut-in / crossing)
  proximity: number;      // 0 (horizon) .. 1 (bottom of the frame)
  offsetM: number;        // lateral offset from our path centre, metres (from the object's own width)
  lateralMs: number;      // sideways speed relative to our heading, m/s
  tauDot: number;         // d(TTC)/dt; near -1 on a collision course, above -0.5 when braking in control
  risk: number;           // 0..1 per-object threat
}

export interface FrameFeatures {
  minTtcPath: number;     // min TTC among in-path objects (capped at 10 s)
  minTtcSoon: number;     // min TTC among objects predicted to enter the path
  maxLooming: number;     // max d(log size)/dt among in-path objects (1/s)
  maxProxPath: number;    // closest in-path object (0..1)
  maxWidthPath: number;   // widest in-path object as a fraction of frame width
  nPath: number;          // in-path road users
  vulnPath: number;       // 1 if a pedestrian / cyclist / motorcyclist / animal is in or entering the path
  egoShift: number;       // |median lateral motion| (px/s, a turning proxy)
  maxHeightPath: number;  // tallest in-path object, fraction of frame height (closeness)
  ttcOfBiggest: number;   // TTC of the biggest in-path object (capped at 10 s)
  minAbsOffset: number;   // smallest |lateral offset| (m) among closing objects (capped at 5)
  nTracks: number;        // tracked road users (scene complexity)
  vulnNear: number;       // proximity of the nearest vulnerable road user in or entering the path (0 if none)
  tauDotPath: number;     // tau-dot of the in-path (or entering) object with the shortest TTC (0.5 if none)
  minTauDot: number;      // most negative tau-dot among closing objects within 1.5 m of our path
  heuristic: number;      // hand-made threat score (0..1)
}

export type AlertLevel = 0 | 1 | 2;   // 0 none, 1 caution, 2 brake

export interface GuardState {
  t: number;
  detections: number;
  threats: TrackThreat[];
  features: FrameFeatures;
  risk: number;           // calibrated, smoothed frame risk 0..1
  level: AlertLevel;
  top?: TrackThreat;      // the object that drives the alert
  inferMs: number;
  geoRisk?: number;       // geometric model output before fusion and smoothing
  aggregate?: number[];   // aggregated features (exported for training only)
}
