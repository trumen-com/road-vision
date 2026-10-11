// Physics checks on synthetic scenes with known answers:  npx tsx --test test/engine.test.ts
import { test } from "node:test";
import assert from "node:assert/strict";
import { Guard } from "../engine/guard";
import { IN_H } from "../engine/types";

const F = 215;                 // focal length in model pixels (about 100° horizontal field of view at 512 px)
const HORIZON = 0.42 * IN_H;   // default vanishing point row
// a car (1.5 m tall, 1.8 m wide) at lateral offset X metres and distance Z metres, camera 1.3 m above the road
function carBox(X: number, Z: number) {
  const h = (F * 1.5) / Z, w = (F * 1.8) / Z;
  const yBottom = HORIZON + (F * 1.3) / Z, cx = 256 + (F * X) / Z;
  return { x1: cx - w / 2, y1: yBottom - h, x2: cx + w / 2, y2: yBottom, score: 0.9, cls: 2 };
}
function run(seq: (t: number) => ReturnType<typeof carBox>[], seconds: number, fps = 10) {
  const g = new Guard(null as never);
  let st = null as ReturnType<Guard["stepDetections"]> | null;
  for (let i = 0; i <= seconds * fps; i++) st = g.stepDetections(seq(i / fps), i / fps);
  return st!;
}

test("time to contact from looming matches the true value", () => {
  // closing at 8 m/s from 30 m: after 1.5 s the car is 18 m away -> TTC = 2.25 s
  const st = run((t) => [carBox(0, 30 - 8 * t)], 1.5);
  const th = st.threats[0];
  assert.ok(Math.abs(th.ttc - 2.25) < 0.25, `ttc ${th.ttc}`);
  assert.ok(th.inPathNow, "car in our lane is in the path");
});

test("tau-dot is about -1 on a steady collision course", () => {
  const st = run((t) => [carBox(0, 30 - 8 * t)], 2.5);
  assert.ok(Math.abs(st.threats[0].tauDot + 1) < 0.35, `tauDot ${st.threats[0].tauDot}`);
});

test("lateral offset in metres stays constant as a parked car approaches", () => {
  // parked car 3.5 m to the right; we drive at 10 m/s
  const st = run((t) => [carBox(3.5, 30 - 8 * t)], 2.0);
  const th = st.threats[0];
  assert.ok(Math.abs(th.offsetM - 3.5) < 0.3, `offset ${th.offsetM}`);
  assert.equal(th.inPathNow, false, "a car one lane over is not in our path");
});

test("a car at constant distance is not a threat", () => {
  const st = run(() => [carBox(0, 15)], 2.0);
  assert.equal(st.threats[0].ttc, Infinity);
  assert.ok(st.risk < 0.3, `risk ${st.risk}`);
});

test("a pedestrian stepping into the path is flagged before entering it", () => {
  // person 0.5 m wide, 1.7 m tall, 12 m ahead, walking from 3 m right towards our path at 1.5 m/s while we close at 5 m/s
  const person = (t: number) => {
    const Z = 12 - 5 * t, X = 3 - 1.5 * t, h = (F * 1.7) / Z, w = (F * 0.5) / Z;
    const yb = HORIZON + (F * 1.3) / Z, cx = 256 + (F * X) / Z;
    return [{ x1: cx - w / 2, y1: yb - h, x2: cx + w / 2, y2: yb, score: 0.9, cls: 0 }];
  };
  const st = run(person, 1.0);
  const th = st.threats[0];
  assert.ok(th.inPathSoon || th.inPathNow, "crossing pedestrian is in or entering the path");
});
