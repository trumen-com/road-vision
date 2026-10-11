#!/usr/bin/env python3
"""Render a recorded demo: the original dashcam video with the Trumen Guard overlay drawn from engine states.

  python eval/render_demo.py --video clip.mp4 --states states.json --out demo.mp4 [--title "Sample 1"]
The overlay is the same information the web app shows: alert band, boxes with time to contact, risk gauge.
"""
from __future__ import annotations

import argparse
import json
import subprocess

import cv2
import numpy as np

IN_W, IN_H = 512, 288
CLS = {0: "person", 1: "bicycle", 2: "car", 3: "motorcycle", 5: "bus", 7: "truck"}
YEL, RED, ASPH, WHITE = (48, 194, 242), (43, 49, 224), (43, 38, 35), (255, 255, 255)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True)
    ap.add_argument("--states", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--title", default="")
    a = ap.parse_args()
    st = json.load(open(a.states))["frames"]
    cap = cv2.VideoCapture(a.video)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    W, H = 1280, 720
    band_h = 86
    ff = subprocess.Popen(["ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{W}x{H + band_h}", "-r", f"{fps:.3f}",
                           "-i", "-", "-c:v", "libx264", "-preset", "veryfast", "-crf", "24", "-pix_fmt", "yuv420p", "-movflags", "+faststart", a.out],
                          stdin=subprocess.PIPE)
    k, i = 0, 0
    sx, sy = W / IN_W, H / IN_H
    while True:
        ok, fr = cap.read()
        if not ok:
            break
        fr = cv2.resize(fr, (W, H))
        t = i / fps
        while k + 1 < len(st) and st[k + 1]["t"] <= t:
            k += 1
        s = st[k] if st and st[k]["t"] <= t + 1e-6 else None
        canvas = np.zeros((H + band_h, W, 3), np.uint8)
        lvl = s["level"] if s else 0
        band = RED if lvl == 2 else YEL if lvl == 1 else (56, 50, 46)
        canvas[:band_h] = band
        word = "BRAKE" if lvl == 2 else "CAUTION" if lvl == 1 else "Watching the road"
        fg = WHITE if lvl != 1 else ASPH
        cv2.putText(canvas, word, (24, 60), cv2.FONT_HERSHEY_DUPLEX, 1.6, fg, 3, cv2.LINE_AA)
        if s and s["top"] and lvl:
            tp = s["top"]; what = CLS.get(tp["cls"], "object")
            where = "ahead in your lane" if tp["now"] else "moving into your path"
            ttc = f", contact in {tp['ttc']:.1f} s" if tp["ttc"] is not None and tp["ttc"] < 99 else ""
            cv2.putText(canvas, f"{what} {where}{ttc}", (360 if lvl == 2 else 420, 58), cv2.FONT_HERSHEY_SIMPLEX, 0.9, fg, 2, cv2.LINE_AA)
        if s:
            for th in s["threats"]:
                x1, y1, x2, y2 = [int(v) for v in (th["box"][0] * sx, th["box"][1] * sy, th["box"][2] * sx, th["box"][3] * sy)]
                hot = th["risk"] >= 0.5
                col = (RED if th["risk"] >= 0.75 else YEL) if hot else ((230, 230, 230) if th["path"] else (150, 150, 150))
                cv2.rectangle(fr, (x1, y1), (x2, y2), col, 3 if hot else 1)
                if th["path"] and th["ttc"] is not None and th["ttc"] < 6:
                    lab = f"{th['ttc']:.1f} s"
                    cv2.rectangle(fr, (x1, y1 - 26), (x1 + 16 + 14 * len(lab), y1), col, -1)
                    cv2.putText(fr, lab, (x1 + 6, y1 - 7), cv2.FONT_HERSHEY_SIMPLEX, 0.7, ASPH if hot and th["risk"] < 0.75 else WHITE, 2, cv2.LINE_AA)
            r = s["risk"]
            cv2.rectangle(fr, (W - 230, H - 56), (W - 20, H - 20), (30, 30, 30), -1)
            cv2.rectangle(fr, (W - 230, H - 56), (W - 230 + int(210 * r), H - 20), RED if r >= 0.75 else YEL if r >= 0.5 else (120, 160, 90), -1)
            cv2.putText(fr, f"risk {r:.2f}", (W - 220, H - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, WHITE, 2, cv2.LINE_AA)
        if a.title:
            cv2.putText(fr, a.title, (20, H - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, WHITE, 2, cv2.LINE_AA)
        canvas[band_h:] = fr
        ff.stdin.write(canvas.tobytes())
        i += 1
    ff.stdin.close(); ff.wait()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
