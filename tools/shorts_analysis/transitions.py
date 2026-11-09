#!/usr/bin/env python3
"""
Смены картинки, включая МЯГКИЕ (наплыв, пролёт камеры сквозь сцену, вспышка),
которые не ловит детектор жёстких склеек. Каждый кадр, 30 fps.
  python3 -I tools/shorts_analysis/transitions.py shorts/ref
Пишет <папка>/analysis/<ролик>.transitions.json и печатает итог.
"""
import json, subprocess, sys
from pathlib import Path

import cv2
import numpy as np

FPS = 30


def hists(path, w=160):
    j = json.loads(subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
                                   "stream=width,height", "-of", "json", str(path)], capture_output=True, text=True).stdout)["streams"][0]
    h = int(round(j["height"] * w / j["width"] / 2) * 2)
    p = subprocess.Popen(["ffmpeg", "-v", "error", "-i", str(path), "-vf", f"fps={FPS},scale={w}:{h}",
                          "-f", "rawvideo", "-pix_fmt", "bgr24", "-"], stdout=subprocess.PIPE)
    out = []
    while True:
        b = p.stdout.read(w * h * 3)
        if len(b) < w * h * 3:
            break
        fr = np.frombuffer(b, np.uint8).reshape(h, w, 3)
        # убираем полосу субтитров и верхний HUD, чтобы считать смену КАРТИНКИ, а не текста
        fr = np.vstack([fr[int(h * 0.12):int(h * 0.60)], fr[int(h * 0.82):]])
        hsv = cv2.cvtColor(fr, cv2.COLOR_BGR2HSV)
        hh = cv2.calcHist([hsv], [0, 1, 2], None, [16, 8, 8], [0, 180, 0, 256, 0, 256])
        # плюс грубая «раскладка» кадра 4x8 — ловит смену композиции при похожих цветах
        lay = cv2.resize(cv2.cvtColor(fr, cv2.COLOR_BGR2GRAY), (4, 8), interpolation=cv2.INTER_AREA).ravel() / 255
        out.append((cv2.normalize(hh, hh).ravel(), lay.astype(np.float32)))
    return out


def run(path):
    H = hists(path)
    n = len(H)
    def dist(a, b):
        return max(1 - cv2.compareHist(H[a][0], H[b][0], cv2.HISTCMP_CORREL), float(np.abs(H[a][1] - H[b][1]).mean()) * 4)
    D1 = np.array([0] + [dist(i, i - 1) for i in range(1, n)])
    D15 = np.array([0] * 15 + [dist(i, i - 15) for i in range(15, n)])
    events = []
    i = 1
    while i < n:
        if D1[i] > 0.45:                                   # жёсткая склейка
            events.append({"t": round(i / FPS, 2), "type": "cut"}); i += 8; continue
        if D15[i] > 0.5:                                   # мягкая смена за <= 0.5 с
            j = i + int(np.argmax(D15[i:i + 15]))
            kind = "dissolve" if D1[max(1, j - 15):j + 1].max() < 0.25 else "fast-move"
            events.append({"t": round(j / FPS, 2), "type": kind}); i = j + 15; continue
        i += 1
    dur = n / FPS
    gaps = np.diff([0] + [e["t"] for e in events] + [dur])
    rep = {"file": Path(path).name, "duration": round(dur, 2), "changes": len(events),
           "changes_per_sec": round(len(events) / dur, 2),
           "avg_hold_sec": round(float(gaps.mean()), 2), "max_hold_sec": round(float(gaps.max()), 2),
           "types": {k: sum(e["type"] == k for e in events) for k in ("cut", "dissolve", "fast-move")},
           "first_change_at": events[0]["t"] if events else None, "events": events}
    return rep


def main():
    d = Path(sys.argv[1]); out = d / "analysis"; out.mkdir(exist_ok=True)
    for f in sorted(d.glob("*.mp4")):
        r = run(f)
        (out / f"{f.stem}.transitions.json").write_text(json.dumps(r, ensure_ascii=False, indent=1))
        print(f"{f.name[:14]}  смен {r['changes']:3d}  {r['changes_per_sec']}/с  ср.держим {r['avg_hold_sec']}с  "
              f"макс {r['max_hold_sec']}с  первая {r['first_change_at']}с  {r['types']}", flush=True)


if __name__ == "__main__":
    main()
