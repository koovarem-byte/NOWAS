#!/usr/bin/env python3
"""Независимая проверка синхрона: время появления субтитра каждой фразы (прочитано по кадрам вручную),
перенесённое в шкалу фильма, против начала речи в РЕАЛЬНОМ звуке фильма (по громкости, без Whisper)."""
import json, subprocess, sys
from pathlib import Path
import numpy as np
edl = json.loads(Path("edl.json").read_text())
ph = json.loads(Path("work/phrases.json").read_text())
x = np.frombuffer(subprocess.run(["ffmpeg", "-v", "error", "-i", sys.argv[1], "-vn", "-ac", "1", "-ar", "16000", "-f", "f32le", "-"],
                                 capture_output=True).stdout, np.float32)
n = len(x) // 160
db = 20 * np.log10(np.sqrt((x[:n * 160].reshape(n, 160) ** 2).mean(1)) + 1e-6)
thr = np.percentile(db, 30) + 12            # порог речи относительно фона (подложка не тишина)
on = db > thr
onsets = np.array([i / 100 for i in range(6, n) if on[i] and not on[i - 6:i].any()])
t = 0.0; res = []
for s in edl["segments"]:
    if not s.get("vis") and s.get("freeze") is None:
        for p in ph:
            if p["part"] == s["part"] and s["in"] - 0.02 <= p["vt"] < s["out"] - 0.02:
                a = t + p["vt"] + 0.04 - s["in"]          # момент появления субтитра в фильме
                d = onsets - a
                near = d[np.abs(d) < 0.6]
                if len(near):
                    res.append((p["part"], float(near[np.argmin(np.abs(near))]), p["text"]))
    t += int(round((s["out"] - s["in"]) * 30)) / 30
d = np.array([r[1] for r in res])
print(f"фраз с паузой перед ними: {len(d)};  речь относительно субтитра: медиана {np.median(d)*1000:+.0f} мс, "
      f"|ошибка| медиана {np.median(np.abs(d))*1000:.0f} мс, 90% {np.percentile(np.abs(d),90)*1000:.0f} мс")
for part in ["p1", "p2", "p3_tiktok", "p4", "p5"]:
    e = np.array([r[1] for r in res if r[0] == part])
    if len(e):
        print(f"  {part:10s} {len(e):3d} фраз  медиана {np.median(e)*1000:+5.0f} мс  |90%| {np.percentile(np.abs(e),90)*1000:4.0f} мс")
bad = [r for r in res if abs(r[1]) > 0.2]
for r in bad:
    print(f"  > 0.2 с: {r[0]} {r[1]:+.2f}  «{r[2][:50]}»")
