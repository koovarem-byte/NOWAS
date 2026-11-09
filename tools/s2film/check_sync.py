#!/usr/bin/env python3
"""Проверка синхрона: для каждого появления субтитра — расстояние до ближайшего начала слова в голосе."""
import json
from pathlib import Path
import numpy as np
ph = json.loads(Path("work/phrases.json").read_text())
asr = {}
for part in ["p1", "p2", "p3_tiktok", "p4", "p5"]:
    cap = np.array(json.loads(Path(f"work/{part}.capon.json").read_text())["onsets"])
    starts = []
    for p in ph:
        if p["part"] != part:
            continue
        w = asr.setdefault(p["file"], json.loads(Path(f"work/asr/{p['file']}.json").read_text()))
        for x in w:
            if p["fs"] <= x["s"] < p["fe"]:
                starts.append(p["vt"] + (x["s"] - p["fs"]) / p["speed"])
    st = np.array(sorted(starts))
    lo, hi = st.min() - 0.3, st.max() + 1.0
    c = cap[(cap > lo) & (cap < hi)]
    d = np.abs(c[:, None] - st[None, :]).min(1)
    print(f"{part:10s} субтитров {len(c):3d}  медиана {np.median(d)*1000:4.0f} мс   90% {np.percentile(d,90)*1000:4.0f} мс   "
          f"дальше 0.25 с: {int((d>0.25).sum())}")
