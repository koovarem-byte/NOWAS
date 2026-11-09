#!/usr/bin/env python3
"""Сквозная проверка готового фильма: субтитры исходников, перенесённые по монтажному листу в шкалу фильма,
против начал слов, распознанных Whisper в РЕАЛЬНОМ звуке фильма."""
import json, sys
from pathlib import Path
import numpy as np
edl = json.loads(Path("edl.json").read_text())
words = json.loads(Path(sys.argv[1]).read_text())
ws = np.array([w["s"] for w in words])
t = 0.0; rows = []
for s in edl["segments"]:
    n = int(round((s["out"] - s["in"]) * 30)) / 30
    if not s.get("vis") and s.get("freeze") is None:
        cap = json.loads(Path(f"work/{s['part']}.capon.json").read_text())["onsets"]
        for c in cap:
            if s["in"] <= c < s["out"]:
                rows.append((s["part"], t + c - s["in"]))
    t += n
parts = {}
for part, c in rows:
    parts.setdefault(part, []).append(np.abs(ws - c).min())
allv = []
for p, d in parts.items():
    d = np.array(d); allv += list(d)
    print(f"{p:10s} субтитров {len(d):3d}  медиана {np.median(d)*1000:4.0f} мс   90% {np.percentile(d,90)*1000:4.0f} мс   >0.25 с: {int((d>0.25).sum())}")
allv = np.array(allv)
print(f"ВСЕГО      субтитров {len(allv):3d}  медиана {np.median(allv)*1000:4.0f} мс   90% {np.percentile(allv,90)*1000:4.0f} мс")
