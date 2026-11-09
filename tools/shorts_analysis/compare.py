#!/usr/bin/env python3
"""Сравнение: медиана референсов против наших роликов по всем метрикам.
  python3 -I tools/shorts_analysis/compare.py shorts/ref shorts/our > shorts/compare.md
"""
import json, sys
from collections import Counter
from pathlib import Path

import numpy as np


def load(d):
    out = []
    for j in sorted((Path(d) / "analysis").glob("*.json")):
        if j.name.count(".") > 1:
            continue
        r = json.loads(j.read_text())
        t = j.with_name(j.stem + ".transitions.json")
        c = j.with_name(j.stem + ".captions.json")
        r["tr"] = json.loads(t.read_text()) if t.exists() else {}
        r["cap"] = json.loads(c.read_text()) if c.exists() else {}
        out.append(r)
    return out


def metrics(r):
    S = r["shots"]; w = np.array([s["frames"] for s in S], float); w /= w.sum()
    avg = lambda k: float(sum(s[k] * x for s, x in zip(S, w)))
    pf = r["per_frame"]
    tx, ty, zm, ob = (np.array(pf[k]) for k in ("cam_tx", "cam_ty", "cam_zoom", "obj"))
    sp = np.hypot(tx, ty) * 30
    moving = (sp > 0.03) | (np.abs(zm) * 30 > 0.02)
    # самая длинная полностью статичная пауза (камера стоит И в кадре ничего не движется)
    still = (~moving) & (ob < 0.004)
    run = best = 0
    for v in still:
        run = run + 1 if v else 0; best = max(best, run)
    tr, cap = r["tr"], r["cap"]
    return {
        "Длительность, с": r["duration"],
        "Смен картинки в секунду (все виды)": tr.get("changes_per_sec"),
        "Держим одну картинку в среднем, с": tr.get("avg_hold_sec"),
        "Самая долгая картинка, с": tr.get("max_hold_sec"),
        "Первая смена картинки, с": tr.get("first_change_at"),
        "Жёстких склеек в секунду": r["cuts_per_sec"],
        "Время, когда камера движется, %": round(float(moving.mean()) * 100, 1),
        "Скорость камеры (ширин кадра/с)": round(float(sp.mean()), 3),
        "Наезд/отъезд (|зум|/с)": round(float(np.abs(zm).mean() * 30), 4),
        "Движение внутри кадра (персонажи)": round(float(ob.mean()), 4),
        "Самая долгая полная статика, с": round(best / 30, 2),
        "Камера+действие в первые 3 с (скорость)": r["hook_first3s"]["camera_speed"],
        "Контраст": round(avg("contrast"), 3),
        "Доля теней в кадре": round(avg("shadow"), 3),
        "Жёсткость теней": round(avg("shadow_hard"), 3),
        "Виньетка (центр ярче краёв)": round(avg("vignette"), 3),
        "Насыщенность": round(avg("sat"), 3),
        "Яркость": round(avg("luma"), 3),
        "Тёплые тона": round(avg("warm"), 3),
        "Холодные тона": round(avg("cool"), 3),
        "Громкость, LUFS": (r.get("audio") or {}).get("integrated_lufs"),
        "Звуковых акцентов в секунду": (r.get("audio") or {}).get("accents_per_sec"),
        "Петля (последний кадр ~ первый)": r["loop_similarity_last_first"],
    }


def main():
    ref, our = load(sys.argv[1]), load(sys.argv[2])
    R = [metrics(r) for r in ref]; O = [metrics(r) for r in our]
    names = [Path(r["file"]).stem for r in our]
    print(f"# Референсы ({len(R)} шт.) против наших\n")
    print("| Метрика | Референсы: медиана (мин–макс) | " + " | ".join(names) + " |")
    print("|---|---|" + "---|" * len(O))
    for k in R[0]:
        v = [x[k] for x in R if x[k] is not None]
        ref_s = f"{np.median(v):.3g} ({min(v):.3g}–{max(v):.3g})" if v else "—"
        print(f"| {k} | {ref_s} | " + " | ".join("—" if o[k] is None else f"{o[k]:.3g}" for o in O) + " |")
    print("\n## Движение камеры по планам\n")
    for label, grp in (("Референсы", ref), ("Наши", our)):
        c = Counter(m.split(" x")[0] for r in grp for s in r["shots"] for m in s["camera"].split(", "))
        tot = sum(len(r["shots"]) for r in grp)
        print(f"- **{label}**: " + ", ".join(f"{m} {n * 100 // tot}%" for m, n in c.most_common(8)))
    print("\n## Ключевой свет по планам\n")
    for label, grp in (("Референсы", ref), ("Наши", our)):
        c = Counter(s["key_light"] for r in grp for s in r["shots"])
        tot = sum(c.values())
        print(f"- **{label}**: " + ", ".join(f"{m} {n * 100 // tot}%" for m, n in c.most_common()))


if __name__ == "__main__":
    main()
