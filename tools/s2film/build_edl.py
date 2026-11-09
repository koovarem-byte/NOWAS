#!/usr/bin/env python3
"""
Монтажный лист фильма «Курьер. Сезон 2 — целиком».  -> s2/edl.json

Решения (почему так — см. s2/FILM_PLAN.md):
  - вырезаны все «пиши в комменты»/варианты/карточки в конце частей, повторный пролог части 2,
    реплика двойника с заголовком в начале части 5;
  - тишина между фразами > 0.5 с ужимается до 0.36 с;
  - длинные планы (> 2.4 с) режутся на «панч»-подпланы на началах фраз, чтобы картинка менялась ~раз в 2 с;
  - заголовки в первых секундах частей 1 и 3 убираются кропом, в части 4 — заменой кадра + свой субтитр.
"""
import json
from pathlib import Path

ROOT = Path("s2") if Path("s2/edl.json").parent.exists() and Path("s2/src").exists() else Path(".")
PARTS = {"p1": "src/p1.mp4", "p2": "src/p2.mp4", "p3_tiktok": "src/p3_tiktok.mp4", "p4": "src/p4.mp4", "p5": "src/p5.mp4"}
LOOK = {"p1": [1.14, 1.06, 0.7], "p2": [1.14, 1.06, 0.7], "p3_tiktok": [1.14, 1.08, 0.7],
        "p4": [1.20, 1.10, 1.0], "p5": [1.20, 1.10, 1.0]}

# куски исходников, которые идут в фильм (время исходного видео)
RANGES = [
    dict(part="p1", a=0.00, b=38.30, top=[[0.0, 3.85, 0.27]]),           # до «Откуда он знает?»
    dict(part="p2", a=3.30, b=34.95),                                      # без пролога и «пиши в комменты»
    dict(part="p3_tiktok", a=0.00, b=22.55, top=[[0.0, 2.95, 0.26]]),     # до «У меня секунд десять»
    dict(part="p4", a=0.00, b=3.20, vis=["p4", 9.30],                      # «Мальчики, а кто из вас настоящий?»
         caption=[["МАЛЬЧИКИ,", 0.06, 1.12, [255, 140, 200]], ["А КТО ИЗ ВАС", 1.16, 2.25, [255, 255, 255]],
                  ["НАСТОЯЩИЙ?", 2.25, 3.20, [255, 215, 0]]]),
    dict(part="p4", a=3.30, b=8.60),
    dict(part="p4", a=12.10, b=36.95),                                     # до «…это моя квартира»
    dict(part="p5", a=3.20, b=56.20),                                      # с лифта до «…тебя уже ждут»
    dict(part="p5", a=57.0, b=58.6, freeze=57.0, zoom=[1.0, 1.08]),      # тизер «Сезон 3» (кадр, где весь заголовок уже проявился)
]
# ударные места (рывок зума + тряска + удар в звуке), время исходника
HITS = {
    "p1": [10.8, 29.5, 33.75, 37.25],
    "p2": [15.55, 25.2, 32.7],
    "p3_tiktok": [5.9, 16.2, 21.9],
    "p4": [0.10, 20.7, 21.95, 25.1, 34.6],
    "p5": [7.0, 19.1, 33.4, 35.2, 48.7, 55.6, 57.0],
}
HEART = {"p3_tiktok": [16.3, 17.9, 19.5, 21.1], "p4": [14.9, 16.4, 17.9, 19.4], "p5": [3.4, 4.9, 6.4, 13.2, 14.7, 16.2]}
MAX_SHOT = 2.4      # дольше — режем на подпланы
GAP_MAX, GAP_KEEP = 0.5, 0.18


def main():
    ph = json.loads((ROOT / "work/phrases.json").read_text())
    cuts = {}
    for p in PARTS:
        t = ROOT / "src/analysis" / f"{p}.transitions.json"
        cuts[p] = [e["t"] for e in json.loads(t.read_text())["events"]] if t.exists() else []
    segs = []
    for r in RANGES:
        part = r["part"]
        P = sorted([(x["vt"], x["vt"] + (x["fe"] - x["fs"]) / x["speed"]) for x in ph if x["part"] == part])
        # 1) ужимаем тишину внутри куска
        pieces = [[r["a"], r["b"]]]
        if not r.get("vis") and not r.get("freeze"):
            pieces = []
            cur = r["a"]
            for (s1, e1), (s2, e2) in zip(P, P[1:]):
                if r["a"] < e1 and s2 < r["b"] and s2 - e1 > GAP_MAX:
                    pieces.append([cur, e1 + GAP_KEEP]); cur = s2 - GAP_KEEP
            pieces.append([cur, r["b"]])
        # 2) режем длинные планы на подпланы по началам фраз
        for a, b in pieces:
            if r.get("freeze") or r.get("vis"):
                bounds = [a, b]
            else:
                bounds = [a] + [c for c in cuts[part] if a < c < b] + [b]
            sub = [bounds[0]]
            for x, y in zip(bounds, bounds[1:]):
                starts = [] if (r.get("freeze") or r.get("vis")) else [s for s, _ in P if x + 1.2 < s < y - 1.0]
                t = x
                for s in starts:
                    if s - t >= MAX_SHOT * 0.7 and y - s >= 1.0:
                        sub.append(s - 0.03); t = s
                sub.append(y)
            for x, y in zip(sub, sub[1:]):
                if y - x < 0.2:
                    continue
                d = dict(part=part, **{"in": round(x, 3), "out": round(y, 3)})
                d["hits"] = [h for h in HITS.get(part, []) if x <= h < y]
                d["heart"] = [h for h in HEART.get(part, []) if x <= h < y]
                for t0, t1, top in r.get("top", []):
                    if x < t1 - 0.1:
                        d["top"] = top
                for k in ("vis", "caption", "freeze", "zoom"):
                    if k in r:
                        d[k] = r[k]
                segs.append(d)
    # чередуем подачу камеры: наезд с разных стартов, иногда отъезд, дрейф в разные стороны
    # от 0.93 (лёгкий отъезд с размытыми краями — субтитры целы) до 1.12 (наезд, если субтитр позволяет)
    pat = [[0.93, 1.03], [0.99, 1.09], [0.95, 1.06], [1.07, 0.96]]
    for i, s in enumerate(segs):
        s.setdefault("zoom", pat[i % 4])
        s["drift"] = 0.02 * (1 if i % 2 else -1)
        s["focus"] = [0.5, 0.5]
    edl = {"parts": PARTS, "looks": LOOK, "segments": segs}
    (ROOT / "edl.json").write_text(json.dumps(edl, ensure_ascii=False, indent=1))
    tot = sum(s["out"] - s["in"] for s in segs)
    print(f"сегментов {len(segs)}, длительность {tot:.1f} с ({int(tot // 60)}:{tot % 60:04.1f})")
    hold = [s["out"] - s["in"] for s in segs]
    print(f"средний план {tot / len(segs):.2f} с, самый длинный {max(hold):.2f} с")


if __name__ == "__main__":
    main()
