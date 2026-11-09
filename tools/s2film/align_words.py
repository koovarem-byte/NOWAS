#!/usr/bin/env python3
"""
Точное выравнивание слов голоса (Whisper) с вшитыми субтитрами исходных частей.
Для каждого «окна» реплики задан якорь: первое слово и время его появления в субтитрах (прочитано по кадрам).
Дальше каждая фраза (слова без паузы > 0.2 с) сдвигается в пределах ±0.5 с от предыдущей,
пока её слова не лягут на моменты появления субтитров (30 fps).
Слова-ремарки из сценария, которые озвучка прочитала вслух («Курьер.», «Двойник.», «Бабушка,»), выкидываются.
Результат: s2/work/phrases.json  [{part, file, fs, fe, vt, text, score}]
"""
import json, re
from pathlib import Path
import numpy as np

# (часть, файл, первые слова окна, время в видео, сколько слов брать (None = до следующего окна))
WINDOWS = [
    ("p1", "11-05_211147_Prince_Nur", "8 утра", 3.9),
    ("p1", "11-05_211147_Prince_Nur", "Он уже у двери", 33.8),
    ("p1", "11-05_211147_Prince_Nur", "Он уже у двери", 0.2, "Откуда он знает"),   # копия реплики в начало (как в оригинале)
    ("p2", "11-05_211339_Prince_Nur", "Десять минут", 3.6),
    ("p3_tiktok", "11-05_211411_Prince_Nur", "Лифт открывается", 0.2),
    ("p4", "11-08_202341_Babushka_Valya", "мальчики", 0.1),
    ("p4", "11-08_201348_Prince_Nur", "Отдай заказ", 12.3),
    ("p4", "11-08_201348_Prince_Nur", "12", 20.7),
    ("p4", "11-08_201213_Prince_Nur", "Он стоит", 4.6),
    ("p4", "11-08_201213_Prince_Nur", "Я достаю", 14.9),
    ("p4", "11-08_201213_Prince_Nur", "Мимо", 22.0),
    ("p5", "11-08_211353_Prince_Nur", "Опоздал", 0.1),
    ("p5", "11-08_211353_Prince_Nur", "Это просто", 19.1),
    ("p5", "11-08_211258_Prince_Nur", "Лифт ползёт", 3.4),
    ("p5", "11-08_211258_Prince_Nur", "И тут открывается", 21.0),
    ("p5", "11-08_210700_Babushka_Valya", "Сынок", 23.8),
    ("p5", "11-08_211258_Prince_Nur", "Я всё понял", 29.6),
]
# Части 1–3: субтитры идут фразами по 2–3 слова, автоматический поиск там ненадёжен.
# Время появления субтитра каждой фразы прочитано по кадрам (шаг 0.1 с), по порядку фраз окна.
ANCHORS = {
    0: [3.9, 5.0, 6.7, 7.7, 9.6, 11.5, 12.9, 14.75, 15.8, 16.9, 18.65, 20.35, 22.15, 24.65, 26.1, 27.35, 28.8, 32.0],
    1: [33.75, 34.95, 37.25, 38.4],
    2: [0.15, 1.35],
    3: [3.6, 4.8, 6.05, 7.2, 8.0, 11.75, 13.9, 15.55, 16.35, 17.45, 18.75, 20.75, 24.0, 26.25, 27.7, 29.05, 35.15, 36.75],
    4: [0.15, 1.6, 3.5, 4.7, 5.9, 7.1, 8.2, 10.5, 15.1, 16.2, 21.1, 22.6, 25.5, 32.9],
    5: [0.1, 1.2],
    6: [12.3],
    7: [20.7],
    8: [4.6, 6.5, 7.5],
    9: [14.9, 16.15, 17.6],
    10: [21.95, 22.75, 24.9, 28.3, 29.75, 30.8, 32.5, 34.6],
    11: [0.1, 1.2],
    12: [19.1],
    13: [3.4, 4.9, 6.0, 7.0, 8.3, 10.0, 11.8, 13.2, 14.8, 16.1, 17.6],
    14: [21.0],
    15: [23.8],
    16: [29.6, 32.4, 33.4, 34.6, 37.2, 41.2, 43.7, 45.0, 47.4, 49.8, 50.9, 52.9, 54.6],
}
# скорость голоса внутри фразы (в оригинальном монтаже голос ускорен), измерено по якорям
SPEED = {0: 1.058, 1: 1.058, 2: 1.058, 3: 1.05, 4: 1.0, **{k: 1.0 for k in range(5, 17)}}   # 4–5: измерено внутри длинных фраз
# слова, которые в оригинальном монтаже вырезаны из голоса (в субтитрах их нет)
# принудительное деление фразы перед этими словами (в оригинале здесь пауза ужата сильнее)
SPLIT = {3: [("я", "его", "уже")]}
EXCLUDE = {3: {"нижнее", "подчеркивание"}}
TAGS = {"курьер", "двойник", "бабушка"}   # ремарка = это слово стоит одно перед паузой в начале реплики


def norm(w):
    return re.sub(r"[^\wё]", "", w.lower())


_env = {}
def snap(f, s, e):
    """Подтягивает начало/конец фразы к реальной речи по громкости (Whisper часто захватывает паузу)."""
    if f not in _env:
        import subprocess
        x = np.frombuffer(subprocess.run(["ffmpeg", "-v", "error", "-i", f"voice/{f}.mp3", "-ac", "1", "-ar", "16000",
                                          "-f", "f32le", "-"], capture_output=True).stdout, np.float32)
        n = len(x) // 160
        db = 20 * np.log10(np.sqrt((x[:n * 160].reshape(n, 160) ** 2).mean(1)) + 1e-6)
        _env[f] = (db > -38, db > -32)
    on, on_end = _env[f]
    a, b = min(int(s * 100), len(on) - 1), min(int(e * 100), len(on) - 1)
    # начало: речь уже идёт — отступаем назад не больше 0.1 с; тишина — ищем речь вперёд не дальше 0.35 с
    if on[a]:
        i = a
        while i > 0 and on[i - 1] and a - i < 10:
            i -= 1
    else:
        i = next((k for k in range(a, min(len(on), a + 35)) if on[k]), a)
    # конец: речь продолжается — вперёд не больше 0.1 с; тишина — назад к последней речи не дальше 0.35 с
    if on_end[b]:
        j = b
        while j + 1 < len(on_end) and on_end[j + 1] and j - b < 10:
            j += 1
    else:
        j = next((k for k in range(b, max(0, b - 35), -1) if on_end[k]), b)
    return i / 100, (j + 1) / 100


def phrases_of(words, gap=0.2, split=()):
    out, cur = [], []
    nw = [norm(x["w"]) for x in words]
    for i, w in enumerate(words):
        forced = any(tuple(nw[i:i + len(k)]) == k for k in split)
        if cur and (w["s"] - cur[-1]["e"] > gap or forced):
            out.append(cur); cur = []
        cur.append(w)
    if cur:
        out.append(cur)
    return out


def main():
    res = []
    by_file = {}
    for part, f, *_ in WINDOWS:
        by_file.setdefault(f, json.loads(Path(f"work/asr/{f}.json").read_text()))
    caps = {p: np.array(json.loads(Path(f"work/{p}.capon.json").read_text())["onsets"])
            for p in {w[0] for w in WINDOWS}}
    for wi, wdef in enumerate(WINDOWS):
        part, f, first, t0 = wdef[:4]
        stop = wdef[4] if len(wdef) > 4 else None
        words = by_file[f]
        key = [norm(x) for x in first.split()]
        nw = [norm(w["w"]) for w in words]
        start = next(i for i in range(len(words)) if nw[i:i + len(key)] == key)
        # конец окна — начало следующего окна этого же файла (если оно позже в файле)
        later = [next(i for i in range(len(words)) if nw[i:i + len(k2)] == k2)
                 for (p2, f2, fi2, *_r) in WINDOWS if f2 == f
                 for k2 in [[norm(x) for x in fi2.split()]]]
        later = [i for i in later if i > start]
        end = min(later) if later else len(words)
        if stop:
            ks = [norm(x) for x in stop.split()]
            end = next(i for i in range(start, len(words)) if nw[i:i + len(ks)] == ks)
        seg = words[start:end]
        # выкидываем ремарки: тег-слово, после которого пауза, внутри окна (не первое слово окна)
        clean = []
        for i, w in enumerate(seg):
            nxt = seg[i + 1] if i + 1 < len(seg) else None
            if norm(w["w"]) in TAGS and w["w"].strip().endswith((".", ",")) and i > 0 \
                    and (nxt is None or nxt["s"] - w["e"] > 0.15) and part == "p4":
                continue
            if norm(w["w"]) in EXCLUDE.get(wi, set()):
                continue
            clean.append(w)
        cap = caps[part]
        phs = phrases_of(clean, split=SPLIT.get(wi, ()))
        if wi in ANCHORS:
            assert len(ANCHORS[wi]) == len(phs), (wi, len(ANCHORS[wi]), [p[0]["w"] for p in phs])
            fs_ = [ph[0]["s"] for ph in phs]
            sp = SPEED.get(wi)
            if sp is None:
                sp = float(np.clip(1 / np.polyfit(fs_, ANCHORS[wi], 1)[0], 0.97, 1.10)) if len(phs) >= 4 else 1.05
            for ph, t in zip(phs, ANCHORS[wi]):
                s0, e0 = snap(f, ph[0]["s"], ph[-1]["e"])
                res.append({"part": part, "file": f, "fs": round(s0 - 0.04, 3), "fe": round(e0 + 0.06, 3),
                            "vt": round(t - 0.04, 3), "speed": round(sp, 3), "text": " ".join(w["w"] for w in ph),
                            "score": 1.0, "window": wi, "how": "якорь"})
            continue
        off = t0 - clean[0]["s"]
        rows = []
        for ph in phs:
            st = np.array([w["s"] for w in ph])
            rng = np.arange(-0.5, 0.501, 0.01) + off
            sc = [np.mean(np.exp(-(np.abs(cap[:, None] - (st + o)[None, :]).min(0)) ** 2 / (2 * 0.06 ** 2))) for o in rng]
            if max(sc) > 0.35:
                off = float(rng[int(np.argmax(sc))])
            rows.append({"part": part, "file": f, "fs": round(ph[0]["s"] - 0.04, 3), "fe": round(ph[-1]["e"] + 0.06, 3),
                         "vt": round(ph[0]["s"] - 0.04 + off, 3), "text": " ".join(w["w"] for w in ph),
                         "score": round(float(max(sc)), 2), "window": wi, "how": "поиск"})
        # скорость окна: наклон время-в-видео от времени-в-файле (если фраз достаточно)
        sp = 1.0
        if len(rows) >= 4:
            a = np.polyfit([r["fs"] for r in rows], [r["vt"] for r in rows], 1)[0]
            sp = float(np.clip(1 / a, 0.95, 1.12))
        for r in rows:
            r["speed"] = round(sp, 3)
        res += rows
    Path("work/phrases.json").write_text(json.dumps(res, ensure_ascii=False, indent=1))
    for r in res:
        flag = "  ?" if r["score"] < 0.35 else ""
        print(f"{r['part']:9s} {r['vt']:6.2f}  x{r['speed']:.3f} {r['how']:5s} [{r['score']:.2f}] {r['text']}{flag}")


if __name__ == "__main__":
    main()
