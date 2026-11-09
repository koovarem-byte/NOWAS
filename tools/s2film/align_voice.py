#!/usr/bin/env python3
"""
Привязка чистых голосов ElevenLabs к исходным частям сезона 2.
Голос режется на фразы по паузам; каждая фраза двигается в своём окне,
пока её начала слов не совпадут с моментами появления вшитых субтитров.
Результат: s2/work/align.json — для каждой фразы: файл, начало/конец в файле, время в исходном видео.
"""
import json, subprocess
from pathlib import Path
import numpy as np

V = Path("voice")
# порядок реплик в каждой части и примерное время в исходном видео (прочитано по субтитрам)
PLAN = {
    "p1": [("11-05_211147_Prince_Nur", [0.0])],
    "p2": [("11-05_211339_Prince_Nur", [0.2])],
    "p3_tiktok": [("11-05_211411_Prince_Nur", [0.1])],
    "p4": [("11-08_202341_Babushka_Valya", [0.1]),
           ("11-08_201348_Prince_Nur", [12.3, 20.7]),
           ("11-08_201213_Prince_Nur", [4.6, 14.9, 22.0])],
    "p5": [("11-08_211353_Prince_Nur", [0.1, 19.1]),
           ("11-08_210700_Babushka_Valya", [23.8]),
           ("11-08_211258_Prince_Nur", [3.3, 21.0, 29.6])],
}
# где в файле кончается одна реплика и начинается следующая (секунды файла), проверено по тексту
CUTS = {"11-08_201348_Prince_Nur": [3.3], "11-08_201213_Prince_Nur": [5.0, 11.5],
        "11-08_211353_Prince_Nur": [2.1], "11-08_211258_Prince_Nur": [15.1, 17.7]}
SR = 16000


def load(p):
    raw = subprocess.run(["ffmpeg", "-v", "error", "-i", str(p), "-ac", "1", "-ar", str(SR), "-f", "f32le", "-"],
                         capture_output=True).stdout
    return np.frombuffer(raw, np.float32)


def env_db(x, hop=160):  # 10 мс
    n = len(x) // hop
    return 20 * np.log10(np.sqrt((x[:n * hop].reshape(n, hop) ** 2).mean(1)) + 1e-6)


def phrases(db, thr=-40, min_gap=0.22):
    """фразы = речь, разделённая паузами >= min_gap"""
    sp = db > thr
    segs, i, n = [], 0, len(sp)
    while i < n:
        if sp[i]:
            j = i
            while j < n:
                k = j
                while k < n and not sp[k]:
                    k += 1
                if k - j >= min_gap * 100 or k >= n:
                    break
                j = k + 1 if k < n else n
                while j < n and sp[j]:
                    j += 1
            segs.append([i / 100, j / 100]); i = j
        else:
            i += 1
    return segs


def word_onsets(db):
    d = np.diff(db, prepend=db[0])
    on = [i / 100 for i in range(3, len(db)) if db[i] > -38 and db[i - 3] < db[i] - 12]
    out = []
    for t in on:
        if not out or t - out[-1] > 0.15:
            out.append(t)
    return np.array(out)


def score(cap, won, off, sigma=0.08):
    if not len(won):
        return 0
    t = won + off
    d = np.abs(cap[:, None] - t[None, :]).min(0)
    return float(np.exp(-d ** 2 / (2 * sigma ** 2)).sum() / len(won))


def main():
    res = []
    for part, files in PLAN.items():
        cap = np.array(json.loads(Path(f"work/{part}.capon.json").read_text())["onsets"])
        for name, starts in files:
            db = env_db(load(V / f"{name}.mp3"))
            ph = phrases(db)
            won_all = word_onsets(db)
            # раскладываем фразы по окнам: новое окно начинается на самой длинной паузе
            cuts = CUTS.get(name, [])
            bounds = [-1] + cuts + [1e9]
            groups = [[x for x in ph if a <= x[0] < b] for a, b in zip(bounds[:-1], bounds[1:])]
            for g, t0 in zip(groups, starts):
                won = won_all[(won_all >= g[0][0] - 0.02) & (won_all < g[-1][1])]
                # 1) глобально: видео = a*(файл - начало) + t, a — растяжение, t — сдвиг
                best = (-1, 1.0, t0)
                for a in np.arange(0.92, 1.12, 0.004):
                    for t in np.arange(t0 - 1.0, t0 + 1.0, 0.02):
                        sc = score(cap, (won - g[0][0]) * a + t, 0)
                        if sc > best[0]:
                            best = (sc, a, t)
                gsc, A, T = best
                # 2) каждая фраза: подстройка только в пределах ±0.25 с
                for s_, e_ in g:
                    w = won_all[(won_all >= s_ - 0.02) & (won_all < e_)]
                    base = (s_ - g[0][0]) * A + T
                    rng = np.arange(-0.25, 0.26, 0.01)
                    sc = [score(cap, (w - s_) + base + o, 0) for o in rng]
                    o = float(rng[int(np.argmax(sc))]) if len(w) and max(sc) > 0.5 else 0.0
                    res.append({"part": part, "file": name, "fs": round(s_, 2), "fe": round(e_, 2),
                                "vt": round(base + o, 2), "score": round(max(sc) if len(w) else 0, 2),
                                "words": len(w), "stretch": round(float(A), 3), "group_score": round(gsc, 2)})
    Path("work/align.json").write_text(json.dumps(res, ensure_ascii=False, indent=1))
    for r in res:
        print(f"{r['part']:9s} {r['file'][:20]:20s} файл {r['fs']:6.2f}-{r['fe']:6.2f}  -> видео {r['vt']:6.2f}  совпадение {r['score']:.2f} ({r['words']} сл.)  растяж {r['stretch']} группа {r['group_score']}")


if __name__ == "__main__":
    main()
