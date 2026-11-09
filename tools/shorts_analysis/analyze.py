#!/usr/bin/env python3
"""
Покадровый разбор шортсов: КАЖДЫЙ кадр на 30 fps, без выборочных кадров.

Что считается на каждом кадре:
  - склейки (смена плана): HSV-гистограмма + средняя разница кадров
  - движение камеры: оптический поток -> аффинная модель (пан X, тилт Y, зум, крен)
  - тряска «с рук»: дрожание сдвига между соседними кадрами
  - движение внутри кадра (персонажи/объекты) = поток минус движение камеры
  - свет и тени: яркость, контраст, доля теней/пересветов, сторона ключевого света,
    виньетка, жёсткость теней (резкость перехода на границе тень/свет)
  - цвет: насыщенность, тёплое/холодное, доминирующие цвета плана
  - звук: громкость каждые 100 мс (EBU R128), тишина, всплески (удары/акценты)
  - петля: насколько последний кадр похож на первый (бесшовный повтор шортса)

Использование:
  python3 -I tools/shorts_analysis/analyze.py shorts/            # все mp4 в папке
  python3 -I tools/shorts_analysis/analyze.py shorts/01_xxx.mp4  # один ролик
Результат: shorts/analysis/<имя>.json, <имя>_sheet.jpg (лист планов), summary.md
"""
import json, os, re, subprocess, sys
from pathlib import Path

import cv2
import numpy as np

FPS = 30
FLOW_W = 240          # ширина кадра для оптического потока
CUT_HIST = 0.55       # корреляция гистограмм ниже -> склейка
MIN_SHOT = 4          # кадров минимум между склейками


# ───────────────────────── кадры ─────────────────────────
def probe(path):
    out = subprocess.run(["ffprobe", "-v", "error", "-print_format", "json",
                          "-show_streams", "-show_format", str(path)],
                         capture_output=True, text=True, check=True).stdout
    j = json.loads(out)
    v = next(s for s in j["streams"] if s["codec_type"] == "video")
    num, den = map(int, v["r_frame_rate"].split("/"))
    return {"w": int(v["width"]), "h": int(v["height"]),
            "src_fps": round(num / den, 3),
            "duration": float(j["format"]["duration"]),
            "has_audio": any(s["codec_type"] == "audio" for s in j["streams"])}


def frames_30fps(path, w, h):
    """Декодирует ролик ровно в 30 кадров/с (ffmpeg fps-фильтр), отдаёт BGR кадры."""
    tw = 540
    th = int(round(h * tw / w / 2) * 2)
    p = subprocess.Popen(["ffmpeg", "-v", "error", "-i", str(path),
                          "-vf", f"fps={FPS},scale={tw}:{th}",
                          "-f", "rawvideo", "-pix_fmt", "bgr24", "-"],
                         stdout=subprocess.PIPE)
    size = tw * th * 3
    while True:
        buf = p.stdout.read(size)
        if len(buf) < size:
            break
        yield np.frombuffer(buf, np.uint8).reshape(th, tw, 3)
    p.wait()


# ───────────────────────── метрики кадра ─────────────────────────
def light_metrics(bgr):
    g = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    H, W = g.shape
    left, right = g[:, :W // 2].mean(), g[:, W // 2:].mean()
    top, bottom = g[:H // 2].mean(), g[H // 2:].mean()
    cy, cx = H // 2, W // 2
    center = g[cy - H // 6:cy + H // 6, cx - W // 6:cx + W // 6].mean()
    corners = np.mean([g[:H // 8, :W // 8].mean(), g[:H // 8, -W // 8:].mean(),
                       g[-H // 8:, :W // 8].mean(), g[-H // 8:, -W // 8:].mean()])
    shadow = g < 0.15
    # жёсткость теней: средний градиент на границе тень/свет
    edge = cv2.morphologyEx(shadow.astype(np.uint8), cv2.MORPH_GRADIENT, np.ones((3, 3), np.uint8)) > 0
    grad = cv2.magnitude(cv2.Sobel(g, cv2.CV_32F, 1, 0), cv2.Sobel(g, cv2.CV_32F, 0, 1))
    hard = float(grad[edge].mean()) if edge.any() else 0.0
    hue = hsv[..., 0].astype(np.int32)
    sat = hsv[..., 1].astype(np.float32) / 255
    colored = sat > 0.25
    warm = ((hue < 25) | (hue > 160)) & colored
    cool = ((hue > 85) & (hue < 135)) & colored
    sharp = cv2.Laplacian(cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY), cv2.CV_64F).var()
    return {
        "luma": float(g.mean()), "contrast": float(g.std()),
        "p5": float(np.percentile(g, 5)), "p95": float(np.percentile(g, 95)),
        "shadow": float(shadow.mean()), "highlight": float((g > 0.9).mean()),
        "key_lr": float(right - left), "key_tb": float(top - bottom),
        "vignette": float(center - corners), "shadow_hard": hard,
        "sat": float(sat.mean()), "warm": float(warm.mean()), "cool": float(cool.mean()),
        "sharp": float(sharp),
    }


def hist(bgr):
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    h = cv2.calcHist([hsv], [0, 1, 2], None, [16, 8, 8], [0, 180, 0, 256, 0, 256])
    return cv2.normalize(h, h).flatten()


_grid = None
def camera_motion(prev_g, g):
    """Глобальное движение камеры + движение внутри кадра."""
    global _grid
    flow = cv2.calcOpticalFlowFarneback(prev_g, g, None, 0.5, 3, 15, 3, 5, 1.2, 0)
    H, W = g.shape
    if _grid is None or _grid[0].shape != (H // 8, W // 8):
        ys, xs = np.mgrid[4:H:8, 4:W:8]
        _grid = (xs[:H // 8, :W // 8].astype(np.float32), ys[:H // 8, :W // 8].astype(np.float32))
    xs, ys = _grid
    f = flow[ys.astype(int), xs.astype(int)]
    src = np.stack([xs.ravel(), ys.ravel()], 1)
    dst = src + f.reshape(-1, 2)
    M, inl = cv2.estimateAffinePartial2D(src, dst, method=cv2.RANSAC, ransacReprojThreshold=1.0)
    if M is None:
        return dict(tx=0, ty=0, zoom=0, rot=0, inliers=0, obj=float(np.linalg.norm(f, axis=2).mean()))
    scale = float(np.hypot(M[0, 0], M[1, 0]))
    rot = float(np.degrees(np.arctan2(M[1, 0], M[0, 0])))
    # перевод сдвига к центру кадра, в долях ширины кадра
    c = np.array([W / 2, H / 2])
    cen = M[:, :2] @ c + M[:, 2] - c
    pred = src @ M[:, :2].T + M[:, 2]
    resid = np.linalg.norm(dst - pred, axis=1)
    return dict(tx=float(cen[0] / W), ty=float(cen[1] / W), zoom=scale - 1, rot=rot,
                inliers=float(inl.mean()), obj=float(np.percentile(resid, 90) / W))


# ───────────────────────── звук ─────────────────────────
def audio_track(path, dur):
    out = subprocess.run(["ffmpeg", "-nostats", "-v", "verbose", "-i", str(path), "-af", "ebur128=framelog=verbose",
                          "-f", "null", "-"], capture_output=True, text=True).stderr
    t, M = [], []
    for m in re.finditer(r"t:\s*([\d.]+)\s+TARGET.*?M:\s*(-?[\d.]+|-inf)", out):
        t.append(float(m.group(1)))
        M.append(-70.0 if "inf" in m.group(2) else float(m.group(2)))
    summ = out[out.rfind("Summary:"):] if "Summary:" in out else out
    I = re.search(r"I:\s+(-?[\d.]+) LUFS", summ)
    LRA = re.search(r"LRA:\s+([\d.]+) LU", summ)
    M = np.array(M)
    if not len(M):
        return {"integrated_lufs": None, "silence_ratio": None, "accents": [], "loud": []}
    # акценты: скачок громкости > 6 LU за 300 мс
    acc = [round(t[i], 2) for i in range(3, len(M)) if M[i] - M[i - 3] > 6 and M[i] > -35]
    acc = [a for k, a in enumerate(acc) if k == 0 or a - acc[k - 1] > 0.4]
    return {"integrated_lufs": float(I.group(1)) if I else None,
            "lra": float(LRA.group(1)) if LRA else None,
            "silence_ratio": float((M < -45).mean()),
            "accents": acc, "accents_per_sec": round(len(acc) / max(dur, 1e-6), 2),
            "loud": [[round(a, 1), round(b, 1)] for a, b in zip(t[::5], M[::5])]}


# ───────────────────────── текст (субтитры) ─────────────────────────
def transcript(path):
    vtts = sorted(Path(path).parent.glob(Path(path).stem + "*.vtt"))
    if not vtts:
        return None
    cues, seen = [], set()
    for block in vtts[0].read_text(errors="ignore").split("\n\n"):
        m = re.search(r"(\d+):(\d+):([\d.]+) --> ", block)
        if not m:
            continue
        t = int(m.group(1)) * 3600 + int(m.group(2)) * 60 + float(m.group(3))
        for line in block.splitlines()[1:]:
            line = re.sub(r"<[^>]+>", "", line).strip()
            if line and "-->" not in line and line not in seen:
                seen.add(line); cues.append((round(t, 2), line))
    words = sum(len(c[1].split()) for c in cues)
    return {"text": " ".join(c[1] for c in cues),
            "first_3s": " ".join(c[1] for c in cues if c[0] < 3),
            "first_speech_at": cues[0][0] if cues else None,
            "words": words, "cues": cues}


# ───────────────────────── классификация движения ─────────────────────────
def classify_shot(cam):
    """cam: list of per-frame dicts within shot -> описание движения камеры."""
    if len(cam) < 2:
        return {"move": "flash", "detail": {}}
    tx = np.array([c["tx"] for c in cam]); ty = np.array([c["ty"] for c in cam])
    zm = np.array([c["zoom"] for c in cam]); rt = np.array([c["rot"] for c in cam])
    n = len(cam)
    tot = {"pan": tx.sum(), "tilt": ty.sum(), "zoom": zm.sum(), "roll": rt.sum()}
    # дрожание = высокочастотная составляющая сдвига
    jitter = float(np.std(np.diff(tx)) + np.std(np.diff(ty))) if n > 3 else 0.0
    speed = float(np.mean(np.hypot(tx, ty)) * FPS)        # ширин кадра / сек
    zspeed = float(zm.mean() * FPS)                        # доля масштаба / сек
    moves = []
    if abs(tot["zoom"]) > 0.06:
        moves.append("наезд" if tot["zoom"] > 0 else "отъезд")
    if abs(tot["pan"]) > 0.15:
        moves.append("панорама вправо" if tot["pan"] < 0 else "панорама влево")
    if abs(tot["tilt"]) > 0.15:
        moves.append("тилт вниз" if tot["tilt"] < 0 else "тилт вверх")
    if abs(tot["roll"]) > 4:
        moves.append("крен")
    step = float(np.mean(np.hypot(tx, ty)))
    if jitter > 0.003 and jitter > 0.35 * step:   # дрожание, а не плавный ход
        moves.append("тряска/с рук")
    # резкие рывки (whip / удар камеры)
    whips = int((np.hypot(tx, ty) > 0.06).sum())
    if whips:
        moves.append(f"рывок x{whips}")
    if not moves:
        moves.append("статика" if speed < 0.02 and abs(zspeed) < 0.01 else "дрейф")
    return {"move": ", ".join(moves),
            "detail": {k: round(float(v), 4) for k, v in tot.items()} |
                      {"jitter": round(jitter, 5), "speed": round(speed, 4),
                       "zoom_per_s": round(zspeed, 4), "whip_frames": whips}}


def dominant_colors(bgr, k=4):
    px = cv2.resize(bgr, (64, 114)).reshape(-1, 3).astype(np.float32)
    _, lab, cen = cv2.kmeans(px, k, None, (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 1.0),
                             3, cv2.KMEANS_PP_CENTERS)
    cnt = np.bincount(lab.ravel(), minlength=k)
    order = np.argsort(-cnt)
    return [{"hex": "#%02x%02x%02x" % tuple(int(c) for c in cen[i][::-1]),
             "share": round(float(cnt[i] / cnt.sum()), 2)} for i in order]


# ───────────────────────── разбор ролика ─────────────────────────
def analyze(path, outdir):
    meta = probe(path)
    frames_light, cam, hd, md = [], [], [], []
    keyframes = {}
    first = last = prev = prev_g = prev_h = None
    for i, fr in enumerate(frames_30fps(path, meta["w"], meta["h"])):
        g = cv2.cvtColor(cv2.resize(fr, (FLOW_W, int(fr.shape[0] * FLOW_W / fr.shape[1]))), cv2.COLOR_BGR2GRAY)
        h = hist(fr)
        frames_light.append(light_metrics(fr))
        if prev is None:
            first = fr.copy(); cam.append(dict(tx=0, ty=0, zoom=0, rot=0, inliers=1, obj=0))
            hd.append(1.0); md.append(0.0)
        else:
            cam.append(camera_motion(prev_g, g))
            hd.append(float(cv2.compareHist(prev_h, h, cv2.HISTCMP_CORREL)))
            md.append(float(cv2.absdiff(prev, fr).mean() / 255))
        keyframes[i] = fr if i % 3 == 0 else None  # держим каждый 3-й для листа планов
        prev, prev_g, prev_h, last = fr, g, h, fr
    n = len(frames_light)
    md_arr = np.array(md)
    # склейки: падение корреляции гистограмм ИЛИ всплеск разницы относительно соседей
    cuts = [0]
    for i in range(1, n):
        local = np.median(md_arr[max(1, i - 8):i + 8]) + 1e-4
        flow_break = cam[i]["inliers"] < 0.25 and cam[i]["obj"] > 0.02   # поток не сшивает кадры
        is_cut = hd[i] < CUT_HIST or (md_arr[i] > 0.12 and md_arr[i] > 4 * local) or flow_break
        if is_cut and i - cuts[-1] >= MIN_SHOT:
            cuts.append(i)
        cam[i] = cam[i] if not is_cut else dict(tx=0, ty=0, zoom=0, rot=0, inliers=0, obj=0)
    bounds = cuts + [n]
    shots = []
    for a, b in zip(bounds[:-1], bounds[1:]):
        L = frames_light[a:b]
        mid = a + (b - a) // 2
        kf = next((keyframes[j] for j in range(mid, a - 1, -1) if keyframes.get(j) is not None), first)
        c = classify_shot(cam[a + 1:b])
        shots.append({
            "start": round(a / FPS, 3), "end": round(b / FPS, 3), "frames": b - a,
            "camera": c["move"], "camera_detail": c["detail"],
            "subject_motion": round(float(np.mean([x["obj"] for x in cam[a + 1:b]] or [0])), 5),
            "luma": round(float(np.mean([x["luma"] for x in L])), 3),
            "contrast": round(float(np.mean([x["contrast"] for x in L])), 3),
            "shadow": round(float(np.mean([x["shadow"] for x in L])), 3),
            "shadow_hard": round(float(np.mean([x["shadow_hard"] for x in L])), 3),
            "key_light": ("справа" if np.mean([x["key_lr"] for x in L]) > 0.03 else
                          "слева" if np.mean([x["key_lr"] for x in L]) < -0.03 else "фронт/центр"),
            "vignette": round(float(np.mean([x["vignette"] for x in L])), 3),
            "sat": round(float(np.mean([x["sat"] for x in L])), 3),
            "warm": round(float(np.mean([x["warm"] for x in L])), 3),
            "cool": round(float(np.mean([x["cool"] for x in L])), 3),
            "colors": dominant_colors(kf),
        })
        keyframes[f"shot{len(shots)-1}"] = kf
    dur = n / FPS
    audio = audio_track(path, dur) if meta["has_audio"] else {}
    loop_sim = float(cv2.compareHist(hist(first), hist(last), cv2.HISTCMP_CORREL))
    sl = np.array([s["frames"] for s in shots]) / FPS
    camall = cam[1:]
    report = {
        "file": Path(path).name, "meta": meta, "frames_analyzed": n, "fps_analyzed": FPS,
        "duration": round(dur, 2), "shots_count": len(shots),
        "cuts_per_sec": round((len(shots) - 1) / dur, 2),
        "avg_shot_sec": round(float(sl.mean()), 2), "median_shot_sec": round(float(np.median(sl)), 2),
        "hook_first3s": {
            "cuts": sum(1 for s in shots if 0 < s["start"] < 3),
            "camera_speed": round(float(np.mean([np.hypot(c["tx"], c["ty"]) for c in cam[1:90]]) * FPS), 4),
            "subject_motion": round(float(np.mean([c["obj"] for c in cam[1:90]])), 5),
            "zoom_per_s": round(float(np.mean([c["zoom"] for c in cam[1:90]]) * FPS), 4),
        },
        "camera_moving_ratio": round(float(np.mean([np.hypot(c["tx"], c["ty"]) * FPS > 0.03 or abs(c["zoom"]) * FPS > 0.02
                                                    for c in camall])), 3) if camall else 0,
        "loop_similarity_last_first": round(loop_sim, 3),
        "audio": audio, "transcript": transcript(path), "shots": shots,
        "per_frame": {  # компактно: каждое значение на каждый кадр
            "luma": [round(x["luma"], 3) for x in frames_light],
            "shadow": [round(x["shadow"], 3) for x in frames_light],
            "cam_tx": [round(c["tx"], 4) for c in cam], "cam_ty": [round(c["ty"], 4) for c in cam],
            "cam_zoom": [round(c["zoom"], 4) for c in cam], "obj": [round(c["obj"], 4) for c in cam],
            "frame_diff": [round(x, 4) for x in md],
        },
    }
    stem = Path(path).stem
    (outdir / f"{stem}.json").write_text(json.dumps(report, ensure_ascii=False, indent=1))
    contact_sheet([keyframes[f"shot{i}"] for i in range(len(shots))], shots, outdir / f"{stem}_sheet.jpg")
    return report


def contact_sheet(frames, shots, out, cols=6):
    if not frames:
        return
    th = 320
    tw = int(frames[0].shape[1] * th / frames[0].shape[0])
    rows = (len(frames) + cols - 1) // cols
    sheet = np.zeros((rows * (th + 22), cols * tw, 3), np.uint8)
    for i, (f, s) in enumerate(zip(frames, shots)):
        r, c = divmod(i, cols)
        y, x = r * (th + 22), c * tw
        sheet[y:y + th, x:x + tw] = cv2.resize(f, (tw, th))
        cv2.putText(sheet, f"#{i} {s['start']:.1f}s", (x + 4, y + th + 16),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
    cv2.imwrite(str(out), sheet, [cv2.IMWRITE_JPEG_QUALITY, 85])


# ───────────────────────── сводка ─────────────────────────
def summary(reports, outdir):
    from collections import Counter
    def info(r):
        p = Path(r["_src"]).with_suffix(".info.json")
        return json.loads(p.read_text()) if p.exists() else {}
    L = ["# Сводка по шортсам (каждый кадр, 30 fps)\n",
         "| # | Ролик | Просм. | Длит. | Планов | Склеек/с | Ср. план, с | Камера движется | Петля | LUFS | Акцентов/с |",
         "|---|---|---|---|---|---|---|---|---|---|---|"]
    cams = Counter()
    for k, r in enumerate(reports, 1):
        i = info(r)
        for s in r["shots"]:
            for m in s["camera"].split(", "):
                cams[re.sub(r" x\d+", "", m)] += 1
        a = r.get("audio") or {}
        L.append(f"| {k} | {i.get('title', r['file'])[:50]} | {i.get('view_count', '—')} | {r['duration']} | "
                 f"{r['shots_count']} | {r['cuts_per_sec']} | {r['avg_shot_sec']} | "
                 f"{int(r['camera_moving_ratio']*100)}% | {r['loop_similarity_last_first']} | "
                 f"{a.get('integrated_lufs')} | {a.get('accents_per_sec')} |")
    allshots = [s for r in reports for s in r["shots"]]
    L += ["", "## Движение камеры (по всем планам)", ""]
    L += [f"- {m}: {c} планов ({c*100//max(1,len(allshots))}%)" for m, c in cams.most_common()]
    L += ["", "## Средние по всем роликам", "",
          f"- Средняя длина плана: {np.mean([r['avg_shot_sec'] for r in reports]):.2f} с",
          f"- Склеек в секунду: {np.mean([r['cuts_per_sec'] for r in reports]):.2f}",
          f"- Доля теней в кадре: {np.mean([s['shadow'] for s in allshots]):.2f}",
          f"- Контраст: {np.mean([s['contrast'] for s in allshots]):.3f}",
          f"- Жёсткость теней: {np.mean([s['shadow_hard'] for s in allshots]):.3f}",
          f"- Насыщенность: {np.mean([s['sat'] for s in allshots]):.3f}",
          f"- Тёплое/холодное: {np.mean([s['warm'] for s in allshots]):.2f} / {np.mean([s['cool'] for s in allshots]):.2f}",
          f"- Ключевой свет: " + ", ".join(f"{k} {v}" for k, v in Counter(s['key_light'] for s in allshots).most_common()),
          ]
    L += ["", "## Крючок: первые 3 секунды", ""]
    for k, r in enumerate(reports, 1):
        tr, h = r.get("transcript") or {}, r["hook_first3s"]
        L.append(f"- {k}. склеек {h['cuts']}, камера {h['camera_speed']}, наезд {h['zoom_per_s']}/с, "
                 f"речь с {tr.get('first_speech_at', '—')} с, слов/с {round(tr['words']/r['duration'], 2) if tr else '—'}: "
                 f"«{tr.get('first_3s', '') if tr else ''}»")
    (outdir / "summary.md").write_text("\n".join(L) + "\n")


def main():
    target = Path(sys.argv[1] if len(sys.argv) > 1 else "shorts")
    files = sorted(target.glob("*.mp4")) if target.is_dir() else [target]
    outdir = (target if target.is_dir() else target.parent) / "analysis"
    outdir.mkdir(exist_ok=True)
    reports = []
    for f in files:
        print(f"→ {f.name}", flush=True)
        r = analyze(f, outdir)
        r["_src"] = str(f)
        reports.append(r)
        print(f"   {r['frames_analyzed']} кадров, {r['shots_count']} планов, {r['cuts_per_sec']} склеек/с", flush=True)
    if reports:
        summary(reports, outdir)
        print(f"Готово: {outdir}/summary.md")


if __name__ == "__main__":
    main()
