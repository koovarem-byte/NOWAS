#!/usr/bin/env python3
"""
Монтаж серии S3 из отрендеренных планов (build_episode.py) + озвучки.

  python3 -I tools/s3blender/assemble_episode.py s3/episodes/s3e1.json [--render s3/render/s3e1] [--formats long,short]

Что делает (правила формата из scripts/S3_plan_i_scenarii.md):
  * 3D-планы — как отрендерены (камера уже движется), между планами наплыв 6 кадров;
  * экраны телефона — 2D-вставки поверх размытого прошлого плана, с наездом (не дольше ~3 с);
  * субтитры — 1–2 слова, белые с чёрной обводкой, мелкие, ~70 % высоты (Shorts) / ~84 % (длинный),
    автоуменьшение по ширине (в S2 «ОТКРЫВАЮТСЯ» не влезало);
  * удары: вспышка + низкий удар в звуке на hits; плашка «ДЕНЬ N / 30» на 1 с;
  * звук: озвучка s3/voice/<ep>_<id>_<голос>.mp3 по началам планов, тихий гул, −14 LUFS.
    Пока озвучки нет — собирается немой черновик с субтитрами (тайминг по длине слов).
  Без HUD, без карточки в конце.
"""
import argparse, json, os, subprocess, math, glob
import numpy as np
import cv2
from PIL import Image, ImageDraw, ImageFont, ImageFilter

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
FPS = 30
SR = 48000
FB = "/usr/share/fonts/opentype/inter/InterDisplay-Black.otf"
FM = "/usr/share/fonts/opentype/inter/Inter-Bold.otf"
SIZES = {"long": (1920, 1080), "short": (1080, 1920)}
XF = 6  # кадров наплыва


# ───────────── экраны телефона (2D) ─────────────
def phone_ui(kind, spec, t, dur, w, h):
    """Возвращает RGBA экрана телефона (w×h) в момент t."""
    im = Image.new("RGBA", (w, h), (14, 16, 22, 255))
    d = ImageDraw.Draw(im)
    s = w / 600
    F = lambda sz, b=True: ImageFont.truetype(FB if b else FM, int(sz * s))
    def center(y, txt, f, fill):
        d.text(((w - d.textlength(txt, font=f)) / 2, y), txt, font=f, fill=fill)
    if kind == "clock":
        center(h * 0.36, spec.get("text", "03:12"), F(150), (255, 70, 60))
        center(h * 0.58, "вторник · ночь", F(34, False), (170, 170, 180))
    elif kind == "loans":
        n = spec.get("count", 9)
        shown = min(n, int(t / (dur * 0.8) * n) + 1)
        sums = [33000, 28500, 41000, 30000, 25000, 36000, 39500, 30000, 37000]
        center(30 * s, "Мои займы", F(40), (255, 255, 255))
        for i in range(shown):
            y = 100 * s + i * 82 * s
            d.rounded_rectangle([24 * s, y, w - 24 * s, y + 70 * s], int(14 * s), fill=(34, 38, 48))
            d.text((44 * s, y + 10 * s), f"Займ №{i + 1}", font=F(26, False), fill=(200, 200, 210))
            d.text((44 * s, y + 38 * s), f"{sums[i % 9]:,} ₽".replace(",", " "), font=F(26), fill=(255, 255, 255))
            d.text((w - 190 * s, y + 22 * s), "ОДОБРЕН", font=F(26), fill=(255, 90, 70))
        center(h - 90 * s, f"Итого: 300 000 ₽", F(40), (255, 90, 70))
    elif kind == "selfie":
        d.rounded_rectangle([40 * s, 80 * s, w - 40 * s, h * 0.66], int(18 * s), fill=(60, 70, 90))
        cx, cy, r = w / 2, h * 0.33, 120 * s
        d.ellipse([cx - r, cy - r, cx + r, cy + r], fill=(238, 160, 110), outline=(0, 0, 0), width=int(5 * s))
        d.chord([cx - r * 1.03, cy - r * 1.05, cx + r * 1.03, cy + r * 0.2], 180, 360, fill=(10, 10, 12))
        for sx in (-1, 1):
            d.ellipse([cx + sx * 42 * s - 16 * s, cy + 2 * s, cx + sx * 42 * s + 16 * s, cy + 34 * s], fill=(10, 10, 10))
        d.rounded_rectangle([cx + 40 * s, cy + 80 * s, cx + 200 * s, cy + 190 * s], int(8 * s), fill=(150, 30, 40))
        d.text((cx + 62 * s, cy + 118 * s), "ПАСПОРТ", font=F(22), fill=(240, 200, 120))
        center(h * 0.70, "Фото с паспортом", F(36), (255, 255, 255))
        center(h * 0.76, "загружено в 03:14", F(28, False), (255, 90, 70))
    elif kind == "bot":
        cx, cy, r = w / 2, h * 0.36, 130 * s
        d.ellipse([cx - r, cy - r, cx + r, cy + r], fill=(220, 30, 40))
        center(cy - 60 * s, "?", F(110), (255, 255, 255))
        center(h * 0.56, spec.get("text", "Поддержка"), F(54), (255, 255, 255))
        center(h * 0.63, "бот · был в сети давно", F(28, False), (160, 160, 170))
    elif kind == "call":
        center(h * 0.18, spec.get("text", "Неизвестный"), F(48), (255, 255, 255))
        center(h * 0.25, "мобильный · " + ("00:%02d" % int(t)), F(30, False), (170, 170, 180))
        for i, (col, sym) in enumerate((((230, 40, 50), "✕"), ((40, 190, 90), "✓"))):
            cx = w * (0.28 + 0.44 * i); cy = h * 0.8; r = 52 * s
            d.ellipse([cx - r, cy - r, cx + r, cy + r], fill=col)
    elif kind == "order":
        center(40 * s, "Заказ №34", F(44), (255, 255, 255))
        rows = [("Адрес", "ул. Лесная, 12, кв. 71"), ("Подъезд", "3, этаж 7"),
                ("Комментарий", "код домофона 147В"), ("", "оставить у двери")]
        for i, (k, v) in enumerate(rows):
            y = 130 * s + i * 105 * s
            hl = i >= 2 and t > dur * 0.4
            d.rounded_rectangle([24 * s, y, w - 24 * s, y + 92 * s], int(14 * s), fill=(90, 30, 30) if hl else (34, 38, 48))
            d.text((44 * s, y + 10 * s), k, font=F(24, False), fill=(170, 170, 180))
            d.text((44 * s, y + 42 * s), v, font=F(32), fill=(255, 255, 255))
    elif kind == "dbrow":
        d.text((24 * s, 30 * s), "база_поддержка.csv", font=F(30, False), fill=(120, 220, 140))
        for i in range(12):
            y = 90 * s + i * 62 * s
            mine = i == 6
            d.rectangle([16 * s, y, w - 16 * s, y + 54 * s], fill=(70, 20, 25) if mine else (24, 28, 34))
            txt = "Лесная 12 · 3п · 147В" if mine else f"адрес {100 + i * 7} · {i % 5 + 1}п · {200 + i * 13}К"
            d.text((28 * s, y + 12 * s), txt, font=F(26, False), fill=(255, 255, 255) if mine else (140, 150, 160))
            if mine and t > dur * 0.3:
                d.text((w - 140 * s, y + 8 * s), "№12", font=F(34), fill=(255, 70, 60))
    elif kind == "credit":
        center(40 * s, "Кредитная история", F(40), (255, 255, 255))
        n = spec.get("count", 10)
        for i in range(n):
            y = 110 * s + i * 66 * s
            new = i == n - 1
            d.rounded_rectangle([24 * s, y, w - 24 * s, y + 56 * s], int(12 * s), fill=(90, 25, 30) if new else (34, 38, 48))
            d.text((44 * s, y + 13 * s), f"{i + 1}. Микрозаём · " + ("сегодня 02:47" if new else "3 дня назад"),
                   font=F(26, not new and False or new), fill=(255, 255, 255))
    elif kind == "nosignal":
        center(h * 0.08, "Нет сети", F(34, False), (255, 90, 70))
        if spec.get("sms", True) and t < dur * 0.5:
            d.rounded_rectangle([24 * s, h * 0.2, w - 24 * s, h * 0.36], int(16 * s), fill=(40, 44, 54))
            d.text((44 * s, h * 0.215), "SMS · Оператор", font=F(24, False), fill=(170, 170, 180))
            d.text((44 * s, h * 0.26), "Ваша SIM-карта заменена", font=F(30), fill=(255, 255, 255))
        else:
            center(h * 0.42, "SOS", F(140), (255, 70, 60))
    elif kind == "proxy":
        im.paste((240, 238, 230, 255), [0, 0, w, h])
        center(60 * s, "ДОВЕРЕННОСТЬ", F(44), (20, 20, 20))
        for i in range(9):
            y = 160 * s + i * 52 * s
            d.line([40 * s, y, w - 40 * s - (i % 3) * 60 * s, y], fill=(150, 150, 150), width=max(2, int(4 * s)))
        d.text((40 * s, h * 0.7), "Подпись:", font=F(30, False), fill=(20, 20, 20))
        pts = [(220 * s + k * 18 * s, h * 0.72 + (12 * s if k % 2 else -10 * s)) for k in range(14)]
        d.line(pts, fill=(30, 40, 160), width=max(2, int(4 * s)))
        if t > dur * 0.5:
            d.ellipse([w * 0.55, h * 0.64, w * 0.95, h * 0.8], outline=(230, 30, 40), width=int(8 * s))
    elif kind == "simban":
        center(60 * s, "Самозапрет", F(48), (255, 255, 255))
        center(122 * s, "на договоры связи без вас", F(32, False), (190, 190, 200))
        on = t > dur * 0.35
        y = h * 0.4
        d.rounded_rectangle([30 * s, y, w - 30 * s, y + 100 * s], int(16 * s), fill=(34, 38, 48))
        d.text((54 * s, y + 30 * s), "Новые SIM-карты", font=F(32, False), fill=(235, 235, 240))
        tx = w - 150 * s
        d.rounded_rectangle([tx, y + 26 * s, tx + 100 * s, y + 74 * s], int(24 * s), fill=(40, 190, 90) if on else (80, 80, 90))
        kx = tx + (54 * s if on else 4 * s)
        d.ellipse([kx, y + 30 * s, kx + 42 * s, y + 72 * s], fill=(255, 255, 255))
        if on:
            center(h * 0.62, "Установлен ✓", F(40), (60, 220, 110))
    elif kind == "selfban":
        center(60 * s, "Самозапрет", F(48), (255, 255, 255))
        center(122 * s, "на кредиты и займы", F(32, False), (190, 190, 200))
        on = t > dur * 0.35
        for i, lbl in enumerate(["Банки", "МФО", "Онлайн и в офисе"]):
            y = h * 0.32 + i * 110 * s
            d.rounded_rectangle([30 * s, y, w - 30 * s, y + 90 * s], int(16 * s), fill=(34, 38, 48))
            d.text((54 * s, y + 26 * s), lbl, font=F(32, False), fill=(235, 235, 240))
            tx = w - 150 * s
            d.rounded_rectangle([tx, y + 22 * s, tx + 100 * s, y + 70 * s], int(24 * s), fill=(40, 190, 90) if on else (80, 80, 90))
            kx = tx + (54 * s if on else 4 * s)
            d.ellipse([kx, y + 26 * s, kx + 42 * s, y + 68 * s], fill=(255, 255, 255))
        if on:
            center(h * 0.78, "Установлен ✓", F(40), (60, 220, 110))
    return im


def phone_frame(ui, H):
    """Корпус телефона вокруг экрана."""
    w, h = ui.size
    pad = int(w * 0.05)
    fr = Image.new("RGBA", (w + pad * 2, h + pad * 2), (0, 0, 0, 0))
    d = ImageDraw.Draw(fr)
    d.rounded_rectangle([0, 0, fr.width - 1, fr.height - 1], int(w * 0.12), fill=(8, 8, 10, 255), outline=(70, 70, 80, 255), width=max(2, w // 150))
    m = Image.new("L", ui.size, 0)
    ImageDraw.Draw(m).rounded_rectangle([0, 0, w - 1, h - 1], int(w * 0.09), fill=255)
    fr.paste(ui, (pad, pad), m)
    return fr


def screen_frames(spec, dur, bg, W, H):
    n = int(round(dur * FPS))
    base = cv2.GaussianBlur(bg, (0, 0), 18).astype(np.float32) * 0.45
    pw = int(H * 0.40) if W > H else int(W * 0.66)
    ph = int(pw * 2.0)
    out = []
    for i in range(n):
        t = i / FPS
        z = 1.0 + 0.12 * (t / dur)                      # наезд ~4 %/с
        ui = phone_ui(spec["kind"], spec, t, dur, pw, ph)
        fr = phone_frame(ui, H)
        fw, fh = int(fr.width * z), int(fr.height * z)
        fr = fr.resize((fw, fh), Image.BILINEAR)
        canvas = Image.fromarray(cv2.cvtColor(np.clip(base, 0, 255).astype(np.uint8), cv2.COLOR_BGR2RGB)).convert("RGBA")
        canvas.alpha_composite(fr, ((W - fw) // 2, (H - fh) // 2 - int(H * 0.03)))
        out.append(cv2.cvtColor(np.array(canvas.convert("RGB")), cv2.COLOR_RGB2BGR))
    return out


# ───────────── 3D-планы ─────────────
def shot_frames(render_dir, fmt, sid, dur, step, W, H):
    files = sorted(glob.glob(os.path.join(render_dir, fmt, sid, "f_*.png")))
    n = int(round(dur * FPS))
    if not files:
        return None
    out = []
    by_frame = {int(os.path.basename(f)[2:6]): f for f in files}
    keys = sorted(by_frame)
    cache = {}
    for i in range(n):
        k = max([x for x in keys if x <= i + 1] or [keys[0]])
        if k not in cache:
            cache.clear()
            cache[k] = cv2.resize(cv2.imread(by_frame[k]), (W, H), interpolation=cv2.INTER_CUBIC)
        out.append(cache[k])
    return out


# ───────────── субтитры ─────────────
def word_chunks(text):
    """1–2 слова на чанк: короткие слова (≤3 букв) приклеиваются к следующему."""
    ws = text.replace("—", "").split()
    out, cur = [], []
    for w in ws:
        cur.append(w)
        if len(cur) == 2 or (len(w.strip(".,!?…«»")) > 3 and not (len(cur) == 1 and len(w) <= 3)):
            out.append(" ".join(cur)); cur = []
    if cur:
        out.append(" ".join(cur))
    return out


def asr_words(ep, sh):
    p = os.path.join(ROOT, "s3", "work", "asr", f"{ep}_{sh['id']}_{sh['voice']}.json")
    return json.load(open(p, encoding="utf-8")) if os.path.exists(p) else None


def sub_timeline(shots, starts, durs, ep):
    """[(t0, t1, текст)]. Если есть слова Whisper (transcribe_s3.py) — по ним, точно в голос;
    иначе внутри фразы пропорционально длине слов."""
    ev = []
    for sh, t0, d in zip(shots, starts, durs):
        words = asr_words(ep, sh)
        if words:
            i = 0
            while i < len(words):
                grp = [words[i]]
                if len(words[i]["w"].strip(".,!?…«»—")) <= 3 and i + 1 < len(words):
                    grp.append(words[i + 1])
                i += len(grp)
                a0 = t0 + grp[0]["s"]
                a1 = t0 + (words[i]["s"] if i < len(words) else grp[-1]["e"] + 0.15)
                ev.append((a0, a1, " ".join(g["w"] for g in grp)))
            continue
        chunks = word_chunks(sh["text"])
        speak = d - 0.3
        total = sum(len(c) + 3 for c in chunks)
        t = t0 + 0.05
        for c in chunks:
            dt = speak * (len(c) + 3) / total
            ev.append((t, t + dt, c))
            t += dt
    return ev


def draw_sub(img, txt, W, H, fmt):
    pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    d = ImageDraw.Draw(pil)
    size = int((0.052 if fmt == "short" else 0.06) * (W if fmt == "short" else H))
    txt = txt.upper()
    f = ImageFont.truetype(FB, size)
    maxw = W * 0.9
    while d.textlength(txt, font=f) > maxw and size > 12:      # автоуменьшение по ширине
        size = int(size * 0.92); f = ImageFont.truetype(FB, size)
    tw = d.textlength(txt, font=f)
    y = H * (0.70 if fmt == "short" else 0.84)
    d.text(((W - tw) / 2, y), txt, font=f, fill=(255, 255, 255), stroke_width=max(3, size // 9), stroke_fill=(0, 0, 0))
    return cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)


def draw_plate(img, txt, W, H, a):
    pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)).convert("RGBA")
    ov = Image.new("RGBA", pil.size, (0, 0, 0, 0))
    d = ImageDraw.Draw(ov)
    size = int(min(W, H) * 0.07)
    f = ImageFont.truetype(FB, size)
    tw = d.textlength(txt, font=f)
    x, y = (W - tw) / 2, H * 0.12
    pad = size * 0.35
    d.rounded_rectangle([x - pad, y - pad * 0.6, x + tw + pad, y + size * 1.25], int(pad), fill=(230, 30, 40, int(235 * a)))
    d.text((x, y), txt, font=f, fill=(255, 255, 255, int(255 * a)))
    pil.alpha_composite(ov)
    return cv2.cvtColor(np.array(pil.convert("RGB")), cv2.COLOR_RGB2BGR)


# ───────────── звук ─────────────
def build_audio(ep, shots, starts, durs, hits_abs, total, out_wav):
    voice_dir = os.path.join(ROOT, "s3", "voice")
    inputs, filters, labels = [], [], []
    idx = 0
    for sh, t0 in zip(shots, starts):
        p = os.path.join(voice_dir, f"{ep}_{sh['id']}_{sh['voice']}.mp3")
        if not os.path.exists(p):
            continue
        inputs += ["-i", p]
        flt = f"[{idx}:a]aresample={SR},pan=stereo|c0=c0|c1=c0"
        if sh["voice"] == "K":       # коллектор — «из телефона»
            flt += ",highpass=f=300,lowpass=f=3400,acrusher=bits=12:mix=0.2"
        flt += f",adelay={int(t0 * 1000)}|{int(t0 * 1000)}[v{idx}]"
        filters.append(flt); labels.append(f"[v{idx}]"); idx += 1
    # подложка: низкий гул + удары
    filters.append(f"sine=f=49:d={total:.3f}:sample_rate={SR},volume=0.06,pan=stereo|c0=c0|c1=c0[drone]")
    labels.append("[drone]")
    for k, t in enumerate(hits_abs):
        filters.append(f"sine=f=52:d=0.5:sample_rate={SR},afade=t=out:st=0:d=0.5,volume=0.9,"
                       f"pan=stereo|c0=c0|c1=c0,adelay={int(t * 1000)}|{int(t * 1000)}[h{k}]")
        labels.append(f"[h{k}]")
    filters.append(f"{''.join(labels)}amix=inputs={len(labels)}:normalize=0,atrim=0:{total:.3f},loudnorm=I=-14:TP=-1.5:LRA=11[a]")
    subprocess.check_call(["ffmpeg", "-y", "-v", "error", *inputs, "-filter_complex", ";".join(filters),
                           "-map", "[a]", "-ar", str(SR), out_wav])
    return idx


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("episode")
    ap.add_argument("--render")
    ap.add_argument("--formats", default="long,short")
    ap.add_argument("--out")
    a = ap.parse_args()
    ep = json.load(open(a.episode, encoding="utf-8"))
    name = ep["episode"]
    rdir = a.render or os.path.join(ROOT, "s3", "render", name)
    man = json.load(open(os.path.join(rdir, "manifest.json"), encoding="utf-8"))
    durs = {m["id"]: m["dur"] for m in man["shots"]}
    step = man.get("step", 1)
    shots = ep["shots"]
    ds = [durs[s["id"]] for s in shots]
    starts = list(np.cumsum([0] + ds[:-1]))
    total = float(sum(ds))
    hits_abs = [t0 + h for s, t0 in zip(shots, starts) for h in s.get("hits", [])]
    subs = sub_timeline(shots, starts, ds, name)
    out_dir = a.out or os.path.join(ROOT, "s3", "out")
    os.makedirs(out_dir, exist_ok=True)
    wav = os.path.join(out_dir, f"{name}.wav")
    n_voice = build_audio(name, shots, starts, ds, hits_abs, total, wav)
    print(f"озвучка: {n_voice}/{len(shots)} фраз" + ("" if n_voice else " — немой черновик"))
    for fmt in a.formats.split(","):
        W, H = SIZES[fmt]
        outp = os.path.join(out_dir, f"{name}_{fmt}.mp4")
        enc = subprocess.Popen(["ffmpeg", "-y", "-v", "error", "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{W}x{H}",
                                "-r", str(FPS), "-i", "-", "-i", wav, "-map", "0:v", "-map", "1:a",
                                "-c:v", "libx264", "-preset", "medium", "-crf", "17", "-pix_fmt", "yuv420p",
                                "-c:a", "aac", "-b:a", "256k", "-shortest", "-movflags", "+faststart", outp],
                               stdin=subprocess.PIPE)
        prev_tail = np.zeros((H, W, 3), np.uint8)
        gi = 0
        for s, t0, d in zip(shots, starts, ds):
            if "set" in s:
                fr = shot_frames(rdir, fmt, s["id"], d, step, W, H)
                if fr is None:
                    fr = [cv2.putText(np.full((H, W, 3), 30, np.uint8), f"план {s['id']} не отрендерен", (40, H // 2),
                                      cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 3)] * int(round(d * FPS))
            else:
                fr = screen_frames(s["screen"], d, prev_tail, W, H)
            for i, f in enumerate(fr):
                if i < XF and gi > 0:                                     # наплыв
                    f = cv2.addWeighted(prev_tail, 1 - (i + 1) / (XF + 1), f, (i + 1) / (XF + 1), 0)
                t = gi / FPS
                for h in hits_abs:                                        # вспышка удара
                    if 0 <= t - h < 0.12:
                        f = cv2.addWeighted(f, 0.6, np.full_like(f, 255), 0.4, 0)
                if s.get("plate") and i < FPS * 1.2:
                    f = draw_plate(f, s["plate"], W, H, min(1, i / 5, (FPS * 1.2 - i) / 5))
                for (a0, a1, txt) in subs:
                    if a0 <= t < a1:
                        f = draw_sub(f, txt, W, H, fmt); break
                enc.stdin.write(np.ascontiguousarray(f).tobytes())
                gi += 1
            prev_tail = fr[-1] if "set" in s else prev_tail
        enc.stdin.close(); enc.wait()
        print("готово:", outp, f"{total:.1f} с")


if __name__ == "__main__":
    main()
