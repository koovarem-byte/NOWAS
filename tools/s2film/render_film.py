#!/usr/bin/env python3
"""
Сборка фильма «Курьер. Сезон 2» из готовых частей по монтажному листу (EDL).

Что делает с картинкой (каждый кадр, 30 fps):
  - срезает верхний HUD (сезон/таймер/этаж) кропом с наездом;
  - на каждом плане медленный наезд + лёгкий дрейф (камера всегда движется);
  - на ударных словах — рывок зума + тряска;
  - между частями — «хлыст» (смазанный проезд) 8 кадров;
  - грейд: насыщенность, контраст, тёплые света/холодные тени, виньетка;
  - свои субтитры там, где исходный кадр заменён.
Звук: чистые голоса ElevenLabs по выравниванию + синтезированная подложка/удары/свисты.

  python3 -I tools/s2film/render_film.py s2/edl.json s2/out/kurer_s2_film.mp4 [--preview]
"""
import json, subprocess, sys, math
from pathlib import Path

import cv2
import numpy as np

FPS = 30
W, H = 1080, 1920
SR = 48000
HUD_TOP = 0.14            # доля кадра сверху, занятая HUD во всех частях
FONT = "/usr/share/fonts/opentype/inter/InterDisplay-Black.otf"


# ───────────────────────── картинка ─────────────────────────
def read_frames(src, t_in, t_out, w=W, h=H):
    dur = t_out - t_in
    p = subprocess.Popen(["ffmpeg", "-v", "error", "-ss", f"{t_in:.3f}", "-i", src, "-t", f"{dur:.3f}",
                          "-vf", f"fps={FPS},scale={w}:{h}:flags=lanczos", "-f", "rawvideo", "-pix_fmt", "bgr24", "-"],
                         stdout=subprocess.PIPE)
    n = int(round(dur * FPS))
    out = []
    while len(out) < n:
        b = p.stdout.read(w * h * 3)
        if len(b) < w * h * 3:
            break
        out.append(np.frombuffer(b, np.uint8).reshape(h, w, 3))
    p.stdout.close(); p.wait()
    while len(out) < n and out:           # добиваем последним кадром, если декодер отдал меньше
        out.append(out[-1])
    return out


def camera(frame, zoom, cx, cy, top_min, shake=(0.0, 0.0), w=W, h=H):
    """Камера по готовому кадру.
    zoom >= 1: вырезает окно W/zoom x H/zoom вокруг (cx, cy) и растягивает (наезд);
    zoom <  1: уменьшает кадр, края заполняет размытой копией (отъезд без потери субтитров по краям)."""
    if zoom < 1:
        bg = cv2.resize(frame, (w // 10, h // 10), interpolation=cv2.INTER_AREA)
        bg = cv2.GaussianBlur(cv2.resize(bg, (w, h), interpolation=cv2.INTER_LINEAR), (0, 0), 18)
        bg = (bg.astype(np.float32) * 0.6).astype(np.uint8)
        sw, sh = int(round(w * zoom)), int(round(h * zoom))
        small = cv2.resize(frame, (sw, sh), interpolation=cv2.INTER_AREA)
        x0 = int(round((w - sw) / 2 + (0.5 - cx) * (w - sw) + shake[0]))
        y0 = int(round((h - sh) / 2 + shake[1]))
        x0 = max(0, min(x0, w - sw)); y0 = max(0, min(y0, h - sh))
        bg[y0:y0 + sh, x0:x0 + sw] = small
        return bg
    cw, ch = w / zoom, h / zoom
    x = cx * w - cw / 2 + shake[0]
    y = cy * h - ch / 2 + shake[1]
    y = max(y, top_min * h); y = min(y, h - ch)
    x = max(0.0, min(x, w - cw))
    M = np.float32([[zoom, 0, -x * zoom], [0, zoom, -y * zoom]])
    return cv2.warpAffine(frame, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)


def fill_top(frame, frac):
    """Закрашивает верхнюю полосу (HUD/заголовок) сильно размытым продолжением картинки, с мягкой границей."""
    h = int(frame.shape[0] * frac)
    if h <= 0:
        return frame
    src = frame[h:h + max(8, h // 3)]
    small = cv2.resize(src, (frame.shape[1] // 16, max(2, src.shape[0] // 16)), interpolation=cv2.INTER_AREA)
    col = cv2.resize(small.mean(0, keepdims=True).astype(np.uint8), (frame.shape[1], h + 40), interpolation=cv2.INTER_LINEAR)
    col = cv2.GaussianBlur(col, (0, 0), 35)
    out = frame.copy()
    a = np.ones((h + 40, 1, 1), np.float32)
    a[h:] = np.linspace(1, 0, 40)[:, None, None]             # мягкая граница 40 px
    region = out[:h + 40].astype(np.float32)
    out[:h + 40] = (col * a + region * (1 - a)).astype(np.uint8)
    return out


def caption_halfwidth(frame):
    """Полуширина вшитого субтитра (доля кадра): светлые пиксели рядом с чёрной обводкой в полосе субтитров."""
    band = frame[int(frame.shape[0] * 0.58):int(frame.shape[0] * 0.87):2, ::2]
    v = band.max(2)
    bright = v > 195
    dark = cv2.dilate((v < 50).astype(np.uint8), np.ones((5, 5), np.uint8)) > 0
    m = bright & dark
    cols = np.where(m.sum(0) >= 2)[0]
    if len(cols) < 6:
        return 0.0
    w = band.shape[1]
    lo, hi = np.percentile(cols, 1), np.percentile(cols, 99)
    return float(max(abs(lo - w / 2), abs(hi - w / 2)) / w)


_vig = None
def grade(img, look):
    global _vig
    if _vig is None:
        yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
        r = np.sqrt(((xx - W / 2) / (W / 2)) ** 2 + ((yy - H * 0.48) / (H / 2)) ** 2)
        _vig = np.clip(1 - 0.32 * np.clip(r - 0.55, 0, None) ** 1.6, 0.55, 1)[..., None].astype(np.float32)
    f = img.astype(np.float32) / 255
    sat, con, warm = look
    g = f.mean(2, keepdims=True)
    f = g + (f - g) * sat                                  # насыщенность
    f = 0.5 + (f - 0.5) * con                              # контраст
    lum = f.mean(2, keepdims=True)
    f[..., 2:3] += warm * lum * 0.06; f[..., 0:1] += warm * (1 - lum) * 0.05   # тёплые света / холодные тени
    f = f * _vig
    return np.clip(f * 255, 0, 255).astype(np.uint8)


def draw_caption(img, text, color=(255, 255, 255), size=82, y=0.72):
    """Субтитр в стиле новых правил: белый/цветной, чёрная обводка (через PIL — кириллица)."""
    from PIL import Image, ImageDraw, ImageFont
    font = ImageFont.truetype(FONT, size)
    im = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    d = ImageDraw.Draw(im)
    tw = d.textlength(text, font=font)
    d.text(((W - tw) / 2, y * H), text, font=font, fill=tuple(color),
           stroke_width=10, stroke_fill=(0, 0, 0))
    return cv2.cvtColor(np.array(im), cv2.COLOR_RGB2BGR)


def ease(x):
    return 0.5 - 0.5 * math.cos(math.pi * min(max(x, 0), 1))


# ───────────────────────── звук ─────────────────────────
def load_audio(path):
    raw = subprocess.run(["ffmpeg", "-v", "error", "-i", path, "-ac", "1", "-ar", str(SR), "-f", "f32le", "-"],
                         capture_output=True).stdout
    return np.frombuffer(raw, np.float32).copy()


def measure(mix):
    """первый проход loudnorm: замер, чтобы второй проход привёл точно к −14 LUFS без превышения пика"""
    import re as _re
    err = subprocess.run(["ffmpeg", "-v", "info", "-f", "f32le", "-ar", str(SR), "-ac", "1", "-i", "-",
                          "-af", "loudnorm=I=-14:TP=-2:LRA=11:print_format=json", "-f", "null", "-"],
                         input=mix.tobytes(), capture_output=True).stderr.decode("utf-8", "replace")
    j = json.loads(err[err.rindex("{"):err.rindex("}") + 1])
    return (f"measured_I={j['input_i']}:measured_TP={j['input_tp']}:measured_LRA={j['input_lra']}:"
            f"measured_thresh={j['input_thresh']}:offset={j['target_offset']}")


def phrase_audio(path, fs, fe, speed):
    af = ["-af", f"atempo={speed:.4f}"] if abs(speed - 1) > 1e-3 else []
    raw = subprocess.run(["ffmpeg", "-v", "error", "-ss", f"{fs:.3f}", "-t", f"{fe - fs:.3f}", "-i", path, *af,
                          "-ac", "1", "-ar", str(SR), "-f", "f32le", "-"], capture_output=True).stdout
    return np.frombuffer(raw, np.float32).copy()


def synth_bed(n, seed=1):
    """Тёмная подложка: два низких тона с медленной пульсацией + приглушённый шум."""
    t = np.arange(n) / SR
    rng = np.random.default_rng(seed)
    lfo = 0.6 + 0.4 * np.sin(2 * np.pi * 0.07 * t)
    tone = 0.5 * np.sin(2 * np.pi * 55 * t) + 0.35 * np.sin(2 * np.pi * 82.4 * t + 1) + 0.15 * np.sin(2 * np.pi * 110.3 * t)
    noise = rng.normal(0, 1, n).astype(np.float32)
    k = int(SR / 400)
    noise = np.convolve(noise, np.ones(k) / k, "same")      # грубый низкочастотный фильтр
    return ((tone * lfo + 0.6 * noise) * 0.05).astype(np.float32)


def synth_hit():
    t = np.arange(int(SR * 0.9)) / SR
    boom = np.sin(2 * np.pi * (48 + 40 * np.exp(-t * 18)) * t) * np.exp(-t * 5)
    click = np.random.default_rng(3).normal(0, 1, len(t)) * np.exp(-t * 60) * 0.3
    return ((boom + click) * 0.55).astype(np.float32)


def synth_whoosh(d=0.45):
    n = int(SR * d); t = np.arange(n) / SR
    x = np.random.default_rng(5).normal(0, 1, n)
    env = np.sin(np.pi * t / d) ** 2
    out = np.zeros(n); y = 0.0
    for i in range(n):                                       # фильтр с «едущей» частотой среза
        a = 0.02 + 0.25 * (t[i] / d)
        y += a * (x[i] - y); out[i] = y
    return (out * env * 0.5).astype(np.float32)


def synth_heart():
    t = np.arange(int(SR * 0.35)) / SR
    b = np.sin(2 * np.pi * 52 * t) * np.exp(-t * 22)
    beat = np.concatenate([b, np.zeros(int(SR * 0.12)), 0.7 * b])
    return (beat * 0.5).astype(np.float32)


def place(buf, clip, t, gain=1.0):
    i = int(t * SR)
    if i >= len(buf) or i + len(clip) <= 0:
        return
    a, b = max(0, i), min(len(buf), i + len(clip))
    buf[a:b] += clip[a - i:b - i] * gain


# ───────────────────────── сборка ─────────────────────────
def main():
    edl = json.loads(Path(sys.argv[1]).read_text())
    out = Path(sys.argv[2]); out.parent.mkdir(parents=True, exist_ok=True)
    preview = "--preview" in sys.argv
    audio_only = "--audio-only" in sys.argv          # пересобрать только звук и подложить к уже готовой картинке
    pw, ph = (540, 960) if preview else (W, H)
    root = Path(sys.argv[1]).parent
    parts, looks = edl["parts"], edl["looks"]
    phrases = json.loads((root / "work/phrases.json").read_text())

    segs = edl["segments"]
    total = sum(s["out"] - s["in"] for s in segs)
    nfr = int(round(total * FPS))
    voice = np.zeros(int((total + 1) * SR), np.float32)
    fx = np.zeros_like(voice)
    placed = []

    enc = None if audio_only else subprocess.Popen(["ffmpeg", "-y", "-v", "error", "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{pw}x{ph}",
                            "-r", str(FPS), "-i", "-", "-c:v", "libx264", "-preset", "medium", "-crf", "18",
                            "-pix_fmt", "yuv420p", str(out.with_suffix(".video.mp4"))], stdin=subprocess.PIPE)
    t_out = 0.0
    log = []
    prev_part = None
    for k, s in enumerate(segs):
        part = s["part"]; src = str(root / parts[part])
        dur = s["out"] - s["in"]
        vis = s.get("vis")                                   # замена картинки: [часть, начало]
        if audio_only:
            frames = [None] * int(round(dur * FPS))
        elif s.get("freeze") is not None:
            f0 = read_frames(src, s["freeze"], s["freeze"] + 0.05)[0]
            frames = [f0] * int(round(dur * FPS))
        elif vis:
            frames = read_frames(str(root / parts[vis[0]]), vis[1], vis[1] + dur)
        else:
            frames = read_frames(src, s["in"], s["out"])
        n = len(frames)
        z0, z1 = s.get("zoom", [1.0, 1.06])
        fill = max(HUD_TOP, s.get("top", HUD_TOP))           # HUD (и заголовок в начале части) закрашиваем, а не режем
        if not audio_only:
            frames = [fill_top(f, fill) for f in frames]
            # предел зума по ширине субтитра: субтитр должен влезать целиком с запасом 3.5 %
            hw = np.array([caption_halfwidth(f) for f in frames])
            cap_z = np.clip(1 / (2 * (hw + 0.035)), 1.0, 1.12)
            kw = 15                                          # заранее «видим» широкий субтитр и плавно отъезжаем
            cap_z = np.array([cap_z[max(0, i - kw):i + kw + 1].min() for i in range(n)])
            cap_z = np.convolve(np.pad(cap_z, 6, mode="edge"), np.ones(13) / 13, "valid")
        fx_, fy_ = s.get("focus", [0.5, 0.5])
        drift = s.get("drift", 0.015 * (1 if k % 2 else -1)) if False else s.get("drift", 0.015)
        hits = [h - s["in"] for h in s.get("hits", [])]
        cap = s.get("caption")                               # свои субтитры [[текст, начало, конец, цвет], ...]
        look = looks[s.get("look", part)]
        whip_in = prev_part is not None and prev_part != part
        for i, fr in enumerate(frames):
            if audio_only:
                break
            tt = i / FPS
            m = z0 + (z1 - z0) * ease(i / max(1, n - 1))
            sx = sy = 0.0
            for h in hits:
                d = tt - h
                if 0 <= d < 0.35:
                    m *= 1 + 0.06 * math.exp(-d * 12)
                    a = 12 * math.exp(-d * 14)
                    sx += a * math.sin(d * 95); sy += a * math.cos(d * 77)
            zoom = min(m, float(cap_z[i]))
            cx = fx_ + drift * (i / max(1, n - 1) - 0.5) * 2
            img = camera(fr, zoom, cx, fy_, 0.0, (sx, sy))
            if whip_in and i < 8:                            # «хлыст» на входе в новую часть
                a = (8 - i) / 8
                kx = int(1 + 120 * a)
                img = cv2.blur(img, (kx, 1))
                img = np.roll(img, int(-260 * a ** 2), axis=1)
            img = grade(img, look)
            for c in (cap or []):
                if c[1] <= tt < c[2]:
                    img = draw_caption(img, c[0], tuple(c[3]) if len(c) > 3 else (255, 255, 255))
            if preview:
                img = cv2.resize(img, (pw, ph), interpolation=cv2.INTER_AREA)
            enc.stdin.write(img.tobytes())
        # голос: фраза принадлежит сегменту, в котором она начинается
        if s.get("freeze") is None:
            for p in phrases:
                if p["part"] == part and s["in"] - 0.02 <= p["vt"] < s["out"] - 0.02:
                    placed.append((t_out + (p["vt"] - s["in"]), p))
        for h in hits:
            place(fx, synth_hit(), t_out + h, 0.5)
        if whip_in:
            place(fx, synth_whoosh(), t_out - 0.15, 0.6)
        for hb in s.get("heart", []):
            place(fx, synth_heart(), t_out + hb - s["in"], 0.45)
        prev_part = part
        t_out += n / FPS
    if enc:
        enc.stdin.close(); enc.wait()
    placed.sort(key=lambda x: x[0])
    for k, (t, p) in enumerate(placed):
        sp = p["speed"]
        if k + 1 < len(placed):                              # не наезжаем на следующую фразу: сначала чуть ускоряем
            room = placed[k + 1][0] - t - 0.02
            need = (p["fe"] - p["fs"]) / sp
            if need > room > 0:
                sp = min(1.3, sp * need / room)
        a = phrase_audio(str(root / "voice" / f"{p['file']}.mp3"), p["fs"], p["fe"], sp)
        cut = 0.0
        if k + 1 < len(placed):
            keep = max(0, int((placed[k + 1][0] - t - 0.01) * SR))
            cut = max(0.0, (len(a) - keep) / SR)
            a = a[:keep]
        if sp > p["speed"] + 0.02:
            print(f"  ускорена до x{sp:.2f}: «{p['text'][:45]}»")
        if cut > 0.08:
            print(f"  ВНИМАНИЕ: фраза «{p['text'][:40]}» обрезана на {cut:.2f} с")
        fade = min(len(a) // 2, int(0.012 * SR))
        if fade:
            a[:fade] *= np.linspace(0, 1, fade); a[-fade:] *= np.linspace(1, 0, fade)
        place(voice, a, t)
        log.append((round(t, 2), p["text"]))

    # микс: подложка приглушается под голосом
    bed = synth_bed(len(voice))
    env = np.convolve(np.abs(voice), np.ones(int(SR * 0.25)) / (SR * 0.25), "same")
    duck = 1 - 0.6 * np.clip(env / (env.max() * 0.15 + 1e-9), 0, 1)
    mix = voice * 1.0 + fx + bed * duck
    for nm, arr in (("голос", voice), ("эффекты", fx), ("подложка", bed * duck), ("микс", mix)):
        print(f"  {nm}: пик {np.nanmax(np.abs(arr)):.3f}, NaN {int(np.isnan(arr).sum())}")
    mix = np.nan_to_num(mix).astype(np.float32)
    wav = out.with_suffix(".wav")
    subprocess.run(["ffmpeg", "-y", "-v", "error", "-f", "f32le", "-ar", str(SR), "-ac", "1", "-i", "-",
                    "-af", "loudnorm=I=-14:TP=-2:LRA=11:linear=true:" + measure(mix),
                    "-ar", str(SR), str(wav)], input=mix.tobytes(), check=True)
    vid = out.with_suffix(".video.mp4")
    if audio_only:
        out.rename(vid)
    subprocess.run(["ffmpeg", "-y", "-v", "error", "-i", str(vid), "-i", str(wav), "-map", "0:v:0", "-map", "1:a:0",
                    "-af", "volume=1.5dB,alimiter=limit=0.84:attack=3:release=60:level=false",
                    "-c:v", "copy", "-c:a", "aac", "-b:a", "192k", "-shortest", "-movflags", "+faststart", str(out)],
                   check=True)
    out.with_suffix(".video.mp4").unlink(); wav.unlink()
    (out.with_suffix(".voice_log.txt")).write_text("\n".join(f"{t:7.2f}  {x}" for t, x in log))
    print(f"Готово: {out}  {t_out:.1f} с, {nfr} кадров")


if __name__ == "__main__":
    main()
