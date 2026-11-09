#!/usr/bin/env python3
"""
Вертикальный фильм (9:16) → горизонтальный длинный ролик YouTube (16:9) БЕЗ обрезки кадра.

Почему: YouTube считает Shorts любой вертикальный/квадратный ролик до 3 минут.
Горизонтальный 16:9 — всегда обычное (длинное) видео, с любой длиной, с главами,
конечной заставкой и монетизацией длинных видео.

Как расширяется кадр (ничего не срезается, исходник 9:16 стоит целиком по центру):
  1. Фон — тот же кадр, растянутый на всю ширину, сильно размытый и притемнённый.
  2. Поверх фона по бокам — «продолжение сцены»: крайние полосы исходного кадра
     отражены и растянуты наружу, с размытием, растущим к краю. Шов не виден,
     кажется, что сцена шире.
  3. Мягкая тень вокруг центрального кадра, виньетка, лёгкое зерно — фон не «пластиковый».
  4. Фон сглажен по времени (без мерцания), но на склейках обновляется сразу.
  5. В конец добавляется конечная заставка (по умолчанию 20 с) под элементы YouTube:
     «Подписаться» и «Следующее видео». Заставка ≥ 5 с и ≤ 20 с — требование YouTube.

  python3 -I tools/longform/to_long.py IN.mp4 OUT.mp4 [--4k] [--end 20] [--end-title "СЕЗОН 3 · СКОРО"]
  --4k        3840×2160 (YouTube даёт 4K-роликам кодек получше — картинка чище даже у тех, кто смотрит в 1080p)
  --end 0     без заставки
  --preview   только первые 20 с + заставка (быстрая проверка)
"""
import argparse, subprocess, sys, json
from pathlib import Path

import cv2
import numpy as np

FPS = 30
FONT_BLACK = "/usr/share/fonts/opentype/inter/InterDisplay-Black.otf"
FONT_BOLD = "/usr/share/fonts/opentype/inter/Inter-Bold.otf"


def probe(path):
    out = subprocess.check_output(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
                                   "stream=width,height:format=duration", "-of", "json", path])
    j = json.loads(out)
    s = j["streams"][0]
    return s["width"], s["height"], float(j["format"]["duration"])


def frames(path, w, h, limit=None):
    cmd = ["ffmpeg", "-v", "error", "-i", path]
    if limit:
        cmd += ["-t", str(limit)]
    cmd += ["-vf", f"fps={FPS},scale={w}:{h}:flags=lanczos", "-f", "rawvideo", "-pix_fmt", "bgr24", "-"]
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE)
    n = w * h * 3
    while True:
        b = p.stdout.read(n)
        if len(b) < n:
            break
        yield np.frombuffer(b, np.uint8).reshape(h, w, 3)
    p.stdout.close(); p.wait()


class Extender:
    def __init__(self, W, H, fw):
        self.W, self.H, self.fw = W, H, fw
        self.side = (W - fw) // 2
        self.prev_bg = None
        self.prev_small = None
        # маски заранее
        x = np.arange(self.side, dtype=np.float32)
        d = (self.side - x) / self.side                      # 0 у шва → 1 у края экрана (для левой панели)
        self.ext_alpha_l = np.clip(1.0 - d * 1.15, 0, 1) ** 1.3  # продолжение сцены гаснет к краю
        self.ext_alpha_r = self.ext_alpha_l[::-1].copy()
        yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
        r = np.sqrt(((xx - W / 2) / (W / 2)) ** 2 + ((yy - H / 2) / (H / 2)) ** 2)
        self.vign = np.clip(1.0 - 0.45 * np.clip(r - 0.35, 0, None) ** 1.4, 0.35, 1.0)[..., None]
        # тень вокруг центрального кадра
        m = np.zeros((H, W), np.float32)
        x0 = self.side
        m[:, x0:x0 + fw] = 1.0
        sh = cv2.GaussianBlur(m, (0, 0), H * 0.025)
        self.shadow = (1.0 - 0.55 * sh)[..., None]
        self.rng = np.random.default_rng(7)

    def bg_cover(self, fr):
        W, H = self.W, self.H
        h0, w0 = fr.shape[:2]
        # cover: растянуть по ширине, взять середину по высоте (чуть выше центра — там лица)
        s = W / w0
        small = cv2.resize(fr, (max(1, int(w0 * s / 8)), max(1, int(h0 * s / 8))), interpolation=cv2.INTER_AREA)
        hh = small.shape[0]
        top = int(max(0, (hh - H // 8) * 0.42))
        small = small[top:top + H // 8]
        small = cv2.GaussianBlur(small, (0, 0), 6)
        return cv2.resize(small, (W, H), interpolation=cv2.INTER_CUBIC)

    def side_ext(self, fr):
        """Отражённое продолжение крайних полос кадра наружу, с растущим размытием."""
        H, side = self.H, self.side
        h0, w0 = fr.shape[:2]
        strip = max(8, int(w0 * 0.16))
        L = cv2.flip(fr[:, :strip], 1)
        R = cv2.flip(fr[:, -strip:], 1)
        out = []
        for s, flip_dir in ((L, "l"), (R, "r")):
            s = cv2.resize(s, (side, H), interpolation=cv2.INTER_LINEAR)
            # размытие растёт к краю: смешиваем 3 уровня
            b1 = cv2.GaussianBlur(s, (0, 0), 6)
            b2 = cv2.GaussianBlur(s, (0, 0), 22)
            x = np.linspace(0, 1, side, dtype=np.float32)       # 0 у шва
            if flip_dir == "l":
                x = x[::-1]
            k1 = np.clip(1 - x * 2, 0, 1)[None, :, None]
            k2 = np.clip(x * 2 - 0.0, 0, 1)[None, :, None]
            mix = (b1 * k1 + b2 * k2) / np.maximum(k1 + k2, 1e-3)
            out.append(mix.astype(np.float32))
        return out

    def __call__(self, fr):
        W, H, side, fw = self.W, self.H, self.side, self.fw
        fg = cv2.resize(fr, (fw, H), interpolation=cv2.INTER_LANCZOS4 if fr.shape[0] < H else cv2.INTER_AREA)
        tiny = cv2.resize(fr, (16, 28), interpolation=cv2.INTER_AREA).astype(np.float32)
        cut = self.prev_small is None or np.abs(tiny - self.prev_small).mean() > 18
        self.prev_small = tiny
        bg = self.bg_cover(fr).astype(np.float32)
        if not cut and self.prev_bg is not None:
            bg = self.prev_bg * 0.6 + bg * 0.4
        self.prev_bg = bg
        bgd = bg * 0.42
        L, R = self.side_ext(fr)
        al = self.ext_alpha_l[None, :, None] * 0.75
        ar = self.ext_alpha_r[None, :, None] * 0.75
        bgd[:, :side] = bgd[:, :side] * (1 - al) + (L * 0.62) * al
        bgd[:, side + fw:side + fw + side] = bgd[:, side + fw:side + fw + side] * (1 - ar[:, :W - side - fw]) + \
                                             (R[:, :W - side - fw] * 0.62) * ar[:, :W - side - fw]
        bgd = bgd * self.shadow * self.vign
        bgd += self.rng.normal(0, 2.2, (H // 4, W // 4, 1)).repeat(4, 0).repeat(4, 1)  # зерно
        out = np.clip(bgd, 0, 255).astype(np.uint8)
        out[:, side:side + fw] = fg
        return out


def text(img, s, y, size, font, color=(255, 255, 255), stroke=0):
    from PIL import Image, ImageDraw, ImageFont
    im = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    d = ImageDraw.Draw(im)
    f = ImageFont.truetype(font, size)
    bw = d.textlength(s, font=f)
    d.text(((im.width - bw) / 2, y), s, font=f, fill=color, stroke_width=stroke, stroke_fill=(0, 0, 0))
    return cv2.cvtColor(np.array(im), cv2.COLOR_RGB2BGR)


def chapter_plate(W, H, side, k, total, title):
    """Подпись главы в левой панели: «ЧАСТЬ k/total» + название. Возвращает (rgb, alpha) на весь кадр."""
    from PIL import Image, ImageDraw, ImageFont
    s = H / 1080
    im = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    d = ImageDraw.Draw(im)
    f1 = ImageFont.truetype(FONT_BOLD, int(26 * s))
    f2 = ImageFont.truetype(FONT_BLACK, int(46 * s))
    x = int(side * 0.12)
    y = int(H * 0.78)
    d.rectangle([x, y, x + int(6 * s), y + int(110 * s)], fill=(255, 140, 30, 255))
    d.text((x + int(22 * s), y), f"ЧАСТЬ {k}/{total}", font=f1, fill=(255, 255, 255, 200))
    # перенос названия по ширине панели
    words, lines, cur = title.upper().split(), [], ""
    maxw = side * 0.8
    for w in words:
        t = (cur + " " + w).strip()
        if d.textlength(t, font=f2) > maxw and cur:
            lines.append(cur); cur = w
        else:
            cur = t
    lines.append(cur)
    for i, ln in enumerate(lines[:2]):
        d.text((x + int(22 * s), y + int((36 + 52 * i) * s)), ln, font=f2, fill=(255, 255, 255, 255),
               stroke_width=int(2 * s), stroke_fill=(0, 0, 0, 255))
    a = np.asarray(im).astype(np.float32)
    rgb = a[..., [2, 1, 0]]
    return rgb, (a[..., 3:4] / 255.0)


def end_screen(last, W, H, seconds, title, sub):
    """Конечная заставка: замедленный отъезд от последнего кадра, заголовок,
    две пустые зоны под элементы YouTube (видео слева, подписка справа)."""
    n = int(seconds * FPS)
    base = cv2.GaussianBlur(cv2.resize(last, (W, H)), (0, 0), H * 0.02).astype(np.float32) * 0.35
    s = H / 1080
    # зоны элементов YouTube (в Studio их ставят сюда же): видео 16:9 слева, круг подписки справа
    vid = (int(W * 0.13), int(H * 0.40), int(W * 0.13 + W * 0.36), int(H * 0.40 + W * 0.36 * 9 / 16))
    sub_c = (int(W * 0.74), int(H * 0.40 + W * 0.36 * 9 / 32))
    sub_r = int(H * 0.12)
    frames_out = []
    for i in range(n):
        t = i / FPS
        z = 1.0 + 0.04 * t / seconds
        M = cv2.getRotationMatrix2D((W / 2, H / 2), 0, z)
        f = cv2.warpAffine(base, M, (W, H), borderMode=cv2.BORDER_REFLECT)
        f = np.clip(f, 0, 255).astype(np.uint8)
        a = min(1.0, t / 0.6)
        ov = f.copy()
        cv2.rectangle(ov, vid[:2], vid[2:], (255, 255, 255), max(2, int(3 * s)))
        cv2.circle(ov, sub_c, sub_r, (255, 255, 255), max(2, int(3 * s)))
        f = cv2.addWeighted(ov, 0.25 * a, f, 1 - 0.25 * a, 0)
        frames_out.append(f)
    # текст рисуем один раз на шаблон и кладём на все кадры (дёшево)
    tpl = np.zeros((H, W, 3), np.uint8)
    tpl = text(tpl, title, int(H * 0.13), int(96 * s), FONT_BLACK, (255, 255, 255), stroke=int(5 * s))
    tpl = text(tpl, sub, int(H * 0.27), int(42 * s), FONT_BOLD, (230, 230, 230), stroke=int(3 * s))
    mask = (tpl.max(axis=2) > 0)
    for i, f in enumerate(frames_out):
        a = min(1.0, (i / FPS) / 0.6)
        f[mask] = (f[mask] * (1 - a) + tpl[mask] * a).astype(np.uint8)
    return frames_out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("inp"); ap.add_argument("out")
    ap.add_argument("--4k", dest="uhd", action="store_true")
    ap.add_argument("--end", type=float, default=20.0)
    ap.add_argument("--end-title", default="СЕЗОН 3 · СКОРО")
    ap.add_argument("--end-sub", default="Подпишись, чтобы не пропустить первую серию")
    ap.add_argument("--chapters", help="JSON [[сек, \"Название\"], ...] — подпись главы на левой панели")
    ap.add_argument("--preview", action="store_true")
    a = ap.parse_args()

    W, H = (3840, 2160) if a.uhd else (1920, 1080)
    w0, h0, dur = probe(a.inp)
    fw = int(round(H * w0 / h0 / 2)) * 2           # ширина центрального кадра без искажения пропорций
    if fw >= W:
        sys.exit("Ролик уже горизонтальный — расширять нечего.")
    limit = 20 if a.preview else None
    ext = Extender(W, H, fw)
    tmp_v = Path(a.out).with_suffix(".video.mp4")
    enc = subprocess.Popen(["ffmpeg", "-y", "-v", "error", "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{W}x{H}",
                            "-r", str(FPS), "-i", "-", "-c:v", "libx264", "-preset", "slow", "-crf", "16",
                            "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(tmp_v)], stdin=subprocess.PIPE)
    chapters = json.load(open(a.chapters, encoding="utf-8")) if a.chapters else []
    plates = [chapter_plate(W, H, ext.side, i + 1, len(chapters), t) for i, (_, t) in enumerate(chapters)]
    last = None
    n = 0
    for fr in frames(a.inp, w0, h0, limit):
        out = ext(fr)
        t = n / FPS
        for i, (t0, _) in enumerate(chapters):
            t1 = chapters[i + 1][0] if i + 1 < len(chapters) else 1e9
            if t0 <= t < t1:
                al = min(1.0, (t - t0) / 0.5, (t1 - t) / 0.3) * 0.9
                rgb, alpha = plates[i]
                m = alpha * al
                out = (out * (1 - m) + rgb * m).astype(np.uint8)
        enc.stdin.write(out.tobytes())
        last = fr; n += 1
        if n % 300 == 0:
            print(f"  {n / FPS:6.1f} с", flush=True)
    if a.end > 0 and last is not None:
        for f in end_screen(last, W, H, a.end, a.end_title, a.end_sub):
            enc.stdin.write(f.tobytes())
    enc.stdin.close(); enc.wait()

    body = n / FPS
    # звук: оригинал + тихий гул под заставкой с затуханием, громкость не трогаем (фильм уже −14 LUFS)
    af = (f"[0:a]atrim=0:{body:.3f},asetpts=N/SR/TB[a0];"
          f"sine=f=55:d={a.end:.3f},volume=0.05,afade=t=in:d=1,afade=t=out:st={max(0, a.end - 3):.3f}:d=3[a1];"
          f"[a0][a1]concat=n=2:v=0:a=1[a]") if a.end > 0 else f"[0:a]atrim=0:{body:.3f}[a]"
    subprocess.check_call(["ffmpeg", "-y", "-v", "error", "-i", a.inp, "-i", str(tmp_v), "-filter_complex", af,
                           "-map", "1:v", "-map", "[a]", "-c:v", "copy", "-c:a", "aac", "-b:a", "320k", "-ar", "48000",
                           "-movflags", "+faststart", a.out])
    tmp_v.unlink()
    print(f"Готово: {a.out}  {W}×{H}, {body + (a.end if a.end > 0 else 0):.1f} с")


if __name__ == "__main__":
    main()
