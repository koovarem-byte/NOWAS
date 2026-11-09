#!/usr/bin/env python3
"""
Оформление канала из тех же 3D-персонажей, что и сериал: аватарка, баннер, обложка фильма S2.

  python3 tools/s3blender/make_channel_art.py OUT_DIR [--samples 64] [--scale 1.0]

Размеры — по требованиям YouTube:
  avatar.png     800×800   (показывается кругом; всё важное в центральном круге)
  banner.png     2560×1440 (безопасная зона для всех устройств — центр 1546×423)
  thumb_s2.png   1280×720  (обложка длинного видео «Сезон 2 целиком»)
"""
import sys, os, argparse, math
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import bpy
import kurer3d as K
from PIL import Image, ImageDraw, ImageFont, ImageFilter

FONT_BLACK = "/usr/share/fonts/opentype/inter/InterDisplay-Black.otf"
FONT_BOLD = "/usr/share/fonts/opentype/inter/Inter-Bold.otf"


def render(path, w, h, samples):
    sc = bpy.context.scene
    K.render_settings(sc, w, h, samples, out=path)
    bpy.ops.render.render(write_still=True)


def scene_avatar(path, samples, s):
    """Крупный план курьера: голова, кепка, кусок оранжевого короба. Фон — тёплый градиент-свет."""
    sc = K.reset()
    K.world_sky((1.0, 0.45, 0.12), 0.0)
    wall = K.add("plane", "BG", (0, 3, 1.5), (6, 6, 1), rot=(math.pi / 2, 0, 0),
                 material=K.mat("BGWarm", (1.0, 0.42, 0.1), 0.9, emit=0.6, noise=0.2, noise_scale=6))
    c = K.character("Kur", "courier", (0, 0, 0), 0.25)
    cam = K.camera(lens=60, dof_target=bpy.data.objects["Kur_head"], fstop=2.0)
    cam.location = (0.25, -2.7, 1.95)
    K.look_at(cam, (0.0, 0, 1.78))
    K.key_light(cam, (0, 0, 1.8), 120)
    K.lamp((1.8, 1.5, 2.6), 400, (0.5, 0.65, 1.0), 0.1)        # холодный контровой
    K.sun((0.8, 0.0, 0.5), 1.5, (1.0, 0.8, 0.6), 1.0)
    render(path, int(1600 * s), int(1600 * s), samples)


def scene_wide(path, w, h, samples, cam_loc, target, lens=32):
    """Двор ночью: курьер и двойник с красными глазами у подъезда."""
    sc = K.reset()
    K.set_courtyard_night()
    c = K.character("Kur", "courier", (0.9, 6.0, 0), 0.35)
    d = K.character("Dbl", "double", (3.0, 6.3, 0), -0.5, eyes="red")
    cam = K.camera(lens=lens, dof_target=bpy.data.objects["Kur_head"], fstop=3.5)
    cam.location = cam_loc
    K.look_at(cam, target)
    K.key_light(cam, (1.9, 6.1, 1.6), 45, color=(1.0, 0.75, 0.55), side=0.4)
    render(path, w, h, samples)


def text_center(d, xy_center, s, font, fill, stroke=0, sfill=(0, 0, 0)):
    bw = d.textlength(s, font=font)
    asc, desc = font.getmetrics()
    d.text((xy_center[0] - bw / 2, xy_center[1] - (asc + desc) / 2), s, font=font, fill=fill,
           stroke_width=stroke, stroke_fill=sfill)


def compose_avatar(src, out):
    im = Image.open(src).convert("RGB").resize((800, 800), Image.LANCZOS)
    im.save(out)


def compose_banner(src, out):
    im = Image.open(src).convert("RGB").resize((2560, 1440), Image.LANCZOS)
    # затемнение слева под текст внутри безопасной зоны (центр 1546×423: x 507..2053, y 508..931)
    ov = Image.new("L", im.size, 0)
    od = ImageDraw.Draw(ov)
    for x in range(0, 1500):
        od.line([(x, 0), (x, 1440)], fill=int(170 * max(0, 1 - x / 1500) ** 1.2))
    im = Image.composite(Image.new("RGB", im.size, (0, 0, 0)), im, ov)
    d = ImageDraw.Draw(im)
    f1 = ImageFont.truetype(FONT_BLACK, 150)
    f2 = ImageFont.truetype(FONT_BOLD, 50)
    x0, y0 = 560, 560
    d.text((x0, y0), "КУРЬЕР", font=f1, fill=(255, 255, 255), stroke_width=6, stroke_fill=(0, 0, 0))
    d.rectangle([x0, y0 + 190, x0 + 14, y0 + 300], fill=(255, 110, 20))
    d.rectangle([x0, y0 + 190, x0 + 14, y0 + 360], fill=(255, 110, 20))
    d.text((x0 + 36, y0 + 190), "Сериал о том, как обманывают", font=f2, fill=(255, 255, 255))
    d.text((x0 + 36, y0 + 250), "у твоей двери.", font=f2, fill=(255, 255, 255))
    d.text((x0 + 36, y0 + 310), "И как не попасться.", font=f2, fill=(255, 200, 150))
    im.save(out)


def compose_thumb(src, out):
    im = Image.open(src).convert("RGB").resize((1280, 720), Image.LANCZOS)
    # насыщенность и контраст — обложку видно в мелком размере
    from PIL import ImageEnhance
    im = ImageEnhance.Color(im).enhance(1.25)
    im = ImageEnhance.Contrast(im).enhance(1.1)
    d = ImageDraw.Draw(im)
    f1 = ImageFont.truetype(FONT_BLACK, 118)
    f2 = ImageFont.truetype(FONT_BLACK, 64)
    # максимум 3 слова на обложке: читается с телефона
    d.text((48, 40), "ДВОЙНИК", font=f1, fill=(255, 255, 255), stroke_width=8, stroke_fill=(0, 0, 0))
    d.text((52, 178), "У МОЕЙ ДВЕРИ", font=f2, fill=(255, 120, 30), stroke_width=6, stroke_fill=(0, 0, 0))
    # плашка «весь сезон»
    d.rounded_rectangle([48, 600, 420, 676], 16, fill=(230, 30, 40))
    d.text((70, 607), "ВЕСЬ СЕЗОН 2", font=ImageFont.truetype(FONT_BLACK, 50), fill=(255, 255, 255))
    im.save(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("--samples", type=int, default=64)
    ap.add_argument("--scale", type=float, default=1.0, help="0.5 — черновик в 2 раза быстрее")
    ap.add_argument("--only", default="avatar,banner,thumb")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    raw = os.path.join(a.out, "raw")
    os.makedirs(raw, exist_ok=True)
    s = a.scale
    only = a.only.split(",")
    if "avatar" in only:
        scene_avatar(os.path.join(raw, "avatar.png"), a.samples, s)
        compose_avatar(os.path.join(raw, "avatar.png"), os.path.join(a.out, "avatar.png"))
    # баннер: герои справа, слева место под текст; всё в безопасной зоне по высоте
    if "banner" in only:
        scene_wide(os.path.join(raw, "banner.png"), int(2560 * s), int(1440 * s), a.samples,
               (-0.9, -1.6, 1.45), (-0.2, 6.2, 1.55), lens=34)
        compose_banner(os.path.join(raw, "banner.png"), os.path.join(a.out, "banner.png"))
    if "thumb" in only:
        scene_wide(os.path.join(raw, "thumb.png"), int(1920 * s), int(1080 * s), a.samples,
               (0.9, 2.9, 1.6), (2.4, 6.2, 1.7), lens=30)
        compose_thumb(os.path.join(raw, "thumb.png"), os.path.join(a.out, "thumb_s2.png"))
    print("готово:", a.out)


if __name__ == "__main__":
    main()
