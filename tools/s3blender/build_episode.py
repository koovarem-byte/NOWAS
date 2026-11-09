#!/usr/bin/env python3
"""
Рендер 3D-планов серии по покадровому плану (s3/episodes/s3eN.json).

Каждый 3D-план собирается с нуля: локация → персонажи → реквизит → свет → камера с движением.
Один и тот же план рендерится в двух форматах из ОДНОЙ камеры (sensor fit AUTO):
  long  1920×1080 — для длинного ролика/фильма сезона (горизонталь, без размытых полос);
  short 1080×1920 — для Shorts/Reels/TikTok.
Вертикальный кадр видит столько же по вертикали, сколько горизонтальный по ширине, — герой в центре
в обоих, поэтому ставим персонажей по центру кадра.

Длительность плана: если есть озвучка s3/voice/<ep>_<id>_<voice>.mp3 — по ней (+0.3 с хвост),
иначе из поля dur.

  python3 tools/s3blender/build_episode.py s3/episodes/s3e1.json --formats long,short
  # черновик здесь (без видеокарты): каждый 3-й кадр, 1/3 разрешения, 12 сэмплов
  python3 tools/s3blender/build_episode.py s3/episodes/s3e1.json --draft
  # только некоторые планы
  python3 tools/s3blender/build_episode.py s3/episodes/s3e1.json --only 01,05

Финал у тебя на ПК: --engine CYCLES (видеокарта находится сама) или --engine BLENDER_EEVEE (быстрее).
"""
import sys, os, json, argparse, subprocess, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import bpy
import kurer3d as K

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
FPS = 30
KINDS = {"courier": "courier", "double": "double", "babushka": "babushka", "collector": "collector", "operator": "operator"}


def voice_path(ep, shot):
    return os.path.join(ROOT, "s3", "voice", f"{ep}_{shot['id']}_{shot['voice']}.mp3")


def shot_duration(ep, shot):
    p = voice_path(ep, shot)
    if os.path.exists(p):
        d = float(subprocess.check_output(["ffprobe", "-v", "error", "-show_entries", "format=duration",
                                           "-of", "csv=p=0", p]).decode().strip())
        return d + 0.3
    return float(shot["dur"])


def build_shot(shot, frames):
    sc = K.reset()
    K.SETS[shot["set"]]()
    actors = []
    for i, c in enumerate(shot.get("cast", [])):
        root = K.character(c["name"], KINDS[c["who"]], tuple(c["at"]), c.get("rot", 0.0),
                           eyes="red" if c["who"] == "double" else "normal")
        if c.get("lie"):
            root.rotation_euler[0] = -1.5708
            root.location[2] += 0.25
        if c.get("walk_to"):
            K.walk(root, frames, tuple(c["at"]), tuple(c["walk_to"]), seed=i)
        else:
            K.idle_motion(root, frames, seed=i)
        if c.get("arm_up"):
            sh = K.raise_arm(root, c["arm_up"], 1, 12, -1.6)
            K.phone(c["name"] + "_ph", (0, -0.05, -0.62), (0.3, 0, 0), parent=sh)
        actors.append(root)
    for j, p in enumerate(shot.get("props", [])):
        if "phone" in p:
            K.phone(f"Prop{j}", tuple(p["phone"]), tuple(p.get("rot", (0, 0, 0))),
                    tuple(p.get("screen", (0.2, 0.45, 1.0))), p.get("glow", 3.0))
    cm = shot["cam"]
    focus = None
    if actors:
        focus = next((o for o in bpy.data.objects if o.name.endswith("_head") and o.name.startswith(actors[0].name)), None)
    cam = K.camera(lens=cm["lens"][0], dof_target=focus, fstop=2.8)
    hits = [int(h * FPS) for h in shot.get("hits", [])]
    K.camera_move(cam, frames, cm["from"], cm["to"], cm["look"], cm["look_to"], cm["lens"][0], cm["lens"][1],
                  shake_at=hits, shake_amp=0.05)
    K.key_light(cam, cm["look"], shot.get("key", 50), side=0.5)   # "key": 0..50 — 5 для тёмных сцен
    return sc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("episode")
    ap.add_argument("--formats", default="long,short")
    ap.add_argument("--only", default="")
    ap.add_argument("--draft", action="store_true")
    ap.add_argument("--samples", type=int, default=48)
    ap.add_argument("--engine", default="CYCLES", choices=["CYCLES", "BLENDER_EEVEE"])
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    ep = json.load(open(a.episode, encoding="utf-8"))
    name = ep["episode"]
    out_root = a.out or os.path.join(ROOT, "s3", "render", name)
    only = set(x for x in a.only.split(",") if x)
    sizes = {"long": (1920, 1080), "short": (1080, 1920)}
    scale, step, samples = (1 / 3, 3, 12) if a.draft else (1.0, 1, a.samples)
    manifest = {"episode": name, "fps": FPS, "step": step, "shots": []}
    for shot in ep["shots"]:
        dur = shot_duration(name, shot)
        rec = {"id": shot["id"], "dur": dur, "kind": "3d" if "set" in shot else "screen"}
        manifest["shots"].append(rec)
        if "set" not in shot or (only and shot["id"] not in only):
            continue
        frames = int(round(dur * FPS))
        for fmt in a.formats.split(","):
            sc = build_shot(shot, frames)
            dev = K.gpu_if_available(sc, a.engine)
            w, h = sizes[fmt]
            d = os.path.join(out_root, fmt, shot["id"])
            os.makedirs(d, exist_ok=True)
            K.render_settings(sc, int(w * scale) // 2 * 2, int(h * scale) // 2 * 2, samples,
                              out=os.path.join(d, "f_"), frames=frames)
            sc.frame_step = step
            t = time.time()
            bpy.ops.render.render(animation=True)
            print(f"[{shot['id']} {fmt}] {frames} кадров ({frames // step} отрендерено) за {time.time() - t:.0f} с на {dev}", flush=True)
    with open(os.path.join(out_root, "manifest.json"), "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=1)


if __name__ == "__main__":
    main()
