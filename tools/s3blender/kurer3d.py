"""
Библиотека 3D-сцен «Курьер» для Blender 5.x (bpy). Всё процедурное: никаких внешних ассетов,
поэтому сцену можно собрать и здесь (pip install bpy), и у тебя в Blender одной командой.

Стиль — как в S2 (части 4–5), но с «кино-светом» по разбору рефов:
  * персонажи: большая круглая голова, чёрная обводка (inverted hull), простые глаза;
  * мир фактурный (кирпич, плитка, шум), жёсткие тени от солнца/лампы, дымка;
  * тёплый свет против холодных теней, насыщенность ~0.45.

Используется из build_episode.py, make_channel_art.py.
"""
import math
import bpy
from mathutils import Vector

# ───────────────────────── база ─────────────────────────

def reset():
    bpy.ops.wm.read_factory_settings(use_empty=True)
    sc = bpy.context.scene
    sc.render.engine = "CYCLES"
    sc.cycles.device = "CPU"
    sc.cycles.samples = 24
    sc.cycles.use_denoising = True
    sc.render.fps = 30
    sc.view_settings.view_transform = "AgX"
    sc.view_settings.look = "AgX - Medium High Contrast"
    sc.render.film_transparent = False
    w = bpy.data.worlds.new("W")
    sc.world = w
    w.use_nodes = True
    return sc


def gpu_if_available(sc, engine="CYCLES"):
    """У тебя на ПК: CUDA/OptiX/HIP/Metal, если есть. Здесь — CPU."""
    sc.render.engine = engine
    if engine != "CYCLES":
        return "EEVEE"
    try:
        prefs = bpy.context.preferences.addons["cycles"].preferences
        for kind in ("OPTIX", "CUDA", "HIP", "METAL", "ONEAPI"):
            try:
                prefs.compute_device_type = kind
                prefs.get_devices()
                devs = [d for d in prefs.devices if d.type == kind]
                if devs:
                    for d in prefs.devices:
                        d.use = d.type == kind
                    sc.cycles.device = "GPU"
                    return kind
            except TypeError:
                continue
    except Exception:
        pass
    sc.cycles.device = "CPU"
    return "CPU"


def mat(name, color, rough=0.6, emit=0.0, noise=0.0, noise_scale=8.0, metallic=0.0):
    """Principled + опциональный шум цвета (фактура: штукатурка, ткань)."""
    m = bpy.data.materials.get(name)
    if m:
        return m
    m = bpy.data.materials.new(name)
    m.use_nodes = True
    nt = m.node_tree
    b = nt.nodes["Principled BSDF"]
    b.inputs["Base Color"].default_value = (*color, 1)
    b.inputs["Roughness"].default_value = rough
    b.inputs["Metallic"].default_value = metallic
    if emit:
        b.inputs["Emission Color"].default_value = (*color, 1)
        b.inputs["Emission Strength"].default_value = emit
    if noise:
        n = nt.nodes.new("ShaderNodeTexNoise")
        n.inputs["Scale"].default_value = noise_scale
        n.inputs["Detail"].default_value = 6
        mix = nt.nodes.new("ShaderNodeMix")
        mix.data_type = "RGBA"
        mix.inputs["Factor"].default_value = noise
        mix.inputs[6].default_value = (*color, 1)
        mix.inputs[7].default_value = (color[0] * 0.55, color[1] * 0.55, color[2] * 0.55, 1)
        nt.links.new(n.outputs["Fac"], mix.inputs["Factor"])
        nt.links.new(mix.outputs[2], b.inputs["Base Color"])
    return m


def brick_mat(name, c1, c2, mortar=(0.55, 0.53, 0.5), scale=4.0):
    m = bpy.data.materials.get(name)
    if m:
        return m
    m = bpy.data.materials.new(name)
    m.use_nodes = True
    nt = m.node_tree
    b = nt.nodes["Principled BSDF"]
    br = nt.nodes.new("ShaderNodeTexBrick")
    br.inputs["Color1"].default_value = (*c1, 1)
    br.inputs["Color2"].default_value = (*c2, 1)
    br.inputs["Mortar"].default_value = (*mortar, 1)
    br.inputs["Scale"].default_value = scale
    tc = nt.nodes.new("ShaderNodeTexCoord")          # координаты объекта: кирпич одного размера на любой стене
    nt.links.new(tc.outputs["Object"], br.inputs["Vector"])
    nt.links.new(br.outputs["Color"], b.inputs["Base Color"])
    b.inputs["Roughness"].default_value = 0.85
    bump = nt.nodes.new("ShaderNodeBump")
    bump.inputs["Strength"].default_value = 0.4
    nt.links.new(br.outputs["Fac"], bump.inputs["Height"])
    nt.links.new(bump.outputs["Normal"], b.inputs["Normal"])
    return m


def tile_mat(name, c1, c2, scale=6.0):
    m = bpy.data.materials.get(name)
    if m:
        return m
    m = bpy.data.materials.new(name)
    m.use_nodes = True
    nt = m.node_tree
    b = nt.nodes["Principled BSDF"]
    ch = nt.nodes.new("ShaderNodeTexChecker")
    ch.inputs["Color1"].default_value = (*c1, 1)
    ch.inputs["Color2"].default_value = (*c2, 1)
    ch.inputs["Scale"].default_value = scale
    nt.links.new(ch.outputs["Color"], b.inputs["Base Color"])
    b.inputs["Roughness"].default_value = 0.35
    return m


OUTLINE = None


def outline(obj, width=0.025):
    """Чёрная обводка: инвертированная оболочка (как в мультах S2)."""
    global OUTLINE
    try:
        alive = OUTLINE is not None and OUTLINE.name in bpy.data.materials
    except ReferenceError:             # после reset() старый материал удалён
        alive = False
    if not alive:
        # Cycles не знает backface culling, поэтому: ближние (вывернутые) грани оболочки прозрачны,
        # дальние — чёрные и видны только кольцом по силуэту.
        OUTLINE = bpy.data.materials.new("Outline")
        OUTLINE.use_nodes = True
        OUTLINE.use_backface_culling = True
        nt = OUTLINE.node_tree
        nt.nodes.remove(nt.nodes["Principled BSDF"])
        geo = nt.nodes.new("ShaderNodeNewGeometry")
        em = nt.nodes.new("ShaderNodeEmission")
        em.inputs["Color"].default_value = (0.005, 0.005, 0.008, 1)
        tr = nt.nodes.new("ShaderNodeBsdfTransparent")
        mix = nt.nodes.new("ShaderNodeMixShader")
        lp = nt.nodes.new("ShaderNodeLightPath")
        inv = nt.nodes.new("ShaderNodeMath"); inv.operation = "SUBTRACT"; inv.inputs[0].default_value = 1.0
        nt.links.new(lp.outputs["Is Camera Ray"], inv.inputs[1])
        mx = nt.nodes.new("ShaderNodeMath"); mx.operation = "MAXIMUM"
        nt.links.new(geo.outputs["Backfacing"], mx.inputs[0])
        nt.links.new(inv.outputs[0], mx.inputs[1])        # для теней/отражений оболочки нет
        nt.links.new(mx.outputs[0], mix.inputs[0])
        nt.links.new(em.outputs[0], mix.inputs[1])
        nt.links.new(tr.outputs[0], mix.inputs[2])
        nt.links.new(mix.outputs[0], nt.nodes["Material Output"].inputs["Surface"])
    obj.data.materials.append(OUTLINE)
    mod = obj.modifiers.new("outline", "SOLIDIFY")
    sc = (abs(obj.scale[0]) + abs(obj.scale[1]) + abs(obj.scale[2])) / 3 or 1.0
    mod.thickness = -width / sc          # толщина в метрах мира, а не в локальных единицах
    mod.offset = 1
    mod.use_flip_normals = True
    mod.material_offset = len(obj.data.materials) - 1
    return mod


def add(kind, name, loc=(0, 0, 0), scale=(1, 1, 1), rot=(0, 0, 0), material=None, parent=None, smooth=True, **kw):
    ops = {
        "sphere": lambda: bpy.ops.mesh.primitive_uv_sphere_add(segments=48, ring_count=24, **kw),
        "cube": lambda: bpy.ops.mesh.primitive_cube_add(**kw),
        "cyl": lambda: bpy.ops.mesh.primitive_cylinder_add(vertices=48, **kw),
        "plane": lambda: bpy.ops.mesh.primitive_plane_add(**kw),
        "cone": lambda: bpy.ops.mesh.primitive_cone_add(vertices=48, **kw),
        "torus": lambda: bpy.ops.mesh.primitive_torus_add(**kw),
    }
    ops[kind]()
    o = bpy.context.active_object
    o.name = name
    o.location = loc
    o.scale = scale
    o.rotation_euler = rot
    if material:
        o.data.materials.append(material)
    if smooth and kind in ("sphere", "cyl", "torus", "cone"):
        bpy.ops.object.shade_smooth()
    if parent:
        o.parent = parent
    return o


def bevel(o, w=0.03, seg=3):
    m = o.modifiers.new("bev", "BEVEL")
    m.width = w
    m.segments = seg
    return m


def empty(name, loc=(0, 0, 0), parent=None):
    o = bpy.data.objects.new(name, None)
    bpy.context.scene.collection.objects.link(o)
    o.location = loc
    if parent:
        o.parent = parent
    return o


# ───────────────────────── персонажи ─────────────────────────

SKIN = (0.93, 0.62, 0.42)
NAVY = (0.05, 0.07, 0.17)
ORANGE = (1.0, 0.33, 0.03)


def character(name, kind="courier", loc=(0, 0, 0), rot_z=0.0, eyes="normal"):
    """kind: courier | double | babushka | collector | operator.
    Возвращает корневой empty; дочерние: head, body, eye_L/R (для анимации взгляда), arm_L/R.
    Рост ~1.75 м, голова ~0.62 м (большая, как в S2)."""
    root = empty(name, loc)
    root.rotation_euler[2] = rot_z
    skin = mat("Skin", SKIN, 0.55)
    black = mat("Black", (0.015, 0.015, 0.02), 0.5)
    white = mat("EyeWhite", (0.95, 0.95, 0.95), 0.25)
    body_col = {"courier": NAVY, "double": NAVY, "babushka": (0.85, 0.2, 0.45),
                "collector": (0.06, 0.06, 0.07), "operator": (0.2, 0.25, 0.3)}[kind]
    body_m = mat(f"Body_{kind}", body_col, 0.8, noise=0.25, noise_scale=40)

    hip = empty(f"{name}_hip", (0, 0, 0.0), root)
    # ноги
    for sx in (-1, 1):
        leg = add("cyl", f"{name}_leg{sx}", (0.11 * sx, 0, 0.42), (0.08, 0.08, 0.42), material=black if kind != "babushka" else skin, parent=hip)
        outline(leg, 0.012)
        shoe = add("sphere", f"{name}_shoe{sx}", (0.11 * sx, -0.05, 0.04), (0.1, 0.16, 0.06), material=black, parent=hip)
    # тело
    if kind == "babushka":
        body = add("cone", f"{name}_body", (0, 0, 0.95), (0.36, 0.32, 0.62), material=body_m, parent=hip)
    else:
        body = add("cyl", f"{name}_body", (0, 0, 1.08), (0.27, 0.2, 0.36), material=body_m, parent=hip)
        bevel(body, 0.08, 4)
    outline(body, 0.015)
    # руки
    arms = []
    for sx in (-1, 1):
        sh = empty(f"{name}_shoulder{sx}", (0.31 * sx, 0, 1.36), hip)
        arm = add("cyl", f"{name}_arm{sx}", (0, 0, -0.27), (0.07, 0.07, 0.27), material=body_m, parent=sh)
        outline(arm, 0.012)
        hand = add("sphere", f"{name}_hand{sx}", (0, 0, -0.56), (0.08, 0.08, 0.08), material=skin, parent=sh)
        outline(hand, 0.012)
        sh.rotation_euler[1] = -0.12 * sx
        arms.append(sh)
    # голова
    neck = empty(f"{name}_neck", (0, 0, 1.48), hip)
    head = add("sphere", f"{name}_head", (0, 0, 0.32), (0.34, 0.31, 0.33), material=skin if kind != "collector" else black, parent=neck)
    outline(head, 0.018)
    if kind == "collector":
        # капюшон, лица не видно — только два тусклых огонька
        hood = add("sphere", f"{name}_hood", (0, 0.03, 0.36), (0.39, 0.37, 0.4), material=body_m, parent=neck)
        for sx in (-1, 1):
            add("sphere", f"{name}_glow{sx}", (0.1 * sx, -0.3, 0.33), (0.025, 0.01, 0.018), material=mat("Glow", (1, 0.1, 0.05), emit=8), parent=neck)
        return root
    # глаза
    pupil_m = mat("PupilRed", (1.0, 0.1, 0.12), 0.3, emit=2.5) if eyes == "red" else black
    for sx in (-1, 1):
        ew = add("sphere", f"{name}_eyeW{sx}", (0.12 * sx, -0.27, 0.38), (0.085, 0.05, 0.085), material=white, parent=neck)
        outline(ew, 0.008)
        ep = add("sphere", f"{name}_eye{sx}", (0.12 * sx, -0.315, 0.37), (0.04, 0.02, 0.045), material=pupil_m, parent=neck)
        if kind == "double":
            # брови — только у двойника, злые, под козырьком
            br = add("cube", f"{name}_brow{sx}", (0.12 * sx, -0.3, 0.47), (0.075, 0.015, 0.016), material=black, parent=neck)
            br.rotation_euler[1] = 0.35 * sx
    # нос, рот, щёки
    add("sphere", f"{name}_nose", (0, -0.32, 0.28), (0.045, 0.04, 0.045), material=skin, parent=neck)
    add("cube", f"{name}_mouth", (0, -0.31, 0.15), (0.07, 0.01, 0.014), material=mat("Mouth", (0.25, 0.05, 0.04)), parent=neck)
    cheek = mat("Cheek", (1.0, 0.35, 0.35), 0.5)
    for sx in (-1, 1):
        add("sphere", f"{name}_cheek{sx}", (0.21 * sx, -0.24, 0.2), (0.05, 0.02, 0.035), material=cheek, parent=neck)
    if kind in ("courier", "double"):
        cap = add("sphere", f"{name}_cap", (0, 0.02, 0.5), (0.355, 0.335, 0.2), material=black, parent=neck)
        outline(cap, 0.012)
        brim = add("cyl", f"{name}_brim", (0, -0.28, 0.47), (0.24, 0.17, 0.02), material=black, parent=neck)
        brim.rotation_euler[0] = 0.18
        hair = add("sphere", f"{name}_hair", (0, 0.07, 0.36), (0.345, 0.3, 0.28), material=black, parent=neck)
        # короб
        box = add("cube", f"{name}_box", (0, 0.36, 1.12), (0.3, 0.17, 0.3), material=mat("BoxOrange", ORANGE, 0.55), parent=hip)
        bevel(box, 0.03)
        outline(box, 0.014)
        add("cube", f"{name}_boxlogo", (0, 0.535, 1.2), (0.12, 0.005, 0.06), material=mat("BoxLogo", (1, 0.85, 0.5), 0.5), parent=hip)
    elif kind == "babushka":
        hair = add("sphere", f"{name}_hair", (0, 0.06, 0.45), (0.35, 0.31, 0.25), material=mat("GreyHair", (0.85, 0.85, 0.82), 0.7), parent=neck)
        bun = add("sphere", f"{name}_bun", (0, 0.12, 0.72), (0.14, 0.14, 0.12), material=mat("GreyHair", (0.85, 0.85, 0.82), 0.7), parent=neck)
        outline(bun, 0.01)
        for sx in (-1, 1):   # очки
            g = add("torus", f"{name}_glass{sx}", (0.12 * sx, -0.33, 0.38), (0.09, 0.09, 0.09), rot=(math.pi / 2, 0, 0),
                    material=black, major_radius=1, minor_radius=0.12)
    elif kind == "operator":
        hair = add("sphere", f"{name}_hair", (0, 0.07, 0.4), (0.35, 0.31, 0.27), material=mat("Brown", (0.2, 0.1, 0.05)), parent=neck)
        hs = add("torus", f"{name}_headset", (0, 0, 0.35), (0.36, 0.36, 0.36), rot=(0, math.pi / 2, 0), material=black,
                 major_radius=1, minor_radius=0.04, parent=neck)
    return root


def phone(name, loc=(0, 0, 0), rot=(0, 0, 0), screen=(0.2, 0.45, 1.0), glow=3.0, parent=None):
    p = add("cube", name, loc, (0.04, 0.005, 0.08), rot=rot, material=mat("PhoneBody", (0.02, 0.02, 0.025), 0.3), parent=parent)
    bevel(p, 0.008)
    s = add("plane", name + "_scr", (0, -1.05, 0), (0.9, 1, 0.92), rot=(math.pi / 2, 0, 0),
            material=mat(f"Screen_{name}", screen, 0.2, emit=glow), parent=p)
    return p


# ───────────────────────── локации ─────────────────────────

def world_sky(color=(0.02, 0.03, 0.08), strength=0.6, haze=0.0):
    w = bpy.context.scene.world
    bg = w.node_tree.nodes["Background"]
    bg.inputs["Color"].default_value = (*color, 1)
    bg.inputs["Strength"].default_value = strength
    if haze:
        vol = w.node_tree.nodes.new("ShaderNodeVolumePrincipled")
        vol.inputs["Density"].default_value = haze
        w.node_tree.links.new(vol.outputs[0], w.node_tree.nodes["World Output"].inputs["Volume"])


def haze_box(size=30, density=0.02, color=(0.6, 0.7, 1.0)):
    """Дымка: объёмный куб вокруг сцены — лучи и воздушная перспектива."""
    m = bpy.data.materials.new("Haze")
    m.use_nodes = True
    nt = m.node_tree
    nt.nodes.remove(nt.nodes["Principled BSDF"])
    v = nt.nodes.new("ShaderNodeVolumePrincipled")
    v.inputs["Density"].default_value = density
    v.inputs["Color"].default_value = (*color, 1)
    nt.links.new(v.outputs[0], nt.nodes["Material Output"].inputs["Volume"])
    b = add("cube", "Haze", (0, 0, size / 2 - 0.5), (size / 2, size / 2, size / 2), material=m)
    b.visible_shadow = False
    return b


def sun(rot=(0.9, 0.2, 0.6), strength=4.0, color=(1.0, 0.75, 0.5), angle=0.6):
    bpy.ops.object.light_add(type="SUN", rotation=rot)
    s = bpy.context.active_object
    s.data.energy = strength
    s.data.color = color
    s.data.angle = math.radians(angle)   # маленький угол → жёсткие тени
    return s


def lamp(loc, power=300, color=(1.0, 0.7, 0.4), size=0.05, kind="POINT"):
    bpy.ops.object.light_add(type=kind, location=loc)
    l = bpy.context.active_object
    l.data.energy = power
    l.data.color = color
    if kind in ("POINT", "SPOT"):
        l.data.shadow_soft_size = size
    return l


def set_courtyard_night():
    """Ночной двор у панельки: асфальт, кирпичная стена с окнами, фонарь, луна."""
    world_sky((0.015, 0.025, 0.07), 0.8)
    add("plane", "Ground", (0, 0, 0), (30, 30, 1), material=mat("Asphalt", (0.08, 0.08, 0.09), 0.9, noise=0.5, noise_scale=60))
    wall = add("cube", "Building", (0, 9, 6), (14, 1, 6), material=brick_mat("BrickNight", (0.35, 0.18, 0.12), (0.3, 0.15, 0.1), scale=1.6))
    win_on = mat("WinOn", (1.0, 0.7, 0.35), 0.3, emit=6)
    win_off = mat("WinOff", (0.03, 0.04, 0.07), 0.2)
    import random
    rnd = random.Random(3)
    for fx in range(-6, 7):
        for fz in range(5):
            add("cube", f"win{fx}_{fz}", (fx * 2.0, 7.95, 1.6 + fz * 2.3), (0.45, 0.05, 0.6),
                material=win_on if rnd.random() < 0.3 else win_off)
    # подъезд
    add("cube", "Door", (2, 7.9, 1.1), (0.7, 0.1, 1.1), material=mat("DoorMetal", (0.12, 0.2, 0.25), 0.5, noise=0.3, noise_scale=20))
    add("cube", "Canopy", (2, 7.4, 2.35), (1.2, 0.6, 0.06), material=mat("Concrete", (0.4, 0.4, 0.4), 0.9, noise=0.4))
    lamp((2, 7.3, 2.2), 120, (1, 0.6, 0.3), 0.05)
    # фонарь
    add("cyl", "Pole", (-3, 3, 2.5), (0.06, 0.06, 2.5), material=mat("Pole", (0.1, 0.1, 0.1)))
    lamp((-3, 3, 4.9), 800, (1.0, 0.65, 0.35), 0.1)
    # луна-контровой
    sun((0.6, -0.3, -2.4), 0.6, (0.5, 0.65, 1.0), 0.5)
    # лавка и урна для фактуры
    add("cube", "Bench", (-1.5, 6.2, 0.45), (1.0, 0.25, 0.04), material=mat("Wood", (0.35, 0.18, 0.08), 0.7, noise=0.5, noise_scale=30))
    add("cyl", "Bin", (0.3, 6.6, 0.35), (0.22, 0.22, 0.35), material=mat("BinGreen", (0.05, 0.25, 0.12)))
    haze_box(40, 0.012, (0.6, 0.7, 1.0))


def set_bedroom_dark():
    """Тёмная комната в 6 утра: кровать, тумба, телефон светит синим, полоса рассвета из окна."""
    world_sky((0.005, 0.008, 0.02), 0.3)
    add("plane", "Floor", (0, 0, 0), (6, 6, 1), material=mat("Laminate", (0.25, 0.15, 0.08), 0.5, noise=0.5, noise_scale=12))
    add("cube", "WallBack", (0, 3, 1.5), (6, 0.05, 1.5), material=mat("Wallpaper", (0.35, 0.4, 0.45), 0.9, noise=0.3, noise_scale=30))
    add("cube", "WallLeft", (-3, 0, 1.5), (0.05, 6, 1.5), material=mat("Wallpaper", (0.35, 0.4, 0.45)))
    bed = add("cube", "Bed", (0.4, 1.6, 0.3), (1.0, 1.2, 0.3), material=mat("Blanket", (0.2, 0.3, 0.5), 0.9, noise=0.4, noise_scale=15))
    bevel(bed, 0.12, 4)
    add("cube", "Pillow", (0.4, 2.5, 0.68), (0.6, 0.25, 0.1), material=mat("Pillow", (0.85, 0.85, 0.8), 0.9))
    add("cube", "Nightstand", (-1.0, 2.4, 0.3), (0.3, 0.3, 0.3), material=mat("Wood", (0.35, 0.18, 0.08)))
    # окно: холодная полоса рассвета
    lamp((-2.9, 1.0, 1.6), 150, (0.5, 0.6, 1.0), 0.0, "AREA").scale = (0.1, 1.2, 0.8)


def set_elevator():
    """Кабина лифта: металл, кнопки, лампа дневного света сверху — жёсткий верхний свет."""
    world_sky((0, 0, 0), 0)
    metal = mat("ElevMetal", (0.55, 0.55, 0.52), 0.35, metallic=0.8, noise=0.3, noise_scale=80)
    add("plane", "EFloor", (0, 0, 0), (1.1, 1.1, 1), material=mat("Rubber", (0.1, 0.1, 0.1), 0.9, noise=0.4, noise_scale=90))
    for nm, loc, sc in (("EBack", (0, 1.1, 1.2), (1.1, 0.03, 1.2)), ("EL", (-1.1, 0, 1.2), (0.03, 1.1, 1.2)),
                        ("ER", (1.1, 0, 1.2), (0.03, 1.1, 1.2)), ("ETop", (0, 0, 2.4), (1.1, 1.1, 0.03))):
        add("cube", nm, loc, sc, material=metal)
    add("cube", "Panel", (1.06, -0.5, 1.2), (0.02, 0.15, 0.35), material=mat("PanelDark", (0.05, 0.05, 0.05), 0.3))
    for i in range(9):
        add("cyl", f"Btn{i}", (1.04, -0.55 + (i % 2) * 0.1, 0.95 + (i // 2) * 0.11), (0.03, 0.03, 0.01),
            rot=(0, math.pi / 2, 0), material=mat("BtnLit" if i == 6 else "Btn", (1, 0.4, 0.2) if i == 6 else (0.8, 0.8, 0.8), emit=4 if i == 6 else 0))
    lamp((0, 0, 2.35), 120, (0.85, 0.95, 1.0), 0.3, "AREA").scale = (1.2, 0.3, 1)


def set_server_corridor():
    """Пустой серверный коридор: стойки с мигающими огнями, одна лампа мигает."""
    world_sky((0, 0, 0), 0)
    add("plane", "SFloor", (0, 0, 0), (2, 20, 1), material=tile_mat("SFloorTile", (0.15, 0.16, 0.18), (0.11, 0.12, 0.14), 40))
    add("plane", "SCeil", (0, 0, 3), (2, 20, 1), rot=(math.pi, 0, 0), material=mat("Ceil", (0.2, 0.2, 0.22)))
    rack = mat("Rack", (0.03, 0.03, 0.035), 0.4, metallic=0.5)
    led_g = mat("LedG", (0.1, 1, 0.3), emit=10)
    led_r = mat("LedR", (1, 0.1, 0.05), emit=10)
    import random
    rnd = random.Random(5)
    for side in (-1, 1):
        for i in range(10):
            y = -16 + i * 3.2
            add("cube", f"Rack{side}_{i}", (1.5 * side, y, 1.1), (0.45, 1.4, 1.1), material=rack)
            for k in range(6):
                add("cube", f"Led{side}_{i}_{k}", (1.04 * side, y - 1 + rnd.random() * 2, 0.4 + rnd.random() * 1.6),
                    (0.01, 0.02, 0.01), material=led_r if rnd.random() < 0.15 else led_g)
    for i in range(5):
        lamp((0, -14 + i * 7, 2.9), 200, (0.7, 0.85, 1.0), 0.0, "AREA").scale = (0.3, 1.5, 1)
    haze_box(40, 0.03, (0.7, 0.8, 1.0))


def set_stairwell():
    """Лестничная площадка панельки: зелёная краска до середины стены, кафель, тёплая лампа, двери квартир."""
    world_sky((0, 0, 0), 0)
    add("plane", "LFloor", (0, 0, 0), (4, 4, 1), material=tile_mat("LTile", (0.45, 0.25, 0.15), (0.35, 0.2, 0.12), 16))
    wall_low = mat("PaintGreen", (0.12, 0.42, 0.38), 0.4, noise=0.25, noise_scale=25)
    wall_hi = mat("Whitewash", (0.75, 0.72, 0.62), 0.9, noise=0.3, noise_scale=20)
    add("cube", "BackLow", (0, 3, 0.75), (4, 0.05, 0.75), material=wall_low)
    add("cube", "BackHi", (0, 3, 2.25), (4, 0.05, 0.75), material=wall_hi)
    add("cube", "LeftLow", (-3, 0, 0.75), (0.05, 4, 0.75), material=wall_low)
    add("cube", "LeftHi", (-3, 0, 2.25), (0.05, 4, 0.75), material=wall_hi)
    door_m = mat("DoorBrown", (0.3, 0.12, 0.05), 0.6, noise=0.3, noise_scale=10)
    for x, nm in ((-1.2, "DoorA"), (1.4, "DoorB")):
        d = add("cube", nm, (x, 2.93, 1.05), (0.45, 0.04, 1.05), material=door_m)
        bevel(d, 0.02)
        add("sphere", nm + "_knob", (x + 0.33, 2.86, 1.0), (0.035, 0.035, 0.035), material=mat("Brass", (0.8, 0.6, 0.2), 0.3, metallic=1))
    lamp((0, 1.5, 2.9), 250, (1.0, 0.7, 0.4), 0.04)
    haze_box(20, 0.02, (1.0, 0.85, 0.7))


SETS = {
    "courtyard_night": set_courtyard_night,
    "bedroom_dark": set_bedroom_dark,
    "elevator": set_elevator,
    "server_corridor": set_server_corridor,
    "stairwell": set_stairwell,
}


# ───────────────────────── камера ─────────────────────────

def camera(name="Cam", lens=35, dof_target=None, fstop=2.8):
    cam_data = bpy.data.cameras.new(name)
    cam_data.lens = lens
    cam = bpy.data.objects.new(name, cam_data)
    bpy.context.scene.collection.objects.link(cam)
    bpy.context.scene.camera = cam
    if dof_target is not None:
        cam_data.dof.use_dof = True
        cam_data.dof.focus_object = dof_target
        cam_data.dof.aperture_fstop = fstop
    return cam


def key_light(cam, target, power=60, color=(1.0, 0.85, 0.7), side=0.6):
    """Мягкий рисующий свет со стороны камеры: лица читаются в любой сцене, тени от основного света остаются."""
    d = (Vector(target) - cam.location)
    right = d.cross(Vector((0, 0, 1))).normalized()
    loc = cam.location + right * side * d.length * 0.5 + Vector((0, 0, 0.6))
    bpy.ops.object.light_add(type="AREA", location=loc)
    l = bpy.context.active_object
    l.data.energy = power
    l.data.color = color
    l.data.size = 1.0
    look_at(l, target)
    return l


def look_at(obj, target):
    d = Vector(target) - obj.location
    obj.rotation_euler = d.to_track_quat("-Z", "Y").to_euler()


def camera_move(cam, frames, p0, p1, t0, t1, lens0=None, lens1=None, shake_at=(), shake_amp=0.04, fps=30):
    """Камера летит из p0 в p1, смотрит с t0 на t1 (плавно, ease in/out), опционально меняет фокусное
    (наезд объективом). На кадрах shake_at — удар-тряска 5 кадров."""
    import random
    rnd = random.Random(11)
    for f in range(frames + 1):
        u = f / max(1, frames)
        e = u * u * (3 - 2 * u)
        cam.location = Vector(p0).lerp(Vector(p1), e)
        tgt = Vector(t0).lerp(Vector(t1), e)
        for s in shake_at:
            k = f - s
            if 0 <= k < 6:
                a = shake_amp * (1 - k / 6)
                cam.location += Vector((rnd.uniform(-a, a), rnd.uniform(-a, a), rnd.uniform(-a, a)))
        look_at(cam, tgt)
        cam.keyframe_insert("location", frame=f + 1)
        cam.keyframe_insert("rotation_euler", frame=f + 1)
        if lens0 and lens1:
            cam.data.lens = lens0 + (lens1 - lens0) * e
            cam.data.keyframe_insert("lens", frame=f + 1)


def idle_motion(root, frames, sway=0.03, seed=0):
    """Персонаж никогда не стоит мёртво: дыхание, лёгкое покачивание, поворот головы."""
    import random
    rnd = random.Random(seed)
    neck = bpy.data.objects.get(root.name + "_neck")
    hip = bpy.data.objects.get(root.name + "_hip")
    ph = rnd.random() * 6
    for f in range(0, frames + 1, 3):
        t = f / 30.0
        if hip:
            hip.location[2] = 0.012 * math.sin(t * 2.4 + ph)
            hip.rotation_euler[1] = sway * 0.3 * math.sin(t * 1.1 + ph)
            hip.keyframe_insert("location", frame=f + 1)
            hip.keyframe_insert("rotation_euler", frame=f + 1)
        if neck:
            neck.rotation_euler[2] = sway * 2.5 * math.sin(t * 0.7 + ph)
            neck.rotation_euler[0] = sway * math.sin(t * 1.3 + ph)
            neck.keyframe_insert("rotation_euler", frame=f + 1)


def walk(root, frames, p0, p1, seed=0):
    """Простая ходьба: перемещение + шаг (подпрыгивание и мах руками)."""
    hip = bpy.data.objects.get(root.name + "_hip")
    shs = [bpy.data.objects.get(f"{root.name}_shoulder{s}") for s in (-1, 1)]
    d = Vector(p1) - Vector(p0)
    root.rotation_euler[2] = math.atan2(d.x, -d.y) if d.length > 0 else root.rotation_euler[2]
    for f in range(frames + 1):
        u = f / max(1, frames)
        root.location = Vector(p0).lerp(Vector(p1), u)
        root.keyframe_insert("location", frame=f + 1)
        if f % 2 == 0:
            ph = f / 30.0 * 2 * math.pi * 1.8
            if hip:
                hip.location[2] = 0.03 * abs(math.sin(ph))
                hip.keyframe_insert("location", frame=f + 1)
            for i, sh in enumerate(shs):
                if sh:
                    sh.rotation_euler[0] = 0.5 * math.sin(ph + i * math.pi)
                    sh.keyframe_insert("rotation_euler", frame=f + 1)


def raise_arm(root, side=1, frame_from=1, frame_to=10, angle=-1.3):
    sh = bpy.data.objects.get(f"{root.name}_shoulder{side}")
    if not sh:
        return None
    sh.keyframe_insert("rotation_euler", frame=frame_from)
    sh.rotation_euler[0] = angle
    sh.keyframe_insert("rotation_euler", frame=frame_to)
    return sh


def render_settings(sc, w, h, samples=24, out=None, frames=None):
    sc.render.resolution_x = w
    sc.render.resolution_y = h
    sc.render.resolution_percentage = 100
    sc.cycles.samples = samples
    sc.cycles.use_adaptive_sampling = True
    sc.cycles.max_bounces = 4
    sc.cycles.volume_bounces = 0
    sc.cycles.volume_step_rate = 4.0
    sc.render.image_settings.file_format = "PNG"
    if frames:
        sc.frame_start, sc.frame_end = 1, frames
    if out:
        sc.render.filepath = out
