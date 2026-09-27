"""**Palettes compared on buildings, in context, before one is adopted.**

    compare(voices, scene, out_dir, *, seed=1, size=(640, 400), mix=None) -> dict

The city attempt adopted its upper ring's copper-green palette without ever putting it
next to an alternative; the user's review named dark prismarine as one it would have
preferred -- an example, not a mandate. Availability was not the bottleneck: nothing
*compared* plausible materials on actual roofs and walls before a palette was replicated
across a ring. This builds the **same small scene** under each candidate voice through
the production builders (`Builder.site` -> `type_builder` -> the type's own `build`, as
`scripts/test_attached_forms.py` stands a row), renders each from the **same cameras**
(`scripts/ca_eye.py`'s ray-marched textured view: one on foot, one raised oblique over
the roofs), and composes a labelled sheet with a JSON beside it: per voice, the resolved
role blocks, the mean rendered colour of the roof and wall pixels, and the image paths.

Scenes (`SCENES`):

  * `street`  -- a terrace of `shop_house` facing a row of `courtyard_house` across a
    five-wide paved street, closed at its far end by a short run of `wall`;
  * `palace`  -- one large hall (`types/ceremonial_hall.py` where it exists and loads,
    else `temple` at two storeys) behind a paved court (`plaza`);
  * `mansion` -- `court_large` and `courtyard_house` on a lane, a compound `wall`
    across the lane and behind them.

`scene="mix"` with `mix={ring_role: voice, ...}` (or a list of such dicts, one sheet
row each) builds each ring's own scene in its own voice -- `lower`/`middle`/`outer`
rings the street, `upper` the mansion, `palace`/`inner`/`court` the palace -- and sets
them side by side, so a proposed per-ring combination is judged as a combination.

A voice with a `variants` recipe is shown **finished** (`material.apply`, contextual,
`maintained`) because that is what would be delivered; the record says so.

Importing this module has no side effects; everything heavy is imported when called.
What reads best is an aesthetic judgement for whoever looks at the sheet -- this
measures colour and draws pictures, it does not rank palettes.
"""
from __future__ import annotations

import importlib.util
import json
import os
import time

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
SCENES = ("street", "palace", "mansion", "wall")
VIEWS = ("eye", "oblique")
GROUND = 64
#: volume extent: x, height, z. The ground is at GROUND; the volume starts 14 below it.
SX, SY, SZ = 104, 64, 84
Y0 = GROUND - 14

#: which scene a ring role is shown in, for `mix`. Matched on the role's words, so
#: `lower_ring` and `upper` both resolve.
ROLE_SCENE = (("palace", "palace"), ("inner", "palace"), ("court", "palace"),
              ("upper", "mansion"), ("mansion", "mansion"), ("noble", "mansion"),
              ("lower", "street"), ("middle", "street"), ("outer", "street"),
              ("street", "street"))

#: For each street side: the flank on the low and high side of the run and the walk-in
#: facing (from the lane into the lot). Runs go along x only here.
_FRONTS = {"north": ("west", "east", "south"), "south": ("west", "east", "north")}


class SceneError(ValueError):
    """A scene or a mix this module cannot build, named."""


# ------------------------------------------------------------------ the scenes

def _run(type_name: str, widths, depth: int, front: str, x_start: int, street_z: int,
         params) -> list:
    """Leaves of one terraced run along x fronting a street on `front`. `street_z` is
    the street row the lots touch (the lots lie north of it for front 'south', south of
    it for front 'north'). Each leaf carries the compiled site `district_compile` would
    give it -- the inset pad, the door mid-front, the landing on the street."""
    from .buildlib import site_pad_rect
    lo_side, hi_side, _walk = _FRONTS[front]
    out, a, n = [], x_start, len(widths)
    for i, w in enumerate(widths):
        att = ([] if i == 0 else [lo_side]) + ([] if i == n - 1 else [hi_side])
        x0, x1 = a, a + w - 1
        if front == "south":
            z0, z1 = street_z - depth, street_z - 1
        else:
            z0, z1 = street_z + 1, street_z + depth
        pad = list(site_pad_rect(x0, z0, x1, z1, att))
        da = (pad[0] + pad[2]) // 2
        edge = pad[3] if front == "south" else pad[1]
        land = (da, street_z)
        lf = {"label": f"{type_name[:5]}_{front[0]}{i}", "kind": "plot", "x0": x0,
              "z0": z0, "x1": x1, "z1": z1, "front": front, "attached": list(att),
              "type": type_name,
              "site": {"pad": pad, "floor": GROUND, "facing": front,
                       "door": [da, edge], "landing": [land[0], land[1]],
                       "attached": sorted(att), "why": "palettesheet"},
              "params": dict(params[i % len(params)])}
        out.append(lf)
        a += w
    return out


def _edge(label: str, pts, width: int, params: dict) -> dict:
    return {"label": label, "kind": "edge", "type": "wall", "width": width,
            "path": [list(p) for p in pts], "params": dict(params)}


def _hall_type() -> tuple:
    """(type name, params) of the palace hall: `ceremonial_hall` where it loads."""
    p = os.path.join(ROOT, "types", "ceremonial_hall.py")
    if os.path.exists(p):
        try:
            ns = _decl("ceremonial_hall")
            if callable(ns.get("build")):
                decl = ns.get("PARAMS") or {}
                params = {}
                if "storeys" in decl:
                    lo, hi = decl["storeys"][1], decl["storeys"][2]
                    params["storeys"] = max(lo, min(hi, 2))
                return "ceremonial_hall", params
        except Exception:                                  # noqa: BLE001 -- fall back
            pass
    return "temple", {"storeys": 2, "plan": "hall"}


def scene_parts(scene: str, hall: tuple | None = None) -> dict:
    """{"leaves": [...], "street": [(x, z), ...], "cams": {view: (at, look, fov)}}.
    `hall` is the palace scene's (type, params); resolved by `_hall_type` if None."""
    g = GROUND
    if scene == "street":
        shops = _run("shop_house", [8, 7, 9, 8, 7, 8], 13, "south", 12, 30,
                     [{"storeys": 2, "trade": "tea"}, {"storeys": 2, "trade": "cloth"},
                      {"storeys": 1, "trade": "grain"}])
        # the courtyard row across the street: lots touch the street's far row (34)
        houses = _run("courtyard_house", [15, 17, 15], 17, "north", 12, 34,
                      [{"storeys": 1, "screen": "wall", "court": 0}])
        for lf in houses:
            lf["site"]["landing"][1] = 34
        leaves = shops + houses + [
            _edge("wall_end", [(64, 14), (64, 56)], 3,
                  {"height": 9, "width": 3, "crown": "crenellated"})]
        street = [(x, z) for x in range(8, 61) for z in range(30, 35)]
        cams = {"eye": ((8.5, g + 2.6, 32.5), (62.5, g + 5.0, 32.5), 70.0),
                "oblique": ((-4.0, g + 38.0, 76.0), (37.0, g + 2.0, 30.0), 60.0)}
    elif scene == "palace":
        name, params = hall or _hall_type()
        hall = {"label": "hall", "kind": "plot", "x0": 22, "z0": 12, "x1": 52, "z1": 38,
                "front": "south", "attached": [], "type": name, "params": params}
        from .buildlib import site_pad_rect
        pad = list(site_pad_rect(22, 12, 52, 38, []))
        mid = (pad[0] + pad[2]) // 2
        hall["site"] = {"pad": pad, "floor": g, "facing": "south", "door": [mid, pad[3]],
                        "landing": [mid, 40], "attached": [], "why": "palettesheet"}
        court = {"label": "court", "kind": "area", "x0": 18, "z0": 41, "x1": 56, "z1": 66,
                 "type": "plaza", "params": {"paving": "framed", "edge": "kerb"}}
        leaves = [hall, court]
        street = [(x, z) for x in range(22, 53) for z in range(39, 41)]
        cams = {"eye": ((37.5, g + 2.6, 64.5), (37.5, g + 10.0, 25.0), 70.0),
                "oblique": ((78.0, g + 36.0, 86.0), (37.0, g + 6.0, 34.0), 58.0)}
    elif scene == "mansion":
        big = _run("court_large", [25], 25, "north", 14, 19,
                   [{"storeys": 1, "yard": "garden"}])
        house = _run("courtyard_house", [15], 17, "north", 42, 19,
                     [{"storeys": 1, "screen": "planted", "court": 0}])
        leaves = big + house + [
            _edge("compound_front", [(8, 12), (62, 12)], 1,
                  {"height": 5, "width": 1, "crown": "solid"}),
            _edge("compound_back", [(8, 48), (62, 48)], 1,
                  {"height": 5, "width": 1, "crown": "solid"})]
        street = [(x, z) for x in range(9, 62) for z in range(14, 20)]
        cams = {"eye": ((9.5, g + 2.6, 15.5), (44.5, g + 5.0, 26.0), 70.0),
                "oblique": ((0.0, g + 34.0, -2.0), (36.0, g + 3.0, 32.0), 58.0)}
    elif scene == "wall":
        # a city wall seen from the street at its foot, with houses for scale: what a
        # wall's palette is judged on, so a wall is never adopted in a palette nobody
        # has seen on a wall
        houses = _run("row_house", [6, 6, 6, 6, 6, 6], 10, "south", 20, 58,
                      [{"storeys": 1, "front": "lattice"}, {"storeys": 2, "front": "open"}])
        wall = {"label": "city_wall", "kind": "edge", "type": "great_wall", "width": 5,
                "path": [[2, 30], [101, 30]],
                "params": {"height": 24, "width": 3, "parapet": "crenellated",
                           "face": "masonry"}}
        leaves = [wall] + houses
        street = [(x, z) for x in range(2, 102) for z in range(33, 58)]
        cams = {"eye": ((52.5, g + 2.6, 72.5), (52.5, g + 16.0, 30.0), 70.0),
                "oblique": ((104.0, g + 30.0, 80.0), (45.0, g + 10.0, 30.0), 58.0)}
    else:
        raise SceneError(f"a scene is one of {', '.join(SCENES)} or 'mix', not {scene!r}")
    return {"scene": scene, "leaves": leaves, "street": street, "cams": cams}


# ------------------------------------------------------------------ building

_DECLS: dict = {}


def _decl(type_name: str) -> dict:
    """The type file's namespace, cached by its modification time (a type another
    worker is editing is re-read, not served stale)."""
    path = os.path.join(ROOT, "types", f"{type_name}.py")
    key = (type_name, os.stat(path).st_mtime_ns)
    if key not in _DECLS:
        ns: dict = {"__name__": "__ethoslm_type__", "__file__": path}
        exec(compile(open(path).read(), path, "exec"), ns)        # noqa: S102
        _DECLS[key] = ns
    return _DECLS[key]


def _flat():
    import numpy as np
    from .observe import Volume
    codes = np.zeros((SX, SY, SZ), np.uint16)
    g = GROUND - Y0
    codes[:, :g, :] = 3
    codes[:, g, :] = 1
    return Volume(0, Y0, 0, codes, ["air", "grass_block", "dirt", "stone"])


class _Registry:
    def __init__(self, leaves):
        self.plots = []
        for lf in leaves:
            if lf.get("kind") == "edge":
                continue
            self.plots.append({"label": lf["label"], "x0": lf["x0"], "z0": lf["z0"],
                               "x1": lf["x1"], "z1": lf["z1"]})
        self.claimed_this_pass = [dict(p) for p in self.plots]

    def plots_list(self):
        return [dict(p) for p in self.plots]

    def reserve(self, *_a, **_k):
        return True


def _street_mat(pal: dict) -> str:
    from .prims import family
    for role in ("ground", "footing"):
        if pal.get(role) and family(pal[role]):
            return family(pal[role])
    return "stone_brick"


def build(scene: str, voice: str, *, seed: int = 1, finish: bool = True,
          hall: tuple | None = None) -> dict:
    """The scene built in one voice. Returns the volume shown (finished where the voice
    carries a recipe), the structural one, and what refused."""
    from . import material as material_mod, pipeline, stages, surfaces as surfaces_mod
    from . import offline, voices as voices_mod
    from .buildlib import Builder, SiteRefused
    from .circulate import Network, Threshold, emit
    from .frontage import Frontage
    t0 = time.perf_counter()
    sc = scene_parts(scene, hall)
    pal = pipeline.voice_palette(voice)
    if not pal:
        raise SceneError(f"there is no voice called {voice!r}")
    roof = pipeline.voice_roof(voice) or {}
    if roof.get("chimney") is None:
        roof = dict(roof, chimney=voices_mod.chimney_default("east_asian"))
    vol = _flat()
    cells = {c: {"y": GROUND, "rank": 1, "face": None} for c in sc["street"]}
    ths = []
    for lf in sc["leaves"]:
        st = lf.get("site")
        if not st:
            continue
        lx, lz = st["landing"]
        ths.append(Threshold(id=lf["label"], x=lx, z=lz, y=GROUND,
                             facing=_FRONTS[lf["front"]][2],
                             door=(st["door"][0], GROUND + 1, st["door"][1])))
    net = Network(cells=cells, thresholds=ths)
    # the street first, as production lays its lanes before the parts
    b0 = Builder(offline.OfflineSite(vol))
    b0._vol = vol
    emit(b0, net, _street_mat(pal), lantern_every=0, post_every=0)
    b0.resolve_steps()
    vol = stages.apply_pending(vol, b0._pending)
    b = Builder(offline.OfflineSite(vol))
    b._vol = vol
    b.frontage = Frontage(vol, net)
    b.registry = _Registry(sc["leaves"])
    refused, built_parts, records = [], [], []
    for lf in sc["leaves"]:
        geo = {k: lf[k] for k in pipeline.PART_GEOMETRY if k in lf}
        try:
            ns = _decl(lf["type"])
            p = b.site(dict(geo), mat=pal, roof=roof)
            if p.get("ground") == "unsited":
                refused.append({"part": lf["label"], "why": "unsited"})
                continue
            res = ns["build"](b.type_builder(p, role=ns.get("ROLE")), p,
                              int(seed) * 31 + len(built_parts), **dict(lf.get("params") or {}))
            if isinstance(res, dict) and res.get("ok") is False:
                refused.append({"part": lf["label"], "why": res.get("reason")})
            built_parts.append(lf["label"])
            try:
                records.append(surfaces_mod.record(
                    b, {**p, "name": lf["label"], "type": lf["type"]},
                    part_index=len(b.parts) - 1, voice_name=voice))
            except Exception:                              # noqa: BLE001 -- no finish
                pass
        except SiteRefused as e:
            refused.append({"part": lf["label"], "why": f"SiteRefused: {e}"})
        except Exception as e:                             # noqa: BLE001 -- reported
            refused.append({"part": lf["label"], "why": f"{type(e).__name__}: {e}"})
    b.resolve_steps()
    built = stages.apply_pending(vol, b._pending)
    shown, finished = built, None
    recipe = {}
    try:
        recipe = material_mod.recipe_for(voice)
    except Exception:                                      # noqa: BLE001
        recipe = {}
    if finish and recipe and records:
        doc = {"record": "surfaces", "version": 1, "parts": records,
               "flags": surfaces_mod.FLAGS, "roles": list(surfaces_mod.SURFACE_ROLES),
               "editable": list(surfaces_mod.EDITABLE_ROLES)}
        try:
            shown, rec = material_mod.apply(
                built, doc, {"by_voice": {voice: recipe}},
                {"condition": "maintained", "weather_from": material_mod.WEATHER_FROM},
                seed=int(seed), mode="contextual")
            finished = {"substituted": rec.get("substituted"),
                        "figure_refused": rec.get("figure_refused")}
        except Exception as e:                             # noqa: BLE001 -- reported
            finished = {"error": f"{type(e).__name__}: {e}"}
            shown = built
    return {"scene": scene, "voice": voice, "volume": shown, "built": built,
            "cams": sc["cams"], "palette": pal, "recipe": recipe, "finished": finished,
            "parts": built_parts, "refused": refused,
            "seconds": round(time.perf_counter() - t0, 1)}


# ------------------------------------------------------------------ rendering

_EYE = None


def _eye():
    global _EYE
    if _EYE is None:
        p = os.path.join(ROOT, "scripts", "ca_eye.py")
        spec = importlib.util.spec_from_file_location("ethoslm_ca_eye", p)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        _EYE = mod
    return _EYE


def role_blocks(pal: dict, recipe: dict | None = None) -> dict:
    """{role: {"exact": [...], "shapes": [...]}}: the blocks a role may appear as.

    `exact` is the role's own cube, stairs and slab (and, for a role that is a bare
    block, that block); `shapes` is every other shape of its family the game has and
    the families its variants may lay. `wall_alt` counts as wall. Pixels are assigned
    to `exact` blocks first, so a floor of `smooth_quartz` is the floor's and not the
    `fine` face of a quartz wall."""
    from .prims import SHAPES, family, shape, solid
    out: dict = {}
    for role, mat in pal.items():
        key = "wall" if role == "wall_alt" else role
        rec = out.setdefault(key, {"exact": set(), "shapes": set()})
        fam = family(str(mat))
        try:
            rec["exact"].add(str(solid(str(mat))).split("[")[0])
        except Exception:                                  # noqa: BLE001
            rec["exact"].add(str(mat))
        if fam:
            for k in SHAPES:
                try:
                    b = str(shape(fam, k)).split("[")[0]
                except Exception:                          # noqa: BLE001
                    continue
                rec["exact" if k in ("full", "stairs", "slab") else "shapes"].add(b)
        for r in (recipe or {}).get(role) or []:
            vf = family(r["family"])
            for k in ("full", "stairs", "slab", "wall"):
                try:
                    rec["shapes"].add(str(shape(vf or r["family"], k)).split("[")[0])
                except Exception:                          # noqa: BLE001
                    pass
    return {k: {t: sorted(v[t]) for t in v} for k, v in out.items()}


#: which role a block counts for where two roles share one: the roof is what the oblique
#: is for, and the wall is what the street view is for
_ROLE_ORDER = ("roof", "wall", "frame", "trim", "footing", "floor", "ground")


def _colours(img, codes, palette, blocks: dict) -> dict:
    import numpy as np
    by_name: dict = {}
    for i, s in enumerate(palette):
        by_name.setdefault(str(s).split("[")[0].split(":")[-1], []).append(i)
    owner: dict = {}
    for tier in ("exact", "shapes"):
        for role in _ROLE_ORDER:
            for n in (blocks.get(role) or {}).get(tier, ()):
                for i in by_name.get(n, ()):
                    owner.setdefault(i, role)
    out = {}
    total = int((codes >= 0).sum()) or 1
    for role in _ROLE_ORDER:
        idx = sorted(i for i, r in owner.items() if r == role)
        m = np.isin(codes, idx) if idx else np.zeros(codes.shape, bool)
        n = int(m.sum())
        rgb = [int(round(v)) for v in img[m].mean(axis=0)] if n else None
        out[role] = {"rgb": rgb, "pixels": n, "share": round(n / total, 3)}
    return out


def render(res: dict, out_dir: str, size=(640, 400), tag: str = "") -> dict:
    """Both views of one built scene to PNGs, with per-role mean colours."""
    import numpy as np
    from PIL import Image
    eye = _eye()
    vol = res["volume"]
    blocks = role_blocks(res["palette"], res.get("recipe"))
    got = {"images": {}, "colour": {}}
    stem = f"{res['scene']}__{res['voice']}{tag}"
    for view in VIEWS:
        at, look, fov = res["cams"][view]
        hits: dict = {}
        img = eye.render(vol, at, look, size=tuple(size), fov=fov, far=260.0, hits=hits)
        p = os.path.join(out_dir, f"{stem}__{view}.png")
        Image.fromarray(img).save(p)
        got["images"][view] = p
        got["colour"][view] = _colours(img.astype(np.float64), hits["code"],
                                       [str(s) for s in vol.palette], blocks)
    # one number per role over both views, pixel-weighted
    got["mean_rgb"] = {}
    for role in ("roof", "wall"):
        tot = sum(got["colour"][v][role]["pixels"] for v in VIEWS)
        if tot:
            got["mean_rgb"][role] = [int(round(sum(
                (got["colour"][v][role]["rgb"] or [0, 0, 0])[c]
                * got["colour"][v][role]["pixels"] for v in VIEWS) / tot)) for c in range(3)]
        else:
            got["mean_rgb"][role] = None
    got["role_blocks"] = blocks
    return got


# ------------------------------------------------------------------ the sheet

def _font(px: int):
    from PIL import ImageFont
    for f in ("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
              "/usr/share/fonts/TTF/DejaVuSans-Bold.ttf"):
        if os.path.exists(f):
            return ImageFont.truetype(f, px)
    try:
        return ImageFont.load_default(size=px)
    except TypeError:
        return ImageFont.load_default()


def _label(img, text: str, sub: str = ""):
    from PIL import ImageDraw
    d = ImageDraw.Draw(img, "RGBA")
    f, fs = _font(18), _font(13)
    w = max(d.textlength(text, font=f), d.textlength(sub, font=fs) if sub else 0) + 14
    d.rectangle([0, 0, w, 46 if sub else 28], fill=(0, 0, 0, 170))
    d.text((7, 4), text, font=f, fill=(255, 255, 255, 255))
    if sub:
        d.text((7, 27), sub, font=fs, fill=(230, 230, 230, 255))
    return img


def _swatch(img, rgb_by_role: dict):
    from PIL import ImageDraw
    d = ImageDraw.Draw(img, "RGBA")
    f = _font(12)
    W, H = img.size
    x = W - 8
    for role in ("wall", "roof"):
        rgb = rgb_by_role.get(role)
        if not rgb:
            continue
        x -= 70
        d.rectangle([x, H - 30, x + 62, H - 6], fill=tuple(rgb) + (255,),
                    outline=(255, 255, 255, 255))
        d.text((x + 3, H - 45), role, font=f, fill=(255, 255, 255, 255),
               stroke_width=2, stroke_fill=(0, 0, 0, 255))
    return img


def _compose(rows: list, path: str, title: str) -> str:
    """rows: [[(png, label, sub, swatch), ...], ...] -> one sheet PNG."""
    from PIL import Image, ImageDraw
    tiles = [[Image.open(t[0]).convert("RGB") for t in r] for r in rows]
    tw = max(t.size[0] for r in tiles for t in r)
    th = max(t.size[1] for r in tiles for t in r)
    ncol = max(len(r) for r in tiles)
    head = 34
    sheet = Image.new("RGB", (ncol * tw + (ncol - 1) * 4, head + len(rows) * (th + 4)),
                      (24, 24, 24))
    d = ImageDraw.Draw(sheet)
    d.text((8, 7), title, font=_font(18), fill=(255, 255, 255))
    for ri, (r, spec) in enumerate(zip(tiles, rows)):
        for ci, (im, (_p, lab, sub, sw)) in enumerate(zip(r, spec)):
            im = _label(im.copy(), lab, sub)
            if sw:
                im = _swatch(im, sw)
            sheet.paste(im, (ci * (tw + 4), head + ri * (th + 4)))
    sheet.save(path)
    return path


def _resolved(mat) -> str:
    from .prims import solid
    try:
        return str(solid(str(mat))).split(":")[-1]
    except Exception:                                      # noqa: BLE001
        return str(mat)


def _scene_of_role(role: str) -> str:
    r = role.lower()
    for key, scene in ROLE_SCENE:
        if key in r:
            return scene
    raise SceneError(f"no scene for the ring role {role!r}; name it with one of "
                     f"{', '.join(k for k, _ in ROLE_SCENE)}")


def compare(voices: list, scene: str, out_dir: str, *, seed: int = 1,
            size=(640, 400), mix=None, finish: bool = True, name: str | None = None,
            log=print) -> dict:
    """Build `scene` under every voice in `voices`, render the same two views of each,
        and compose one labelled sheet. Returns (and writes beside the sheet as JSON):

            {"scene", "sheet": <png>, "json": <path>, "seconds",
             "voices": {voice: {"roles": {role: block}, "role_blocks", "images":
                        {"eye", "oblique"}, "colour": {view: {role: {rgb, pixels, share}}},
                        "mean_rgb": {"roof", "wall"}, "finished", "refused", "seconds"}}}

        `scene="mix"`: `mix` is `{ring_role: voice}` or a list of them, each shown as one
        row of the sheet with each role's own scene in its own voice (`ROLE_SCENE`); the
        record is then under "mixes". `voices` may be empty for a mix.

    """
    from . import pipeline
    os.makedirs(out_dir, exist_ok=True)
    t0 = time.perf_counter()
    stem = name or (f"palette_{scene}" if scene != "mix" else "palette_mix")
    out = {"record": "palette_comparison", "scene": scene, "seed": int(seed),
           "size": list(size), "voices": {},
           "note": "the same scene, cameras and lighting under each voice; the mean "
                   "colours are of the rendered roof and wall pixels (ca_eye's "
                   "textured diagnostic view), approximate. Which reads best is an "
                   "aesthetic judgement for the reader of the sheet."}

    # **One hall for the whole sheet**, resolved once: a type file that appears or
    # changes mid-comparison must not give one voice a different building.
    hall_pin = _hall_type()
    out["hall"] = {"type": hall_pin[0], "params": hall_pin[1]}

    def one(sc: str, v: str) -> dict:
        res = build(sc, v, seed=seed, finish=finish, hall=hall_pin)
        got = render(res, out_dir, size=size)
        rec = {"roles": {r: _resolved(m) for r, m in res["palette"].items()},
               "role_blocks": got["role_blocks"], "images": got["images"],
               "colour": got["colour"], "mean_rgb": got["mean_rgb"],
               "finished": res["finished"], "parts": res["parts"],
               "refused": res["refused"], "seconds": res["seconds"]}
        if log:
            log(f"   palette {sc}/{v}: {len(res['parts'])} part(s) in "
                f"{res['seconds']}s, roof {got['mean_rgb']['roof']}, wall "
                f"{got['mean_rgb']['wall']}"
                + (f", refused {[r['part'] for r in res['refused']]}"
                   if res["refused"] else ""))
        return rec

    rows = []
    if scene == "mix":
        mixes = mix if isinstance(mix, list) else [mix] if mix else []
        if not mixes:
            raise SceneError("scene 'mix' needs mix={ring_role: voice, ...}")
        out["mixes"] = []
        for m in mixes:
            row_top, row_bot, rec = [], [], {}
            for role, v in m.items():
                if not pipeline.voice_palette(v):
                    raise SceneError(f"there is no voice called {v!r}")
                sc = _scene_of_role(role)
                r = one(sc, v)
                rec[role] = {"voice": v, "scene": sc, **r}
                row_top.append((r["images"]["oblique"], f"{role}: {v}", f"{sc}, raised",
                                r["mean_rgb"]))
                row_bot.append((r["images"]["eye"], f"{role}: {v}", f"{sc}, on foot",
                                None))
            out["mixes"].append(rec)
            rows += [row_top, row_bot]
        title = (f"palette mix -- {len(mixes)} combination(s), two rows each (raised, "
                 f"on foot); each tile is one ring role's scene in its own voice")
    else:
        if scene not in SCENES:
            raise SceneError(f"a scene is one of {', '.join(SCENES)} or 'mix', not "
                             f"{scene!r}")
        for v in voices:
            if not pipeline.voice_palette(v):
                raise SceneError(f"there is no voice called {v!r}")
        for v in voices:
            r = one(scene, v)
            out["voices"][v] = r
            fin = r.get("finished") or {}
            sub = (f"finished: {fin.get('substituted')} variant cell(s)"
                   if fin.get("substituted") is not None else "structural")
            rows.append([(r["images"]["eye"], v, f"{scene}, on foot -- {sub}", None),
                         (r["images"]["oblique"], v,
                          f"roof {r['roles']['roof']}, wall {r['roles']['wall']}",
                          r["mean_rgb"])])
        title = f"palette comparison -- {scene} -- same scene and cameras per voice"
    out["sheet"] = _compose(rows, os.path.join(out_dir, f"{stem}.png"), title)
    out["json"] = os.path.join(out_dir, f"{stem}.json")
    out["seconds"] = round(time.perf_counter() - t0, 1)
    with open(out["json"], "w") as fh:
        json.dump(out, fh, indent=1)
    return out
