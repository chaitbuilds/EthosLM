"""**The designed-place stages**: the model designs, the compiler resolves, regions build.

The design synthesis round. A round with `flags.design` runs, after the reading, the
interpretation and the spec:

  - `design`         -- the staged design job: request, claims, references, a review of
                        the last built candidate, candidate sites off the terrain atlas
                        with their maps, the forms' lot sizes, the grain vocabulary and
                        palette comparison sheets built in context. The agent writes 2-3
                        materially different whole-place proposals (`citydesign`).
  - `design_compare` -- every proposal compiled on its site's ground (`cityresolve`):
                        plan previews, capacity, costs against the current city. The
                        agent adopts one (with named revisions) in a second job.
  - `design_resolve` -- the adopted design compiled for construction: plan tree, network,
                        rasters, regions. A blocking finding goes back to the design as
                        a revision job, not to a fallback.
  - `regions`        -- bounded construction of the regions `flags.design.build` names,
                        each alone with its neighbours as context, resumable: a region
                        whose inputs have not changed is not rebuilt, a changed design
                        rebuilds only the regions whose own inputs moved.
  - `region_views`   -- textured frames of what was built, off the adopted artifact.

Coarse planning is cheap (seconds); construction is bounded per region. The design's
extent is the design's; resource limits govern execution batches, never the ambition.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import shutil
import time

import numpy as np

from .round import ROOT


# ---------------------------------------------------------------- config

def _cfg(rnd) -> dict:
    return dict(rnd.flags.get("design") or {})


def _p(rnd, *parts) -> str:
    return rnd.rel("design", *parts)


def _world(rnd) -> str:
    """The delivery save the round names. There is no default: a fresh round never
    writes into, or reads the ground of, another place's save."""
    w = _cfg(rnd).get("world")
    if not w:
        raise ValueError("flags.design.world names no delivery save")
    return os.path.join(ROOT, w)


def _terrain(rnd) -> str:
    """The ground as found: immutable. The delivery save is written into, so once a
    region has been delivered it is no longer the ground as found; construction reads
    `flags.design.terrain` (the save as it stood before the first write), and the save
    itself only while nothing has been written into it."""
    t = _cfg(rnd).get("terrain")
    if t:
        return os.path.join(ROOT, t)
    snap = os.path.join(ROOT, "run", "snapshots", f"{rnd.name}.before-write")
    return snap if os.path.isdir(snap) else _world(rnd)


def _atlas_dir(rnd) -> str:
    """The terrain atlas of the round's save (`flags.design.atlas`), else the round's
    own (`out/<round>-atlas`): never another place's."""
    return os.path.join(ROOT, _cfg(rnd).get("atlas") or f"out/{rnd.name}-atlas")


def _jload(p, default=None):
    return json.load(open(p)) if os.path.exists(p) else default


def _needs(role: str, request: str, write: str, note: str = "", **kw) -> dict:
    return {"status": "needs_model", "role": role, "request": request, "write": write,
            "note": note, **kw}


# ---------------------------------------------------------------- sites

def _setting(rnd) -> dict:
    """The place spec's `setting` (surface, water, relief, biome), or {}."""
    return dict((rnd.place_spec() or {}).get("setting") or {})


def ensure_biomes(rnd) -> bool:
    """The biome layer of the round's atlas, read from the save the atlas was read from
    where it is missing. False where it cannot be (no save on record): the candidates
    are then offered without the setting's biome, and the design job is told so."""
    from .. import atlas
    d = _atlas_dir(rnd)
    if atlas.has_biomes(d):
        return True
    rec = atlas.coverage(d)
    world = rec.get("world")
    src = os.path.join(ROOT, world) if world else _terrain(rnd)
    if not os.path.isdir(os.path.join(src, "region")):
        return False
    atlas.build(src, d, workers=int(_cfg(rnd).get("workers") or 4))
    return atlas.has_biomes(d)


def site_candidates(rnd, radii=None) -> list:
    """Candidate centres off the terrain atlas of the design's save, per radius, with a
    map of each; written under `design/sites/`. Held to the place spec's setting where
    it names biomes (`atlas.scan_sites`): an oasis is not offered a snowfield."""
    from .. import atlas, groundread
    cfg = _cfg(rnd)
    radii = radii or cfg.get("radii") or default_radii(rnd)
    out = _p(rnd, "sites")
    os.makedirs(out, exist_ok=True)
    want = [b for b in (_setting(rnd).get("biome") or []) if b != "any"]
    have_biomes = ensure_biomes(rnd) if want else atlas.has_biomes(_atlas_dir(rnd))
    key = json.dumps({"radii": radii, "atlas": atlas.coverage(_atlas_dir(rnd)),
                      "biomes": want if have_biomes else None}, sort_keys=True)
    have = _jload(os.path.join(out, "sites.json"))
    if have and have.get("key") == key:
        return have["sites"]
    co = atlas.coarse(8, _atlas_dir(rnd))
    excl = (groundread.excluded if cfg.get("exclusions", False) else None)
    sites = []
    for R in radii:
        for n, c in enumerate(atlas.scan_sites(R, co=co, stride=int(cfg.get("stride", 64)),
                                               water=tuple(cfg.get("water", (0.0, 0.25))),
                                               excluded=excl, top=3,
                                               biomes=want if have_biomes else None)):
            sid = f"r{R}_{n}"
            c["id"] = sid
            c["map"] = site_map(rnd, c, os.path.join(out, f"{sid}.png"))
            sites.append(c)
    json.dump({"key": key, "sites": sites,
               "setting": {"biome": want, "held": bool(want and have_biomes)}},
              open(os.path.join(out, "sites.json"), "w"), indent=1)
    return sites


def default_radii(rnd) -> list:
    """Candidate site radii off the request's own kind of place: the ground its
    structure band stands on at the registered fabric (`spec.SIZE_BANDS`,
    `spec.CEILING`), at 0.6, 1 and 1.4 of that. A village searches for a village's
    ground; nothing here is a city's number unless the request is a city."""
    from .. import spec as spec_mod
    s = rnd.place_spec() or {}
    kind = s.get("kind") or "village"
    lo, hi = spec_mod.SIZE_BANDS.get(kind, spec_mod.SIZE_BANDS["village"])
    per = spec_mod.CEILING["footprint"] ** 2 / float(spec_mod.CEILING["structures"])
    r = math.sqrt(hi * per / math.pi)
    return sorted({max(48, int(8 * round(r * f / 8))) for f in (0.6, 1.0, 1.4)})


def site_map(rnd, site: dict, path: str, pad: int = 64) -> str:
    """A height-shaded map of a candidate with water and the circle drawn."""
    from PIL import Image, ImageDraw
    from .. import atlas
    R = int(site["radius"])
    cx, cz = site["centre"]
    x0, z0 = cx - R - pad, cz - R - pad
    n = 2 * (R + pad)
    g, ws, wet = atlas.ground(x0, z0, n, n, _atlas_dir(rnd))
    g = g.astype(np.float32)
    ok = g > -30000
    lo, hi = (np.percentile(g[ok], 2), np.percentile(g[ok], 98)) if ok.any() else (60, 80)
    v = np.clip((g - lo) / max(1.0, hi - lo), 0, 1)
    gx = np.gradient(g, axis=0)
    gz = np.gradient(g, axis=1)
    shade = np.clip(1.0 - 0.12 * (gx + gz), 0.5, 1.4)
    rgb = np.stack([90 + 120 * v, 130 + 90 * v, 70 + 60 * v], -1) * shade[..., None]
    bio = atlas.biome_window(x0, z0, n, n, _atlas_dir(rnd))
    tint = {"desert": (225, 205, 140), "badlands": (200, 110, 60), "snowy": (235, 240, 245),
            "savanna": (170, 165, 90), "jungle": (40, 120, 40), "swamp": (70, 90, 60)}
    for name, col in tint.items():
        k = atlas.BIOME_CLASSES.index(name)
        sel = bio == k
        if sel.any():
            rgb[sel] = np.array(col, np.float32) * (0.75 + 0.25 * v[sel, None]) \
                * shade[sel, None]
    rgb[wet] = (60, 100, 200)
    rgb[~ok] = (60, 0, 0)
    img = Image.fromarray(np.clip(rgb, 0, 255).astype(np.uint8).transpose(1, 0, 2))
    step = max(1, n // 700)
    img = img.resize((n // step, n // step))
    dr = ImageDraw.Draw(img)
    c = (pad + R) / step
    dr.ellipse((c - R / step, c - R / step, c + R / step, c + R / step), outline=(255, 40, 40),
               width=2)
    dr.text((6, 6), f"{site.get('id', '')} centre {cx},{cz} r{R}: fall {site['fall']}, "
                    f"water {site['water']:.0%}, rough {site['rough']:.0%}, y~{site['median']}",
            fill=(255, 255, 255))
    if site.get("biomes"):
        dr.text((6, 20), "biomes " + ", ".join(
            f"{k} {v:.0%}" for k, v in sorted(site["biomes"].items(),
                                               key=lambda kv: -kv[1])[:4]),
            fill=(255, 255, 255))
    img.save(path)
    return os.path.relpath(path, ROOT)


def site_ground(rnd, city_frame: tuple) -> tuple:
    """`(found, water_surface, wet)` over a city's frame, off the atlas of its save."""
    from .. import atlas
    X0, Z0, W = city_frame
    g, ws, wet = atlas.ground(X0, Z0, W, W, _atlas_dir(rnd))
    miss = g <= -30000
    if miss.any():
        med = int(np.median(g[~miss])) if (~miss).any() else 64
        g[miss] = med
        ws[miss] = med
    return g, ws, wet


# ---------------------------------------------------------------- the design job

def _forms_card() -> str:
    """Every form in the library a design can name, off the type files themselves: what
    each is for, its tradition, the lot it stands on, what it can be asked for and its
    parameters. Discovered, not listed: a form added to `types/` is offered here."""
    from .. import pipeline, envelope
    tdir = os.path.join(ROOT, "types")
    groups = {"dwelling": [], "plot": [], "area": [], "edge": [], "point": []}
    for f in sorted(os.listdir(tdir)):
        if not f.endswith(".py") or f.startswith("_"):
            continue
        t = f[:-3]
        try:
            d = pipeline.load_type(os.path.join(tdir, f))
        except Exception as e:                    # noqa: BLE001 -- reported in the card
            groups["plot"].append(f"- `{t}`: does not load ({type(e).__name__})")
            continue
        fp = (d.get("needs") or {}).get("footprint") or (1, 1, 512, 512)
        prm = ", ".join(f"{k} {v[1] if v[0] == 'choice' else f'{v[1]}..{v[2]}'}"
                        for k, v in (d.get("params") or {}).items())
        feats = envelope.declared_features(t)
        row = (f"- `{t}` ({d['kind']}, {d.get('form') or 'any tradition'}, "
               f"{d.get('role')}{', ' + d['function'] if d.get('function') else ''}"
               f"{', attaches in runs' if d.get('attached') else ''}): pad "
               f"{fp[0]}x{fp[1]}..{fp[2]}x{fp[3]}"
               + (f"; can be asked for {', '.join(feats)}" if feats else "")
               + (f"; params {prm}" if prm else "")
               + (f". {_first_line(d.get('src'))}" if d.get("src") else ""))
        key = "dwelling" if d.get("function") == "dwelling" else d["kind"]
        groups.setdefault(key, []).append(row)
    heads = {"dwelling": "Dwellings (a grain's `dwelling`; a lot is the form's own pad "
                         "plus one column of inset each side)",
             "plot": "Other buildings", "area": "Open ground (a grain's leftovers, a "
                                               "landmark's `form`)",
             "edge": "Walls", "point": "Gates"}
    out = []
    for k in ("dwelling", "plot", "area", "edge", "point"):
        if groups.get(k):
            out.append(f"**{heads[k]}**\n" + "\n".join(groups[k]))
    return "\n\n".join(out)


def _first_line(src: str) -> str:
    """A type file's first docstring sentence: what the form is, in its author's words."""
    import ast
    try:
        doc = ast.get_docstring(ast.parse(src)) or ""
    except SyntaxError:
        return ""
    line = doc.strip().split("\n\n")[0].replace("\n", " ")
    return line[:200]


def _palette_sheets(rnd) -> list:
    """Comparison sheets of the candidate voices, built in context; cached per voice set."""
    cfg = _cfg(rnd)
    voices = cfg.get("palette_candidates") or []
    if not voices:
        return []
    out = _p(rnd, "palette")
    os.makedirs(out, exist_ok=True)
    got = []
    from .. import palettesheet
    for scene, vs in (("street", voices), ("mansion", voices),
                      ("palace", cfg.get("palace_candidates") or voices)):
        js = os.path.join(out, f"{scene}.json")
        rec = _jload(js)
        if not rec or rec.get("voices") != vs:
            try:
                rec = palettesheet.compare(vs, scene, out, name=scene)
                rec["voices"] = vs
                json.dump(rec, open(js, "w"), indent=1, default=str)
            except Exception as e:                # noqa: BLE001 -- reported
                got.append(f"(the {scene} sheet failed: {type(e).__name__}: {e})")
                continue
        got.append(os.path.relpath(rec.get("sheet") or rec.get("path") or js, ROOT))
    return got


def _manifests(rnd) -> list:
    """Those the round supplies (`flags.design.references`, a path or a list of paths) and
    the one its own `design_references` job wrote. Nothing is read by default: a fresh
    round inherits no other place's references.
    """
    got = []
    sup = _cfg(rnd).get("references")
    for q in ([sup] if isinstance(sup, str) else list(sup or [])):
        got.append(os.path.join(ROOT, q))
    own = _p(rnd, "references.json")
    if os.path.exists(own):
        got.append(own)
    return got


def _ref_rows(rnd) -> list:
    """Every reference the design reads: `(path, record)`, the file resolved beside
    the manifest that lists it."""
    rows = []
    seen = set()
    for p in _manifests(rnd):
        doc = _jload(p) or []
        refs = doc.get("views") if isinstance(doc, dict) else doc
        d = os.path.dirname(p)
        for r in refs or []:
            f = r.get("path") or os.path.join(d, r["file"])
            f = os.path.join(ROOT, f) if not os.path.isabs(f) else f
            if f in seen:
                continue
            seen.add(f)
            rows.append((f, r))
    return rows


def _refs(rnd) -> str:
    rows = _ref_rows(rnd)
    return "\n".join(f"- `{os.path.relpath(f, ROOT)}` -- {r.get('shows', '')} "
                     f"({r.get('kind', '')}; {r.get('url', '')})"
                     + (f" **Role:** {r['role']}." if r.get("role") else "")
                     for f, r in rows) or "(no visual references)"


def _ref_brief(rnd) -> str:
    """The reference-derived brief the `design_references` job wrote: what makes the
    place recognisable, what must dominate what, how the approach unfolds, what may be
    compressed and what stays uncertain."""
    doc = _jload(_p(rnd, "references.json")) or {}
    b = doc.get("brief") if isinstance(doc, dict) else None
    if not b:
        return "(no reference brief on record)"
    if isinstance(b, str):
        return b
    return "\n".join(f"- **{k}**: {v if isinstance(v, str) else json.dumps(v)}"
                     for k, v in b.items())


def _meaning(rnd) -> str:
    """The interpretation's reads, as the design must honour them: hierarchy, features,
    functions and qualities with their scope and whether they are binding. The design
    job reads the meaning itself, not a programme summary of it."""
    doc = _jload(rnd.rel("interpretation.json")) or {}
    rows = []
    for r in doc.get("reads") or []:
        kind = r.get("kind")
        if kind in ("identity",):
            continue
        rows.append(f"- `{r.get('id')}` ({kind}{', binding' if r.get('hard') else ''}"
                    f"{', ' + r['scope'] if r.get('scope') else ''}): {r.get('says')} "
                    f"-- wants {json.dumps(r.get('wants'))}")
    return "\n".join(rows) or "(no interpretation on record)"


def _claims(rnd) -> str:
    p = rnd.rel("reading.claims.json")
    doc = _jload(p) or {}
    rows = []
    for c in (doc.get("claims") or [])[:40]:
        rows.append(f"- `{c.get('id')}` ({c.get('source')}, {c.get('confidence')}): "
                    f"{c.get('says') or c.get('text') or c.get('claim')}")
    return "\n".join(rows) or "(no reading on record)"


def _spec_note(rnd) -> str:
    s = rnd.place_spec() or {}
    if not s:
        return "(no place spec)"
    rings = s.get("defining_parts") or s.get("rings") or []
    parts = [f"kind {s.get('kind')}, form {s.get('form')}, structures band "
             f"{s.get('size_band')} (the spec's programme; the design sets the extent)"]
    for r in rings:
        parts.append(f"  - part `{r.get('name')}` ({r.get('relation')}): density "
                     f"{r.get('density')}, walled {bool(r.get('walled'))}, voice "
                     f"{r.get('voice')}; {(r.get('notes') or '')[:260]}")
    return "\n".join(parts)


DESIGN_PROMPT = """# Design a place: its spatial organisation, before anything is built

Request: **{sentence}**

You are the designer. Write the file named at the end. The compiler (`ethoslm.cityresolve`)
turns your design into walls, streets, blocks, lots, compounds and designed ground, and
tells you what it could not honour. It never invents organisation you did not state.

## What was learned about the request

{spec}

Claims on record (sources in `sources/INDEX.md`):
{claims}

## What the request means (binding where marked)

{meaning}

Every binding read reaches the design. A **hierarchy** read is realised through land,
silhouette, approach and subordination, not only size: state each in the design's
`hierarchy` list (what dominates what, by which means) so the compiler can measure it
and the inspection can judge it. A request with no hierarchy read asks for none.

## Visual references (look at them before you design)

{refs}

### Reference brief (from the `design_references` job, read before allocating land)

{ref_brief}
{baseline}
## The last built candidate and its review

{review}

## Candidate sites (terrain atlas of the delivery save; open the maps)

Ground heights are the top solid block. `fall` is p95-p5 of the ground across the
circle, `rough` the share of 8-block cells with more than {rough} blocks of relief,
`water` the share of wet cells, `usable` dry and smooth. Flatness is measured locally
and across the site, never as a fraction of the side.

{sites}

## Forms available (lot sizes the compiler packs)

{forms}

## Grains a ring or a ward can be composed in

{grains}

## Ground policies

{ground}

## Palettes compared in context (look at the sheets; choose by what you see)

{palette}

Voice names you may use: {voices}

{schema}

## Who decides what

- **You decide:** {model}
- **The compiler resolves:** {compiler}

## Write

`{write}`:

```
{{"brief": {{"silhouette": ..., "grain": ..., "monument": ..., "ground": ..., "scale": ...,
            "references": [...], "decisions": {{"model": [...], "compiler": [...]}}}},
 "proposals": [ design, design, design ]}}
```

Two or three **materially different** whole-place proposals (outline, ring widths,
grains, street hierarchy, monument composition, site or extent) -- not one design with a
parameter moved. Each is a complete `ethoslm.design/1` record with its own `id` and
`site.candidate`. They are compiled and compared on their sites before one is adopted.
"""


def _baseline_note(rnd) -> str:
    """`flags.design.baseline`), or nothing. A revision round keeps what was accepted and
    changes what the request asks changed; its proposals are whole designs all the same,
    compiled and compared in context.
    """
    bp = _cfg(rnd).get("baseline")
    if not bp:
        return ""
    base = _jload(os.path.join(ROOT, bp, "design.json"))
    if not base:
        return ""
    keep = _cfg(rnd).get("keep") or []
    return ("\n## The accepted design this round revises\n\n"
            f"`{os.path.join(bp, 'design.json')}` (id `{base.get('id')}`) is built and "
            "accepted. Each proposal is a whole design that keeps what is accepted and "
            "set `lineage` to "
            f"`{base.get('lineage') or base.get('id')}` so its unchanged choices are "
            "kept, and the compiler keeps unchanged blocks' identity, so only changed "
            "land is rebuilt.\n"
            + ("Keep: " + "; ".join(keep) + "\n" if keep else "")
            + f"\nPlan of the accepted design: `{os.path.join(bp, 'design', 'plan.png')}`\n")


REFERENCES_PROMPT = """# Visual evidence for the design: find it, look at it, say what it shows

Request: **{sentence}** ({why})

The design that follows reads the references you register here and the brief you write
from them. Look at every image you register; a caption or a filename is not evidence
(an image called `Main_hall.png` may show an interior).

## What the request means

{meaning}

## Identity-critical subjects

Decide which built elements carry this place's identity (the ones a person would
recognise it by: its landmark's silhouette and ensemble, its approach, its walls, its
typical street) and which views of each the design needs: an overview of the ensemble
and its setting, the principal exterior, the approach or entrance. Use the evidence you
already have where it answers the question; research only what is missing.

## Evidence already on record

{supplied}

Cached manifests elsewhere in `out/` (reuse an image only with its recorded provenance,
and only if it shows *this* request's subject):
{cached}

## Bounded research, where a view is missing

At most {cap} new images. For each: the page URL and the image URL, title, access date,
and the specific thing it shows; save it under `{dir}` and add a row to
`sources/INDEX.md`. Distinguish what is visible from what you infer: perspectives and
concept art are not surveys, and give relationships, not dimensions. Prefer primary
views of the requested place over look-alikes; a look-alike may inform construction
language, never replace the place's own silhouette.

## Write

`{write}`:

```
{{"views": [{{"file": "<path from the repository root>", "url": "...", "image_url": "...",
             "title": "...", "accessed": "YYYY-MM-DD", "shows": "what is visible",
             "role": "which identity-critical subject this answers", "kind": "..."}}],
 "needs": [{{"subject": "...", "view": "...", "answered_by": ["file", ...] | []}}],
 "brief": {{"recognisable": "what makes it recognisable",
           "dominance": "what must dominate what, and by which means",
           "approach": "how the approach unfolds: thresholds, reveals, framed views",
           "secondary": "which secondary elements establish identity",
           "compress": "what may be compressed at this scale",
           "uncertain": "what the evidence does not settle"}}}}
```

`views` may be empty for a request whose identity needs no image; say so in `brief`.
"""

#: new images one references job may retrieve; a bound, not a target
REFERENCE_CAP = 8


def stage_design_references(rnd, be, results: dict) -> dict:
    """The visual evidence the design needs, found and read before it is designed.

    Staged for a named place or tradition (`evidence.needs_evidence`) and for any round
    handed a manifest to read; skipped for a self-contained request, which uses the
    supported forms at its own scale with no image search. The agent names the
    identity-critical subjects, reuses cached evidence with its provenance, retrieves
    what is missing (bounded), registers the images and writes the reference brief the
    design job reads."""
    from .. import evidence
    cfg = _cfg(rnd)
    what = evidence.classify(rnd.sentence or "")
    supplied = cfg.get("references")
    if cfg.get("references_job") is False or (
            not supplied and not evidence.needs_evidence(rnd.sentence or "")):
        return {"skipped": f"{what['kind']}: {what['why']}; no visual research"}
    ans = _p(rnd, "references.json")
    if os.path.exists(ans):
        doc = _jload(ans) or {}
        errs = []
        for i, v in enumerate(doc.get("views") or []):
            f = v.get("path") or v.get("file") or ""
            if not os.path.exists(os.path.join(ROOT, f)):
                errs.append(f"views[{i}]: {f!r} is not a file")
            for k in ("url", "title", "accessed", "shows"):
                if not v.get(k):
                    errs.append(f"views[{i}] ({f}): {k} is required")
        if not isinstance(doc.get("brief"), (dict, str)) or not doc.get("brief"):
            errs.append("brief is required")
        if errs:
            n = len([f for f in os.listdir(_p(rnd)) if f.startswith("references.rejected")]) + 1
            os.replace(ans, _p(rnd, f"references.rejected.{n}.json"))
            return _needs("design_references", rnd.rel("references_prompt.md"), ans,
                          "handed back: " + "; ".join(errs)[:300])
        return {"views": len(doc.get("views") or []),
                "needs": len(doc.get("needs") or []), "written": ans}
    os.makedirs(_p(rnd, "refs"), exist_ok=True)
    sup = "\n".join(f"- `{os.path.relpath(m, ROOT)}`" for m in _manifests(rnd)) or "(none)"
    import glob
    mine = {os.path.realpath(m) for m in _manifests(rnd)}
    cached = []
    for m in sorted(glob.glob(os.path.join(ROOT, "out", "*", "refs.json")) +
                    glob.glob(os.path.join(ROOT, "out", "*", "design", "references.json"))):
        if os.path.realpath(m) in mine:
            continue
        doc = _jload(m) or []
        views = doc.get("views") if isinstance(doc, dict) else doc
        titles = "; ".join(str(v.get("title") or v.get("file")) for v in (views or [])[:6])
        cached.append(f"- `{os.path.relpath(m, ROOT)}`: {len(views or [])} image(s): "
                      f"{titles}")
    prompt = rnd.rel("references_prompt.md")
    open(prompt, "w").write(REFERENCES_PROMPT.format(
        sentence=rnd.sentence, why=what["why"], meaning=_meaning(rnd), supplied=sup,
        cached="\n".join(cached) or "(none)", cap=REFERENCE_CAP,
        dir=os.path.relpath(_p(rnd, "refs"), ROOT), write=os.path.relpath(ans, ROOT)))
    return _needs("design_references", prompt, ans,
                  "identity-critical views, their evidence and the reference brief")


def stage_design(rnd, be, results: dict) -> dict:
    """The design job, staged; or its answer, read and checked."""
    from .. import citydesign as CD, voices as V
    os.makedirs(_p(rnd), exist_ok=True)
    ans = rnd.rel("design.proposals.json")
    prompt = rnd.rel("design_prompt.md")
    if os.path.exists(ans):
        doc = json.load(open(ans))
        errs = []
        for i, pr in enumerate(doc.get("proposals") or []):
            try:
                CD.read(pr)
            except CD.DesignError as e:
                errs.append(f"proposals[{i}] ({pr.get('id')}): {e}")
            if not (pr.get("site", {}).get("candidate") or pr.get("site", {}).get("centre")):
                errs.append(f"proposals[{i}] ({pr.get('id')}): site.candidate is required")
        if not doc.get("proposals"):
            errs.append("no proposals")
        if errs:
            n = len([f for f in os.listdir(rnd.state)
                     if f.startswith("design.proposals.rejected")]) + 1
            os.replace(ans, rnd.rel(f"design.proposals.rejected.{n}.json"))
            with open(prompt, "a") as fh:
                fh.write("\n\n---\n\n## Your design was read and does not compile\n\n"
                         + "\n".join(f"- {e}" for e in errs) + "\n")
            if n >= 3:
                return {"status": "error", "stop": True, "error": "; ".join(errs)}
            return _needs("design", prompt, ans, "handed back: " + "; ".join(errs)[:300])
        return {"proposals": [p.get("id") for p in doc["proposals"]],
                "written": ans}
    sites = site_candidates(rnd)
    site_rows = "\n".join(
        f"- **{s['id']}**: centre {s['centre']}, radius {s['radius']}: fall {s['fall']}, "
        f"median y {s['median']}, water {s['water']:.1%}, rough {s['rough']:.1%}, usable "
        f"{s['usable']:.1%}"
        + (("; biomes " + ", ".join(f"{k} {v:.0%}" for k, v in sorted(
            s["biomes"].items(), key=lambda kv: -kv[1])[:4])) if s.get("biomes") else "")
        + f"; map `{s['map']}`" for s in sites) or "(no candidate sites)"
    rec_sites = _jload(_p(rnd, "sites", "sites.json")) or {}
    held = (rec_sites.get("setting") or {})
    if held.get("biome"):
        site_rows = (f"Held to the setting's biomes ({', '.join(held['biome'])}, at least "
                     f"half of each circle): "
                     + ("yes." if held.get("held") else
                        "**no** -- this atlas has no biome layer and its save is not on "
                        "record, so the candidates ignore the setting.")
                     + "\n\n" + site_rows)
    review_p = _cfg(rnd).get("review")
    review = (open(os.path.join(ROOT, review_p)).read()
              if review_p and os.path.exists(os.path.join(ROOT, review_p)) else "(none)")
    sheets = _palette_sheets(rnd)
    grains = "\n".join(f"- `{g}`: {v['what']}. Params: " +
                       ", ".join(f"{k} {spec[1:] if spec[0] != 'choice' else spec[1]}"
                                 for k, spec in v["params"].items())
                       for g, v in CD.GRAINS.items())
    ground = "\n".join(f"- `{g}`: {v}" for g, v in CD.GROUND.items())
    from .. import atlas
    text = DESIGN_PROMPT.format(
        sentence=rnd.sentence, spec=_spec_note(rnd), claims=_claims(rnd), refs=_refs(rnd),
        meaning=_meaning(rnd), ref_brief=_ref_brief(rnd), baseline=_baseline_note(rnd),
        review=review, sites=site_rows, rough=atlas.ROUGH, forms=_forms_card(),
        grains=grains, ground=ground,
        palette="\n".join(f"- `{s}`" for s in sheets) or "(no sheets)",
        voices=", ".join(sorted(V.names())), schema=CD.SCHEMA_DOC,
        model=MODEL_DECIDES, compiler=COMPILER_RESOLVES,
        write=os.path.relpath(ans, ROOT))
    open(prompt, "w").write(text)
    return _needs("design", prompt, ans, "the whole-place design: 2-3 proposals")


MODEL_DECIDES = ("silhouette and boundary outline; extent within the measured bounds; "
                 "site; rings and their purposes; street hierarchy and orientation; each "
                 "ring's grain and parameters and its wards; landmarks and where they "
                 "stand; the monument's axial sequence and wings; ground policy per ring; "
                 "the palette, after looking at the sheets")
COMPILER_RESOLVES = ("coordinates, wall paths and gate runs; street rasters; blocks; lots "
                     "and their sites from the forms' sizes; compound dimensions; the "
                     "designed ground and its costs; construction regions; every "
                     "shortfall as a finding addressed to a design field")


# ---------------------------------------------------------------- compare

def _current(rnd) -> dict | None:
    """What the proposals are measured against: the accepted resolution the round
    names (`flags.design.baseline`), or nothing. A fresh request has no incumbent and is
    compared only proposal against proposal."""
    bp = _cfg(rnd).get("baseline")
    rec = _jload(os.path.join(ROOT, bp, "city.json")) if bp else None
    if not rec:
        return None
    m = rec.get("metrics") or {}
    return {"name": f"{rec.get('design')} {rec.get('design_digest')}", "side": 2 * rec["radius"],
            "area": m.get("area") or 1, "parts": m.get("leaves"),
            "dwellings": m.get("dwellings"), "compounds": m.get("compounds"),
            "hierarchy": m.get("hierarchy"), "state": bp}


def _site_of(rnd, pr: dict) -> dict | None:
    sid = (pr.get("site") or {}).get("candidate")
    if (pr.get("site") or {}).get("centre"):
        return {"id": sid or "given", "centre": list(pr["site"]["centre"]),
                "radius": pr["extent"]["radius"]}
    for s in site_candidates(rnd):
        if s["id"] == sid:
            return s
    c = (pr.get("site") or {}).get("centre")
    if c:
        return {"id": "given", "centre": list(c), "radius": pr["extent"]["radius"]}
    return None


def compile_on_site(rnd, design: dict):
    """A design compiled on the ground of its chosen site."""
    from .. import citydesign as CD, cityresolve as C
    d = CD.read(design)
    site = _site_of(rnd, d)
    if not site:
        raise CD.DesignError(f"{d.get('id')}: site.candidate names no candidate site")
    centre = tuple(d.get("site", {}).get("centre") or site["centre"])
    R = d["extent"]["radius"]
    X0, Z0 = int(centre[0]) - R - C.MARGIN, int(centre[1]) - R - C.MARGIN
    W = 2 * (R + C.MARGIN) + 1
    ground = site_ground(rnd, (X0, Z0, W))
    return C.compile_design(d, centre, ground=ground, prior=_prior(rnd, (X0, Z0, W))), site


def _prior(rnd, frame) -> "np.ndarray | None":
    """The block raster of the latest resolution of this place on the same frame. Blocks
    whose land is unchanged keep their identity (`cityresolve._stable_ids`), so a
    revision rebuilds only what it changed.
    """
    cands = [rnd.rel("city.npz")]
    for key in ("identity", "baseline"):
        bp = _cfg(rnd).get(key)
        if bp:
            cands.append(os.path.join(ROOT, bp, "city.npz"))
    for c in cands:
        if not os.path.exists(c):
            continue
        z = np.load(c)
        o = [int(v) for v in z["origin"]]
        if o == [frame[0], frame[1]] and z["block"].shape == (frame[2], frame[2]) \
                and "open_id" in z:
            return {"block": z["block"], "open": z["open_id"]}
    return None


def _cost(city) -> dict:
    """Execution estimate: parts by kind and the per-part seconds measured so far."""
    rec = _jload(os.path.join(ROOT, "out", "ds-work", "costs.json")) or {}
    per = float(rec.get("seconds_per_plot", 9.0))
    per_edge = float(rec.get("seconds_per_edge_run", 20.0))
    plots = sum(1 for lf in city.leaves if lf["kind"] == "plot")
    areas = sum(1 for lf in city.leaves if lf["kind"] == "area")
    edges = sum(1 for lf in city.leaves if lf["kind"] == "edge")
    secs = plots * per + areas * per * 0.3 + edges * per_edge
    return {"plots": plots, "areas": areas, "edges": edges,
            "worker_hours": round(secs / 3600, 1), "regions": len(city.regions),
            "per_plot_s": per, "from": rec.get("from", "a prior estimate, not yet measured")}


def stage_design_compare(rnd, be, results: dict) -> dict:
    from .. import citydesign as CD, cityresolve as C
    doc = _jload(rnd.rel("design.proposals.json"))
    if not doc:
        return {"status": "error", "stop": True, "error": "no design.proposals.json"}
    out = _p(rnd, "compare")
    os.makedirs(out, exist_ok=True)
    rows = []
    cur = _current(rnd)
    for pr in doc["proposals"]:
        t0 = time.perf_counter()
        city, site = compile_on_site(rnd, pr)
        png = C.preview(city, os.path.join(out, f"{pr['id']}.png"))
        m = city.metrics
        from .. import designground as DG
        est = DG.estimate(city.ground[0], city.target, city.treat, found_wet=city.wet,
                          pave=city.pave)
        m["ground"] = {**m["ground"], "cut": int(est.get("cut_blocks") or 0),
                       "fill": int(est.get("fill_blocks") or 0),
                       "street_steps": int(est.get("street_steps_unresolved") or 0),
                       "over_bound": int(est.get("over_bound_count") or 0),
                       "retained_length": int(est.get("retained_length") or 0),
                       "tallest_face": est.get("max_retain")}
        rows.append({"id": pr["id"], "site": site.get("id"), "centre": [city.cx, city.cz],
                     "radius": city.R, "area": m["area"],
                     "area_vs_current": round(m["area"] / cur["area"], 2) if cur else None,
                     "hierarchy": m.get("hierarchy"), "demand": getattr(city,
                                                                        "monument_demand", None),
                     "dwellings": m["dwellings"], "compounds": m["compounds"],
                     "people_estimate": m["people_estimate"],
                     "by_type": m["by_type"], "rings": m["rings"], "walls": m["walls"],
                     "ground": m["ground"], "findings": city.findings[:30],
                     "finding_counts": m["findings"], "cost": _cost(city),
                     "preview": os.path.relpath(png, ROOT),
                     "seconds": round(time.perf_counter() - t0, 1)})
        if _cfg(rnd).get("massing", True) and getattr(city, "monument", None):
            mdir = os.path.join(out, f"{pr['id']}.massing")
            mrec = _jload(os.path.join(mdir, "massing.json"))
            key = CD.digest(pr) + "." + _compiler_digest() + "." + _code_digest()
            if not mrec or mrec.get("key") != key:
                mrec = massing(rnd, city, mdir)
                mrec["key"] = key
                json.dump(mrec, open(os.path.join(mdir, "massing.json"), "w"), indent=1,
                          default=str)
            rows[-1]["massing"] = mrec.get("contact")
            rows[-1]["massing_views"] = [v["png"] for v in mrec.get("views") or []]
    rec = {"current": cur, "proposals": rows}
    json.dump(rec, open(os.path.join(out, "compare.json"), "w"), indent=1)
    md = os.path.join(out, "compare.md")
    open(md, "w").write(_compare_md(rec))
    adopt_p = rnd.rel("design.adopt.json")
    if not os.path.exists(adopt_p):
        prompt = rnd.rel("design_adopt_prompt.md")
        open(prompt, "w").write(
            "# Adopt one design\n\nThe proposals, compiled on their sites (open every "
            "preview; each is the plan the compiler resolved, over the designed ground's "
            "shading):\n\n" + open(md).read() +
            "\n\n## Write\n\n`" + os.path.relpath(adopt_p, ROOT) + "`:\n\n```\n"
            '{"adopt": "proposal id", "set": {"rings[2].params.street_every": 5, ...},\n'
            ' "palette": {"rings": {...}, "walls": "...", "monument": "...", "why": "..."},\n'
            ' "why": "what the previews showed and why this one"}\n```\n\n`set` revises '
            "named fields of the adopted proposal (dotted paths, list indices in "
            "brackets); every revision is recorded.\n")
        return _needs("design_adopt", prompt, adopt_p, "choose one proposal", compared=rows)
    adopt = json.load(open(adopt_p))
    chosen = next((p for p in doc["proposals"] if p["id"] == adopt.get("adopt")), None)
    if chosen is None:
        return {"status": "error", "stop": True,
                "error": f"design.adopt.json adopts {adopt.get('adopt')!r}, not a proposal"}
    design = apply_set(chosen, adopt.get("set") or {})
    if adopt.get("palette"):
        design["palette"] = {**(design.get("palette") or {}), **adopt["palette"]}
    design["adopted"] = {"from": chosen["id"], "set": adopt.get("set") or {},
                         "why": adopt.get("why")}
    CD.read(design)
    _write_design(rnd, design, "adopted from the compared proposals")
    return {"compared": [r["id"] for r in rows], "adopted": chosen["id"],
            "compare": os.path.relpath(md, ROOT)}


def _write_design(rnd, design: dict, why: str) -> None:
    """`design.json` is the adopted design; every version is kept beside it."""
    from .. import citydesign as CD
    p = rnd.rel("design.json")
    hist = _p(rnd, "history")
    os.makedirs(hist, exist_ok=True)
    dg = CD.digest(design)
    if os.path.exists(p):
        was = json.load(open(p))
        if CD.digest(was) == dg:
            return
        shutil.copyfile(p, os.path.join(hist, f"design.{CD.digest(was)}.json"))
    design = dict(design)
    json.dump(design, open(p, "w"), indent=1)
    log = _jload(os.path.join(hist, "log.json"), [])
    log.append({"digest": dg, "why": why, "t": time.strftime("%Y-%m-%dT%H:%M:%S")})
    json.dump(log, open(os.path.join(hist, "log.json"), "w"), indent=1)


def apply_set(design: dict, sets: dict) -> dict:
    """`design` with each dotted path in `sets` replaced (`rings[2].params.court`)."""
    import copy
    import re
    d = copy.deepcopy(design)
    for path, value in sets.items():
        toks = [t for t in re.split(r"\.|\[|\]", path) if t != ""]
        cur = d
        for t in toks[:-1]:
            cur = cur[int(t)] if isinstance(cur, list) else cur.setdefault(t, {})
        last = toks[-1]
        if isinstance(cur, list):
            cur[int(last)] = value
        else:
            cur[last] = value
    return d


def _compare_md(rec: dict) -> str:
    cur = rec["current"]
    lines = [(f"Accepted design ({cur['name']}): {cur['side']} across, {cur['area']:,} "
              f"columns, {cur['parts']} parts, about {cur['dwellings']} dwellings.\n")
             if cur else "No incumbent: the proposals are compared with each other.\n",
             "| proposal | site | radius | area (x current) | dwellings | compounds | "
             "people est. | cut / fill | street steps / faces over bound | findings b/d/i "
             "| est. worker-hours | preview |",
             "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in rec["proposals"]:
        f = r["finding_counts"]
        g = r["ground"]
        lines.append(f"| {r['id']} | {r['site']} | {r['radius']} | {r['area']:,} "
                     f"({r['area_vs_current']}x) | {r['dwellings']} | {r['compounds']} | "
                     f"{r['people_estimate']:,} | {g['cut']:,} / {g['fill']:,} | "
                     f"{g.get('street_steps', '?')} / {g.get('over_bound', '?')} | "
                     f"{f['blocking']}/{f['design']}/{f['info']} | "
                     f"{r['cost']['worker_hours']} | `{r['preview']}` |")
    for r in rec["proposals"]:
        lines.append(f"\n### {r['id']}\n")
        if r.get("massing"):
            lines.append(f"- **massing in context** (the production forms, built round "
                         f"the accepted city; open it): `{r['massing']}`")
        h = r.get("hierarchy") or {}
        if h:
            lines.append(f"- hierarchy, as planned: {json.dumps({k: v for k, v in h.items() if k != 'statements'})}")
        dm = r.get("demand") or {}
        if dm:
            lines.append(f"- the monument asks an interior radius of {dm.get('needs_r_in')} "
                         f"(the ring gives {dm.get('r_in')}; axis depth {dm.get('depth')})")
        for ring in r["rings"]:
            lines.append(f"- {ring['name']}: width {ring['width']}, level {ring['level']}, "
                         f"built {ring['built_share']:.0%}, streets "
                         f"{ring['street_share']:.0%}, open {ring['open_share']:.0%}")
        for fd in r["findings"][:12]:
            lines.append(f"- finding ({fd['severity']}, `{fd['field']}`): {fd['what']}")
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------- resolve

def stage_design_resolve(rnd, be, results: dict) -> dict:
    """The adopted design, compiled for construction."""
    from .. import citydesign as CD, cityresolve as C
    dp = rnd.rel("design.json")
    if not os.path.exists(dp):
        return {"status": "error", "stop": True, "error": "no adopted design.json"}
    design = json.load(open(dp))
    # a revision answer is applied before the compile
    rev_p = rnd.rel("design.revision.json")
    if os.path.exists(rev_p):
        rev = json.load(open(rev_p))
        design = apply_set(design, rev.get("set") or {})
        design.setdefault("revisions", []).append({"set": rev.get("set"),
                                                   "why": rev.get("why")})
        _write_design(rnd, design, f"revision: {rev.get('why')}")
        os.replace(rev_p, _p(rnd, "history", f"revision.{int(time.time())}.json"))
    dg = CD.digest(design) + "." + _compiler_digest()
    cr = _jload(rnd.rel("city.json"))
    if cr and cr.get("resolved_by") == dg and os.path.exists(rnd.rel("plan.json")):
        return {"skipped": "the adopted design is unchanged since it was resolved",
                "design_digest": dg, "regions": len(cr.get("regions") or {})}
    t0 = time.perf_counter()
    city, site = compile_on_site(rnd, design)
    rec = C.save(city, rnd.state)
    rec["resolved_by"] = dg
    json.dump(rec, open(rnd.rel("city.json"), "w"), indent=1)
    C.preview(city, _p(rnd, "plan.png"))
    net = C.network(city)
    net.save(rnd.rel("network.json"))
    # the registry every builder and the lint read: one row per footprint leaf
    plots = [{"label": lf["name"], "kind": lf["kind"], "type": lf["type"],
              "x0": lf.get("x0", (lf.get("at") or [0, 0])[0]),
              "z0": lf.get("z0", (lf.get("at") or [0, 0])[1]),
              "x1": lf.get("x1", (lf.get("at") or [0, 0])[0]),
              "z1": lf.get("z1", (lf.get("at") or [0, 0])[1])}
             for lf in city.leaves if lf["kind"] in ("plot", "area", "point")]
    json.dump(plots, open(rnd.rel("plots.json"), "w"))
    json.dump({"origin": [city.X0, city.Z0], "size": city.W, "centre": [city.cx, city.cz],
               "radius": city.R, "by": "stage_design_resolve"},
              open(rnd.rel("site.json"), "w"), indent=1)
    # the designed ground, estimated whole before any region is built: unresolved street
    # steps and over-bound retaining walls are the design's to change
    from .. import designground as DG
    est = DG.estimate(city.ground[0], city.target, city.treat, found_wet=city.wet,
                      pave=city.pave)
    rec["ground_estimate"] = est
    json.dump(rec, open(rnd.rel("city.json"), "w"), indent=1)
    n_steps = int(est.get("street_steps_unresolved") or 0)
    n_over = int(est.get("over_bound_count") or 0)
    if n_steps or n_over:
        city.find("design", "rings[*].ground",
                  f"the designed ground leaves {n_steps} street step(s) of more than one "
                  f"block (largest {est.get('max_unresolved_step')}) and {n_over} "
                  f"retaining face(s) over the bound (tallest {est.get('max_retain')})",
                  steps=n_steps, over_bound=n_over)
        rec["findings"] = city.findings
        counts = {sv: sum(1 for f in city.findings if f["severity"] == sv)
                  for sv in ("blocking", "design", "info")}
        city.metrics["findings"] = counts
        rec["metrics"]["findings"] = counts
        json.dump(rec, open(rnd.rel("city.json"), "w"), indent=1)
    blocking = [f for f in city.findings if f["severity"] == "blocking"]
    print(f"   resolved {design.get('id')}: {len(city.leaves)} leaves, "
          f"{city.metrics['dwellings']} dwellings, {len(city.regions)} regions, findings "
          f"{city.metrics['findings']} in {time.perf_counter() - t0:.1f}s", flush=True)
    if blocking:
        prompt = rnd.rel("design_revision_prompt.md")
        open(prompt, "w").write(
            "# The adopted design has findings the compiler cannot resolve\n\n" +
            "\n".join(f"- `{f['field']}`: {f['what']}" for f in blocking) +
            "\n\nWrite `" + os.path.relpath(rev_p, ROOT) + "` as "
            '`{"set": {"dotted.path": value}, "why": "..."}` to revise the named fields.\n')
        return _needs("design_revision", prompt, rev_p, "blocking design findings",
                      findings=blocking)
    return {"design": design.get("id"), "design_digest": dg, "leaves": len(city.leaves),
            "regions": len(city.regions), "metrics": city.metrics,
            "findings": city.metrics["findings"], "plan": rnd.rel("plan.json")}


# ---------------------------------------------------------------- regions

REGION_MARGIN = 40


def _load_city(rnd):
    """The resolved city's record, rasters and leaves (no recompile)."""
    rec = json.load(open(rnd.rel("city.json")))
    ras = np.load(rnd.rel("city.npz"))
    plan = json.load(open(rnd.rel("plan.json")))
    from .. import pipeline
    leaves = pipeline.plan_parts(plan)
    return rec, ras, leaves


def _run_view(lf: dict) -> dict:
    """A ring wall run as far as its own build can depend on it. A run carries its
    whole ring (path, floors, gates) so runs built apart meet; but the walk's level is
    an envelope falling one half-block a station (`types/wall._cw_envelope`), so a
    floor can reach at most twice the ring's floor range, plus a gate's flat run, along
    the wall. Floors of segments farther than that (in the plane, which is never
    farther than along the wall) cannot change this run, and a revision elsewhere on
    the ring does not rebuild it."""
    rf = lf.get("ring_floors")
    rp = lf.get("ring_path")
    if not rf or not rp or len(rf) != len(rp) - 1:
        return lf
    reach = 2 * (max(rf) - min(rf)) + 64
    own = lf.get("path") or []
    near = {}
    for i, (a, b) in enumerate(zip(rp, rp[1:])):
        mx, mz = (a[0] + b[0]) / 2, (a[1] + b[1]) / 2
        half = math.hypot(b[0] - a[0], b[1] - a[1]) / 2
        if any(math.hypot(mx - q[0], mz - q[1]) - half <= reach for q in own):
            near[str(i)] = rf[i]
    return {**lf, "ring_floors": near}


def _leaf_digest(leaves: list) -> str:
    view = [_run_view(lf) if lf.get("kind") == "edge" else lf for lf in leaves]
    return hashlib.sha256(json.dumps(sorted(view, key=lambda l: l["name"]),
                                     sort_keys=True).encode()).hexdigest()[:16]


def _compiler_digest() -> str:
    """The compiler a resolution was made by: a changed compiler re-resolves."""
    h = hashlib.sha256()
    for f in ("cityresolve.py", "citydesign.py", "boundary.py"):
        h.update(open(os.path.join(ROOT, "src", "ethoslm", f), "rb").read())
    return h.hexdigest()[:12]


#: the library every region's build runs through, whatever it builds
_LIBRARY = ("buildlib.py", "prims.py", "designground.py")


def _code_digest(types=None, voices=None, root: str = ROOT) -> str:
    """The construction code a region's build depends on: the library that builds and
    grounds, the type files of the forms it instantiates and the voices it builds in.
    `types`/`voices` None is every file (the whole-tree digest); a region passes its
    own, so a changed form rebuilds the regions that use it and no others."""
    h = hashlib.sha256()
    tdir = os.path.join(root, "types")
    names = sorted(f for f in os.listdir(tdir) if f.endswith(".py")) if types is None \
        else sorted(f"{t}.py" for t in set(types))
    for f in names:
        fp = os.path.join(tdir, f)
        h.update(f.encode())
        h.update(open(fp, "rb").read() if os.path.exists(fp) else b"(missing)")
    for f in _LIBRARY:
        p = os.path.join(root, "src", "ethoslm", f)
        if os.path.exists(p):
            h.update(open(p, "rb").read())
    vdir = os.path.join(root, "voices")
    vs = sorted(os.listdir(vdir)) if voices is None else \
        sorted(f"{v}.json" for v in set(voices) if v)
    for f in vs:
        fp = os.path.join(vdir, f)
        h.update(f.encode())
        h.update(open(fp, "rb").read() if os.path.exists(fp) else b"(missing)")
    return h.hexdigest()[:16]


def region_inputs(rnd, rec, ras, leaves, rid: str) -> dict:
    """What a region's build is made from: its own leaves, its owned ground, the code."""
    mine = [lf for lf in leaves if lf.get("region") == rid]
    i, j = (int(v) for v in rid.split("_"))
    from .. import cityresolve as C
    X0, Z0, W = rec["frame"]
    sl = (slice(i * C.REGION, (i + 1) * C.REGION), slice(j * C.REGION, (j + 1) * C.REGION))
    own = ras["owner"][sl] == (i * 1000 + j)
    g = hashlib.sha256()
    for k in ("target", "treat", "pave"):
        g.update(np.where(own, ras[k][sl], 0).tobytes())
    mats = hashlib.sha256(json.dumps(ground_materials(rnd, rec), sort_keys=True,
                                     default=str).encode()).hexdigest()[:12] \
        if rnd is not None else None
    return {"leaves": _leaf_digest(mine), "ground": g.hexdigest()[:16],
            "materials": mats,
            "code": _code_digest([lf["type"] for lf in mine],
                                 [lf.get("voice") for lf in mine]),
            "n_leaves": len(mine)}


def stage_regions(rnd, be, results: dict) -> dict:
    """Build the regions `flags.design.build` names (ids, or `all`), each bounded, each
    resumable; a region whose inputs are unchanged since it was built is kept."""
    cfg = _cfg(rnd)
    want = cfg.get("build") or []
    if not want:
        return {"skipped": "flags.design.build names no regions"}
    rec, ras, leaves = _load_city(rnd)
    rids = sorted(rec["regions"]) if want == "all" else list(want)
    # one stream of several (`ETHOSLM_REGION_STREAM=i/n`): streams build disjoint regions
    # at once. Safe because a region writes only the columns it owns; a neighbour built
    # concurrently is context-only ground to it, exactly as an unbuilt one is.
    stream = os.environ.get("ETHOSLM_REGION_STREAM")
    if stream:
        i_s, n_s = (int(v) for v in stream.split("/"))
        rids = [r for n, r in enumerate(rids) if n % n_s == i_s]
    # every region whose inputs moved is stale context for the others, whichever stream
    # rebuilds it and whenever
    _STALE.clear()
    for r in rec["regions"]:
        was = _jload(rnd.rel("regions", r, "region.json"))
        if not was or was.get("inputs") != region_inputs(rnd, rec, ras, leaves, r):
            _STALE.add(r)
    force = cfg.get("rebuild")
    out = {"built": [], "kept": [], "failed": [], "regions": {}, "stale": sorted(_STALE)}
    for rid in rids:
        if rid not in rec["regions"]:
            out["failed"].append({"region": rid, "why": "not a region of this design"})
            continue
        inp = region_inputs(rnd, rec, ras, leaves, rid)
        d = rnd.rel("regions", rid)
        was = _jload(os.path.join(d, "region.json"))
        forced = force is True or (isinstance(force, list) and rid in force)
        if was and was.get("inputs") == inp and os.path.exists(os.path.join(d, "diff.npz")) \
                and not forced:
            out["kept"].append(rid)
            out["regions"][rid] = {"kept": True, **{k: was.get(k) for k in
                                                    ("built", "failed", "seconds")}}
            print(f"   region {rid}: unchanged since it was built -- kept", flush=True)
            continue
        if was and was.get("inputs") != inp:
            changed = [k for k in inp if (was.get("inputs") or {}).get(k) != inp[k]]
            print(f"   region {rid}: rebuilt, its {', '.join(changed)} changed", flush=True)
        r = build_region(rnd, rec, ras, leaves, rid, inp)
        out["regions"][rid] = {k: r.get(k) for k in ("built", "failed", "seconds",
                                                     "blocks", "ground")}
        (out["built"] if r.get("ok") else out["failed"]).append(rid)
    json.dump(out, open(rnd.rel("regions.json" if not stream else
                                f"regions.{stream.replace('/', 'of')}.json"), "w"), indent=1)
    return out


#: regions whose built diff is stale (its inputs moved; it is to be rebuilt), set by
#: `stage_regions`: a stale neighbour is no context, so a region never builds its edges
#: against -- or leaves standing -- the old design's work on land it now owns
_STALE: set = set()


def _neighbour_diffs(rnd, rid: str, rect) -> list:
    """Built, current regions whose diff reaches into `rect` (context for this build).
    A stale neighbour (`_STALE`) is treated as unbuilt: its ground is designed ground."""
    base = rnd.rel("regions")
    got = []
    if not os.path.isdir(base):
        return got
    for other in sorted(os.listdir(base)):
        if other == rid or other in _STALE:
            continue
        rj = _jload(os.path.join(base, other, "region.json"))
        if not rj or not rj.get("ok"):
            continue
        b = rj.get("bounds")
        if b and not (b[2] < rect[0] or b[0] > rect[2] or b[3] < rect[1] or b[1] > rect[3]):
            got.append(os.path.join(base, other, "diff.npz"))
    return got


def load_diff(path: str) -> tuple:
    d = np.load(path, allow_pickle=False)
    return d["xyz"], d["codes"], [str(s) for s in d["palette"]]


def apply_diff(vol, path: str) -> int:
    xyz, codes, pal = load_diff(path)
    index = {s: i for i, s in enumerate(vol.palette)}
    remap = np.empty(len(pal), np.int64)
    for i, s in enumerate(pal):
        if s not in index:
            index[s] = len(vol.palette)
            vol.palette.append(s)
        remap[i] = index[s]
    x = xyz[:, 0] - vol.x0
    y = xyz[:, 1] - vol.y0
    z = xyz[:, 2] - vol.z0
    sx, sy, sz = vol.codes.shape
    ok = (x >= 0) & (x < sx) & (y >= 0) & (y < sy) & (z >= 0) & (z < sz)
    vol.codes[x[ok], y[ok], z[ok]] = remap[codes[ok]].astype(vol.codes.dtype)
    vol._tables = None
    return int(ok.sum())


def base_volume(rnd, X: int, Z: int, w: int, h: int, y0: int, y1: int):
    """The delivery save's ground as found, cropped vertically."""
    from ..savedworld import SavedWorld
    from ..observe import Volume
    sw = SavedWorld(_terrain(rnd), cache_chunks=64)
    v = sw.volume(X, Z, w, h)
    a, b = y0 - v.y0, y1 - v.y0
    return Volume(X, y0, Z, v.codes[:, a:b, :].copy(), list(v.palette))


def Volume_like(codes, vol):
    """`codes` as a volume on `vol`'s frame and palette (the ground as found)."""
    from ..observe import Volume
    return Volume(vol.x0, vol.y0, vol.z0, codes, list(vol.palette))


class _RegionBackend:
    """A region build's backend: one volume, commits recorded as the region's own."""
    live = False
    dry_run = True

    def __init__(self, vol):
        self._vol = vol
        self.blocks: dict = {}

    @property
    def volume(self):
        return self._vol

    def commit(self, builder):
        pending = dict(getattr(builder, "_pending", {}) or {})
        if pending:
            self._vol = self._vol.overlay(pending)
            self.blocks.update(pending)
        return {"placed": len(pending), "failed": 0, "note": "a region's own build"}


def _order(leaves: list) -> list:
    """Edges first (walls), then points on them (gates), then areas, then plots."""
    rank = {"edge": 0, "point": 1, "area": 2, "plot": 3}
    return sorted(leaves, key=lambda lf: (rank.get(lf["kind"], 4), lf["name"]))


def build_region(rnd, rec, ras, leaves, rid: str, inp: dict) -> dict:
    """One region: ground as found (+ built neighbours as context), the designed ground
    on the columns it owns, then its leaves through `instantiate_part`. Saved as a sparse
    diff over the ground as found; `region.json` records what it was made from."""
    from .. import cityresolve as C, designground as DG, pipeline
    from .stages_build import annotate_gates
    t0 = time.perf_counter()
    X0, Z0, W = rec["frame"]
    i, j = (int(v) for v in rid.split("_"))
    rx0, rz0 = X0 + i * C.REGION, Z0 + j * C.REGION
    rx1, rz1 = rx0 + C.REGION - 1, rz0 + C.REGION - 1
    mine = [lf for lf in leaves if lf.get("region") == rid]
    # the build's window: the region, every leaf of it, and the margin
    xs = [rx0, rx1] + [v for lf in mine for v in _leaf_xs(lf)]
    zs = [rz0, rz1] + [v for lf in mine for v in _leaf_zs(lf)]
    bx0, bz0 = min(xs) - REGION_MARGIN, min(zs) - REGION_MARGIN
    bx1, bz1 = max(xs) + REGION_MARGIN, max(zs) + REGION_MARGIN
    # clip to the frame
    bx0, bz0 = max(bx0, X0), max(bz0, Z0)
    bx1, bz1 = min(bx1, X0 + W - 1), min(bz1, Z0 + W - 1)
    w, h = bx1 - bx0 + 1, bz1 - bz0 + 1
    sl = (slice(bx0 - X0, bx1 - X0 + 1), slice(bz0 - Z0, bz1 - Z0 + 1))
    target = ras["target"][sl]
    treat = ras["treat"][sl]
    pave = ras["pave"][sl]
    owner = ras["owner"][sl]
    from .. import atlas
    found, wsurf, wet = atlas.ground(bx0, bz0, w, h, _atlas_dir(rnd))
    lo = int(min(found.min(), np.where(target > -30000, target, 999).min())) - 12
    hi = int(max(found.max(), target.max())) + 80
    vol = base_volume(rnd, bx0, bz0, w, h, lo, hi)
    base_codes = vol.codes.copy()
    ctx = 0
    for p in _neighbour_diffs(rnd, rid, (bx0, bz0, bx1, bz1)):
        ctx += apply_diff(vol, p)
    # context-only ground for unbuilt neighbours, so edges meet designed ground
    built_ids = set()
    for p in _neighbour_diffs(rnd, rid, (bx0, bz0, bx1, bz1)):
        built_ids.add(os.path.basename(os.path.dirname(p)))
    own = owner == (i * 1000 + j)
    unbuilt_ctx = np.zeros_like(own)
    for q in np.unique(owner):
        if q < 0 or q == i * 1000 + j:
            continue
        if f"{q // 1000}_{q % 1000}" not in built_ids:
            unbuilt_ctx |= owner == q
    mats = ground_materials(rnd, rec)
    gp = DG.plan(found, wet, target, treat, pave=pave)
    g_rec = {}
    if unbuilt_ctx.any():
        vol, _ = DG.apply(vol, bx0, bz0, gp, materials=mats, only=unbuilt_ctx)
    ctx_codes = vol.codes.copy()
    # the ground as found is `base`: a neighbour's build standing over these columns (a
    # wall's tower, an overhanging eave) is left standing, its ground under it kept
    vol, g_rec = DG.apply(vol, bx0, bz0, gp, materials=mats, only=own,
                          base=Volume_like(base_codes, vol))
    # own ground edits: what changed in owned columns
    be = _RegionBackend(vol)
    # the leaves, with the gates annotated onto their wall runs
    parts = _order([dict(lf) for lf in mine])
    annotate_gates(parts)
    voices: dict = {}

    def voice_of(lf):
        v = lf.get("voice")
        if v not in voices:
            roof = pipeline.voice_roof(v)
            if roof is not None and roof.get("chimney") is None:
                roof = dict(roof, chimney=False)
            voices[v] = (pipeline.voice_palette(v), roof)
        return voices[v]
    rows = []
    # walls and gates first, on the region's own volume; then every block's leaves in
    # parallel: lots and compounds of different blocks never share a column a compound's
    # own wall is laid after the buildings inside it, so no pad is levelled to the
    # wall's body; ring walls and their gates come first
    serial = [lf for lf in parts if lf["kind"] in ("edge", "point")
              and lf.get("role") != "compound"]
    last = [lf for lf in parts if lf["kind"] == "edge" and lf.get("role") == "compound"]
    rest = [lf for lf in parts if lf["kind"] not in ("edge", "point")]
    for lf in serial:
        rows.append(_one(rnd, be, lf, voice_of, rid))
    groups: dict = {}
    for lf in rest:
        key = lf.get("block")
        if key is None:
            key = lf["name"].rsplit("_", 1)[0]
        groups.setdefault(key, []).append(lf)
    n_workers = int(os.environ.get("ETHOSLM_DESIGN_WORKERS") or _cfg(rnd).get("workers")
                    or rnd.flags.get("workers") or 1)
    chunks = _balance(list(groups.values()), max(1, n_workers * 2))
    for lf in rest:
        voice_of(lf)
    if n_workers > 1 and len(chunks) > 1:
        from ..parallel import par_map
        global _CTX
        _CTX = (rnd, be.volume, voices, rid)
        got = par_map(_chunk_worker, chunks, n=n_workers)
        _CTX = None
        for res in got:
            be._vol = be._vol.overlay(res["blocks"])
            be.blocks.update(res["blocks"])
            rows.extend(res["rows"])
    else:
        for lf in rest:
            rows.append(_one(rnd, be, lf, voice_of, rid))
    for lf in last:
        rows.append(_one(rnd, be, lf, voice_of, rid))
    vol = be.volume
    # the diff: every cell this region changed, against the context it was built in
    diff_mask = np.zeros(vol.codes.shape, bool)
    own3 = np.broadcast_to(own[:, None, :], vol.codes.shape)
    diff_mask |= own3 & (vol.codes != base_codes)          # own ground and own builds
    diff_mask |= (vol.codes != ctx_codes) & ~own3          # own builds overhanging
    # context-only ground outside the owned columns is not this region's
    xs_, ys_, zs_ = np.nonzero(diff_mask)
    codes = vol.codes[xs_, ys_, zs_]
    uniq, inv = np.unique(codes, return_inverse=True)
    pal = np.array([vol.palette[int(u)] for u in uniq])
    d = rnd.rel("regions", rid)
    os.makedirs(d, exist_ok=True)
    np.savez_compressed(os.path.join(d, "diff.npz"),
                        xyz=np.stack([xs_ + vol.x0, ys_ + vol.y0, zs_ + vol.z0], 1)
                        .astype(np.int32), codes=inv.astype(np.int32), palette=pal)
    built = sum(1 for r in rows if r["status"] == "built")
    failed = [r for r in rows if r["status"] != "built"]
    out = {"region": rid, "ok": True, "rect": [rx0, rz0, rx1, rz1],
           "window": [bx0, bz0, bx1, bz1],
           "bounds": [int(xs_.min() + vol.x0) if len(xs_) else rx0,
                      int(zs_.min() + vol.z0) if len(zs_) else rz0,
                      int(xs_.max() + vol.x0) if len(xs_) else rx1,
                      int(zs_.max() + vol.z0) if len(zs_) else rz1],
           "inputs": inp, "leaves": len(mine), "built": built, "failed": len(failed),
           "failures": failed[:40], "blocks": int(len(xs_)), "context_cells": ctx,
           "ground": g_rec, "delivered": delivered(rows), "rows": rows,
           "seconds": round(time.perf_counter() - t0, 1),
           "t": time.strftime("%Y-%m-%dT%H:%M:%S")}
    json.dump(out, open(os.path.join(d, "region.json"), "w"), indent=1, default=str)
    print(f"   region {rid}: {built}/{len(mine)} leaves built, {failed and len(failed)} "
          f"failed, {len(xs_):,} cells in {out['seconds']}s", flush=True)
    return out


_CTX = None


def _one(rnd, be, lf, voice_of, rid) -> dict:
    from .stages_build import instantiate_part
    mat, roof = voice_of(lf)
    r = instantiate_part(rnd, be, lf, mat, roof, paths_sink=[], ground=None)
    r.pop("surface_record", None)
    if r.get("status") != "built":
        print(f"   region {rid}/{lf['name']}: {r.get('status')} -- "
              f"{str(r.get('error') or r.get('preflight'))[:160]}", flush=True)
    row = {k: r.get(k) for k in ("part", "status", "type", "blocks", "seconds", "error",
                                 "floor_y", "door", "way_in", "stood", "type_said")}
    em = r.get("emitted") or {}
    if em:
        # what the part delivered, measured at emission (`construction.outcome`): the
        # features it was asked for and whether each was found standing
        row["delivered"] = {"storeys": em.get("storeys"),
                            "features": em.get("features") or {},
                            "source": em.get("features_source") or {},
                            "omitted": em.get("omitted") or []}
    return row


def delivered(rows: list) -> dict:
    """Per type, per feature: `[verified, claimed]` over a region's rows -- a feature is
    verified where the emitted blocks show it standing, not where the type said so."""
    out: dict = {}
    for r in rows:
        d = r.get("delivered") or {}
        t = r.get("type") or "?"
        for f, ok in (d.get("features") or {}).items():
            if f == "chimney":
                continue
            v = out.setdefault(t, {}).setdefault(f, [0, 0])
            v[1] += 1
            v[0] += bool(ok) and (d.get("source") or {}).get(f) in ("verified", "measured")
    return out


def _chunk_worker(chunk: list) -> dict:
    """One share of a region's blocks, in a forked process off the region's volume."""
    rnd, vol, voices, rid = _CTX
    be = _RegionBackend(vol)
    rows = [_one(rnd, be, lf, lambda l: voices[l.get("voice")], rid) for lf in chunk]
    return {"blocks": be.blocks, "rows": rows}


def _balance(groups: list, n: int) -> list:
    """Groups of leaves into `n` chunks of about equal leaf count, groups kept whole."""
    bins = [[] for _ in range(n)]
    for g in sorted(groups, key=len, reverse=True):
        min(bins, key=len).extend(g)
    return [b for b in bins if b]


def _leaf_xs(lf):
    if lf["kind"] == "edge":
        return [p[0] for p in lf["path"]]
    if lf["kind"] == "point":
        return [lf["at"][0] - 8, lf["at"][0] + 8]
    return [lf["x0"], lf["x1"]]


def _leaf_zs(lf):
    if lf["kind"] == "edge":
        return [p[1] for p in lf["path"]]
    if lf["kind"] == "point":
        return [lf["at"][1] - 8, lf["at"][1] + 8]
    return [lf["z0"], lf["z1"]]


#: The designed ground's own skin, by the setting's surface word: what graded land is
#: covered in and filled with where the place stands in that country. Green is the
#: library's old default and is unchanged.
SURFACE_GROUND = {
    "green": {},
    "sand": {"cover": "sand", "fill": "sandstone", "path": "sandstone"},
    "badlands": {"cover": "red_sand", "fill": "red_sandstone", "path": "red_sandstone"},
    "snow": {"cover": "snow_block", "fill": "dirt", "path": "packed_ice"},
    "stone": {"cover": "stone", "fill": "stone", "path": "gravel"},
    "earth": {"cover": "coarse_dirt", "fill": "dirt", "path": "dirt_path"},
}


def ground_materials(rnd, rec) -> dict:
    """What the designed ground is made of: the library's defaults, the land's own skin
    by the setting's surface (`SURFACE_GROUND`), the design's `palette.ground` voice for
    the made ground -- streets, retaining walls, their caps and banks -- and last what
    the round states outright (`flags.design.ground_materials`)."""
    cfg = _cfg(rnd)
    m = {"fill": "dirt", "cover": "grass_block", "retain": "stone_bricks",
         "cap": "stone_brick_slab", "bank": "stone_bricks", "path": "dirt_path",
         "pave": {1: "polished_andesite", 2: "packed_mud", 3: "stone_bricks"}}
    if rnd is not None:
        m.update(SURFACE_GROUND.get((_setting(rnd) or {}).get("surface") or "green", {}))
        design = _jload(rnd.rel("design.json")) or {}
        gv = (design.get("palette") or {}).get("ground")
        if gv:
            from .. import prims
            from .stages_build import voice_palette
            pal = voice_palette(gv) or {}
            if pal:
                # the roles are blocks as well as families: a cube is laid as named
                made = pal.get("ground") or pal["floor"]
                m.update({"retain": pal["footing"], "bank": pal["footing"],
                          "cap": prims.shape(pal["trim"], "slab"),
                          "pave": {1: pal["floor"], 2: made, 3: pal["trim"]}})
    for k, v in (cfg.get("ground_materials") or {}).items():
        if k == "pave":
            m["pave"].update({int(a): b for a, b in v.items()})
        else:
            m[k] = v
    return m


# ---------------------------------------------------------------- assembly and views

def assemble(rnd, rect, *, regions=None, pad: int = 0, state: str | None = None,
             top: int = 90):
    """The ground as found over `rect` with every built region's diff laid over it: what
    the regions built, as one volume, for views, lint and delivery. `state` assembles
    another state's regions (an accepted baseline) over the same ground."""
    from .. import atlas
    x0, z0, x1, z1 = rect
    x0, z0, x1, z1 = x0 - pad, z0 - pad, x1 + pad, z1 + pad
    w, h = x1 - x0 + 1, z1 - z0 + 1
    found, _, _ = atlas.ground(x0, z0, w, h, _atlas_dir(rnd))
    lo = int(found.min()) - 12
    hi = int(found.max()) + top
    vol = base_volume(rnd, x0, z0, w, h, lo, hi)
    base = os.path.join(ROOT, state, "regions") if state else rnd.rel("regions")
    used = []
    for rid in sorted(os.listdir(base)) if os.path.isdir(base) else []:
        if regions and rid not in regions:
            continue
        rj = _jload(os.path.join(base, rid, "region.json"))
        if not rj or not rj.get("ok"):
            continue
        b = rj["bounds"]
        if b[2] < x0 or b[0] > x1 or b[3] < z0 or b[1] > z1:
            continue
        apply_diff(vol, os.path.join(base, rid, "diff.npz"))
        used.append(rid)
    return vol, used


def stage_region_views(rnd, be, results: dict) -> dict:
    """Frames of what the regions built: aerial and street level, off the assembly.
    `flags.design.views` names the cameras, or says `auto`: the place's own cameras
    (`place_cameras`), chosen from its resolution."""
    cfg = _cfg(rnd)
    views = cfg.get("views") or []
    if not views:
        return {"skipped": "flags.design.views names no cameras"}
    import sys
    sys.path.insert(0, os.path.join(ROOT, "scripts"))
    import ca_eye
    from PIL import Image
    out = _p(rnd, "views")
    os.makedirs(out, exist_ok=True)
    rect = cfg.get("view_rect")
    if views == "auto":
        rec = _jload(rnd.rel("city.json"))
        if not rec or not os.path.isdir(rnd.rel("regions")):
            return {"skipped": "nothing built to look at yet"}
        cx, cz = rec["centre"]
        R = rec["radius"] + 16
        rect = rect or [cx - R, cz - R, cx + R, cz + R]
    vol, used = assemble(rnd, rect)
    if views == "auto":
        views = place_cameras(rnd, vol)
    from .. import offline
    offline.save_volume(vol, rnd.rel("world_built.npz"))
    from .. import artifact
    artifact.adopt(rnd.state, rnd.rel("world_built.npz"), kind="structural",
                   why=f"the assembly of built regions {used} over the ground as found")
    got = []
    for v in views:
        img = ca_eye.render(vol, tuple(v["at"]), tuple(v["look"]),
                            size=tuple(v.get("size", (960, 540))),
                            fov=float(v.get("fov", 70)), far=float(v.get("far", 500)))
        p = os.path.join(out, f"{v['name']}.png")
        Image.fromarray(img).save(p)
        got.append(os.path.relpath(p, ROOT))
    json.dump(views, open(os.path.join(out, "cameras.json"), "w"), indent=1)
    return {"views": got, "regions": used}


def _ground_at(vol, x, z, default=64):
    import sys
    sys.path.insert(0, os.path.join(ROOT, "scripts"))
    import ca_eye
    y = ca_eye.ground_y(vol, x, z)
    return default if y is None else y


def place_cameras(rnd, vol) -> list:
    """The cameras a place with no monument is looked at from, chosen from its own
    resolution: two overviews from outside the boundary, a low oblique across it, and
    eye-level views in a street near the middle and at the edge of its central open
    ground. Nothing here knows what the place is."""
    rec = _jload(rnd.rel("city.json"))
    cx, cz = rec["centre"]
    R = rec["radius"]
    g0 = _ground_at(vol, cx, cz)
    cams = [{"name": "overview", "at": [cx + 0.95 * R, g0 + 0.8 * R + 8, cz + 1.05 * R],
             "look": [cx, g0, cz], "fov": 62, "far": 6 * R + 200},
            {"name": "overview_north", "at": [cx - 0.3 * R, g0 + 0.7 * R + 8, cz - 1.3 * R],
             "look": [cx, g0, cz], "fov": 62, "far": 6 * R + 200},
            {"name": "oblique", "at": [cx - 1.2 * R, g0 + 0.25 * R + 6, cz + 0.6 * R],
             "look": [cx, g0 + 4, cz], "fov": 60, "far": 6 * R + 200}]
    z = np.load(rnd.rel("city.npz"))
    from .. import cityresolve as C
    use = z["use"]
    X0, Z0 = (int(v) for v in z["origin"])
    target = z["target"]

    def solid(x, y, zz):
        i, j, k = int(x) - vol.x0, int(y) - vol.y0, int(zz) - vol.z0
        if not (0 <= i < vol.codes.shape[0] and 0 <= j < vol.codes.shape[1]
                and 0 <= k < vol.codes.shape[2]):
            return False
        return str(vol.palette[int(vol.codes[i, j, k])]).split("[")[0] not in (
            "air", "cave_air", "water", "short_grass", "poppy", "dandelion")

    def stand(x, zz, near):
        """The first floor at or above `near - 3` a person stands on: solid under two
        clear courses."""
        for y in range(int(near) - 3, int(near) + 40):
            if solid(x, y, zz) and not solid(x, y + 1, zz) and not solid(x, y + 2, zz):
                return y
        return None

    def run(x, zz, y, ux, uz, n=48):
        """How far along (ux, uz) a person at eye height sees clear from (x, zz)."""
        for k in range(1, n):
            if solid(x + ux * k, y + 2, zz + uz * k) or solid(x + ux * k, y + 1,
                                                              zz + uz * k):
                return k
        return n

    street = np.argwhere(np.isin(use, (C.LANE, C.ROAD))) + [X0, Z0]
    d = np.hypot(street[:, 0] - cx, street[:, 1] - cz)
    pick = street[(d >= 0.4 * R) & (d <= 0.8 * R)]
    top = None
    for (x, zz) in pick[:: max(1, len(pick) // 300)]:
        t = int(target[x - X0, zz - Z0])
        y = stand(x, zz, t if t > -30000 else _ground_at(vol, x, zz))
        if y is None:
            continue
        for (ux, uz) in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            k = run(x, zz, y, ux, uz)
            # a street view looks along a street with houses either side of it
            if k < 12:
                continue                       # a view of the wall in front of it
            walls = sum(solid(x + ux * q + uz * sd, y + 2, zz + uz * q + ux * sd)
                        for q in range(2, min(k, 14)) for sd in (-3, -2, 2, 3))
            score = min(k, 24) + 0.25 * walls
            if top is None or score > top[0]:
                top = (score, int(x), int(zz), y, ux, uz)
    if top:
        _, x, zz, y, ux, uz = top
        # the middle of the cell stood on, not its corner: a camera on a voxel face
        # between a lane and a wall starts inside the wall
        cams.append({"name": "street", "at": [x + 0.5, y + 1.7, zz + 0.5],
                     "look": [x + 0.5 + 30 * ux, y + 2.5, zz + 0.5 + 30 * uz],
                     "fov": 75, "far": 3 * R + 100, "stands": True})
    # from a roof on the edge of the open middle, across it: a flat roof is where a
    # person stands, and the middle is seen over its kerbs and garden walls
    lots = np.argwhere(use == C.LOT) + [X0, Z0]
    if len(lots):
        d = np.hypot(lots[:, 0] - cx, lots[:, 1] - cz)
        best = None
        for (x, zz) in lots[np.argsort(d)[:200]]:
            g = _ground_at(vol, x, zz)
            t = int(target[x - X0, zz - Z0])
            base = t if t > -30000 else g - 8
            if g - base < 4:
                continue                          # not a roof
            y = stand(x, zz, g)
            if y is None:
                continue
            best = (int(x), int(zz), y)
            break
        if best:
            x, zz, y = best
            v = np.array([cx - x, cz - zz], float)
            v /= (np.hypot(*v) or 1.0)
            gc = _ground_at(vol, cx, cz)
            cams.append({"name": "centre", "at": [x + 0.5, y + 1.7, zz + 0.5],
                         "look": [x + 0.5 + 40 * v[0], gc - 2, zz + 0.5 + 40 * v[1]],
                         "fov": 75, "far": 3 * R + 100, "stands": True})
    return cams


def _yaw_pitch(at, look) -> tuple:
    import math as m
    dx, dy, dz = (look[0] - at[0], look[1] - at[1], look[2] - at[2])
    yaw = m.degrees(m.atan2(-dx, dz))
    pitch = -m.degrees(m.atan2(dy, m.hypot(dx, dz)))
    return round(yaw, 1), round(pitch, 1)


def _tp(at, look) -> str:
    yaw, pitch = _yaw_pitch(at, look)
    return f"/tp @s {at[0]:.1f} {at[1]:.1f} {at[2]:.1f} {yaw} {pitch}"


def place_pose(rnd) -> str | None:
    """The overview pose as a `/tp` command, for a person in the delivered save."""
    cams = _jload(_p(rnd, "views", "cameras.json")) or []
    cam = next((c for c in cams if c.get("name") == "overview"), None)
    return _tp(cam["at"], cam["look"]) if cam else None


def place_move(rnd, frames: int = 60, size=(640, 360), seconds: float = 5.0) -> dict:
    """A five-second camera move over the built place, drawn frame by frame off the
    assembled regions into a GIF: from high outside the boundary, swinging round and
    down toward the middle. The same move as start and end `/tp` poses for capture in
    game (a camera path between them over five seconds)."""
    import math as m
    import sys
    sys.path.insert(0, os.path.join(ROOT, "scripts"))
    import ca_eye
    from PIL import Image
    rec = _jload(rnd.rel("city.json"))
    cx, cz = rec["centre"]
    R = rec["radius"]
    vol, used = assemble(rnd, [cx - R - 24, cz - R - 24, cx + R + 24, cz + R + 24])
    g0 = _ground_at(vol, cx, cz)
    out = _p(rnd, "views", "move")
    os.makedirs(out, exist_ok=True)
    poses = []
    ims = []
    for i in range(frames):
        t = i / max(1, frames - 1)
        e = t * t * (3 - 2 * t)                    # ease in and out
        b = m.radians(135 - 75 * e)                # swing round the place
        r = R * (1.55 - 0.75 * e)
        h = g0 + R * (0.95 - 0.6 * e) + 6
        at = [cx + r * m.sin(b), h, cz + r * m.cos(b)]
        look = [cx, g0 + 2 + 4 * e, cz]
        poses.append((at, look))
        img = ca_eye.render(vol, tuple(at), tuple(look), size=tuple(size), fov=62.0,
                            far=float(5 * R + 200))
        im = Image.fromarray(img)
        im.save(os.path.join(out, f"f{i:03d}.png"))
        ims.append(im.convert("P", palette=Image.ADAPTIVE, colors=255))
    gif = os.path.join(out, "move.gif")
    ims[0].save(gif, save_all=True, append_images=ims[1:], loop=0,
                duration=int(1000 * seconds / frames))
    cmds = [_tp(*poses[0]), _tp(*poses[-1])]
    json.dump({"frames": frames, "seconds": seconds, "poses": poses, "commands": cmds},
              open(os.path.join(out, "move.json"), "w"), indent=1)
    return {"gif": os.path.relpath(gif, ROOT), "frames": frames, "seconds": seconds,
            "commands": cmds}


# ---------------------------------------------------------------- composition views

def composition_cameras(design: dict, rec: dict, level) -> list:
    """The cameras a composition is judged from, chosen from the design itself: the
    city from far and high on its principal axis, the approach from the air and at eye
    level outside the gate of the ring round the monument, the monument's own gate, its
    court before the principal hall, an oblique over the ensemble, and aerials across
    the rings round it. The same cameras draw a baseline and every candidate, so views
    compare like with like. `level(x, z)` is the designed ground (or None)."""
    import math as m
    cx, cz = rec["centre"]
    R = rec["radius"]
    rings = design["rings"]
    mon = rec.get("monument") or {}
    k = mon.get("ring")
    ax = (design.get("axis") or {}).get("bearing", 180)
    ux, uz = m.sin(m.radians(ax)), -m.cos(m.radians(ax))

    def rr(i):
        i = max(0, min(len(rings) - 1, i))
        return rings[i]["outer"] * R

    def L(x, z, d=72):
        v = level(x, z)
        return v if v is not None else d
    c0 = L(cx, cz)
    cams = []
    k0 = k if k is not None else 0
    far = rr(k0 + 3)
    cams.append({"name": "city_overview", "at": [cx + ux * far * 1.05, c0 + far * 0.75,
                                                 cz + uz * far * 1.05],
                 "look": [cx, c0, cz], "fov": 62, "far": 2600})
    r1 = rr(k0 + 1)
    cams.append({"name": "approach_aerial", "at": [cx + ux * (r1 + 70), c0 + 95,
                                                   cz + uz * (r1 + 70)],
                 "look": [cx - ux * rr(k0) * 0.2, c0 + 18, cz - uz * rr(k0) * 0.2],
                 "fov": 60, "far": 1400})
    gx, gz = cx + ux * (r1 + 30), cz + uz * (r1 + 30)
    cams.append({"name": "approach_gate", "at": [gx, L(gx, gz) + 2.6, gz],
                 "look": [cx, L(gx, gz) + 22, cz], "fov": 70, "far": 900})
    if k is not None:
        r0 = rr(k)
        px, pz = cx + ux * (r0 + 28), cz + uz * (r0 + 28)
        cams.append({"name": "palace_gate", "at": [px, L(px, pz) + 2.6, pz],
                     "look": [cx, L(px, pz) + 20, cz], "fov": 70, "far": 700})
        hall = next((p for p in mon.get("parts") or [] if p["kind"] == "hall"
                     and (p.get("role") in (None, "principal"))), None)
        if hall:
            r = hall["rect"]
            hx, hz = (r[0] + r[2]) / 2, (r[1] + r[3]) / 2
            depth = abs((r[3] - r[1]) if abs(uz) > 0.5 else (r[2] - r[0]))
            ex, ez = hx + ux * (depth / 2 + 38), hz + uz * (depth / 2 + 38)
            cams.append({"name": "court", "at": [ex, L(ex, ez) + 2.6, ez],
                         "look": [hx, L(ex, ez) + 18, hz], "fov": 75, "far": 600})
        ob = m.radians(ax + 40)
        ox, oz = cx + m.sin(ob) * r0 * 1.5, cz - m.cos(ob) * r0 * 1.5
        cams.append({"name": "palace_oblique", "at": [ox, c0 + r0 * 0.75, oz],
                     "look": [cx, c0 + 12, cz], "fov": 62, "far": 1200})
    for tag, db in (("rings_east", 120), ("rings_west", -120)):
        b = m.radians(ax + db)
        r2 = rr(k0 + 2)
        ox, oz = cx + m.sin(b) * r2 * 1.25, cz - m.cos(b) * r2 * 1.25
        cams.append({"name": tag, "at": [ox, c0 + r2 * 0.55, oz],
                     "look": [cx + m.sin(b) * r2 * 0.45, c0, cz - m.cos(b) * r2 * 0.45],
                     "fov": 65, "far": 1400})
    return cams


def render_views(vol, cams, out_dir, size=(960, 540), only=None) -> list:
    """Draw `cams` over `vol` into `out_dir`; eye-level cameras stand on what was built."""
    import sys
    sys.path.insert(0, os.path.join(ROOT, "scripts"))
    import ca_eye
    from PIL import Image
    os.makedirs(out_dir, exist_ok=True)
    got = []
    for c in cams:
        if only and c["name"] not in only:
            continue
        at = list(c["at"])
        # a camera already stood on a floor it chose (`place_cameras`) stays there: the
        # highest block over a roofed lane is its roof, not the lane
        if at[1] < 200 and c.get("fov", 70) >= 70 and not c.get("stands"):
            gy = ca_eye.ground_y(vol, at[0], at[2])
            if gy is not None and gy + 1.6 > at[1] - 1.0:
                at[1] = gy + 2.6
        img = ca_eye.render(vol, (at[0] + .5, at[1], at[2] + .5),
                            (c["look"][0] + .5, c["look"][1], c["look"][2] + .5),
                            size=size, fov=float(c.get("fov", 70)),
                            far=float(c.get("far", 600)))
        p = os.path.join(out_dir, f"{c['name']}.png")
        Image.fromarray(img).save(p)
        got.append({**c, "at": at, "png": os.path.relpath(p, ROOT)})
    return got


def contact_sheet(views: list, path: str, cols: int = 3) -> str:
    from PIL import Image, ImageDraw
    ims = [Image.open(os.path.join(ROOT, v["png"])) for v in views]
    if not ims:
        return ""
    w, h = ims[0].size
    rows = (len(ims) + cols - 1) // cols
    sheet = Image.new("RGB", (w * cols, h * rows), (0, 0, 0))
    for i, (im, v) in enumerate(zip(ims, views)):
        sheet.paste(im, ((i % cols) * w, (i // cols) * h))
        ImageDraw.Draw(sheet).text(((i % cols) * w + 6, (i // cols) * h + 6), v["name"],
                                   fill=(255, 255, 0))
    sheet.save(path)
    return os.path.relpath(path, ROOT)


def massing(rnd, city, out_dir: str, *, only=None) -> dict:
    """A proposal's centre built through the production forms in its surrounding city,
    before any region is: the accepted baseline assembled round it (or the ground as
    found), the monument ring and every landmark of the rings round it cleared to the
    designed ground (`designground`) and built from the compiled leaves, then drawn from
    the composition cameras. Cheap enough to compare proposals; not a region build."""
    from .. import cityresolve as C, designground as DG, pipeline, atlas
    mon = getattr(city, "monument", None)
    if not mon:
        return {"skipped": "no monument"}
    t0 = time.perf_counter()
    k = mon["ring"]
    rings = city.design["rings"]
    reach = int(rings[min(k + 2, len(rings) - 1)]["outer"] * city.R) + 8
    x0, z0 = city.cx - reach, city.cz - reach
    x1, z1 = city.cx + reach, city.cz + reach
    rect = (x0, z0, x1, z1)
    base = _cfg(rnd).get("baseline")
    vol, used = assemble(rnd, rect, state=base, top=120) if base else \
        assemble(rnd, rect, state="(none)", top=120)
    X0, Z0 = city.X0, city.Z0
    sl = (slice(x0 - X0, x1 - X0 + 1), slice(z0 - Z0, z1 - Z0 + 1))
    leaves = pipeline.plan_parts(C.plan_tree(city))

    def in_rect(lf):
        xs, zs = _leaf_xs(lf), _leaf_zs(lf)
        return min(xs) >= x0 and max(xs) <= x1 and min(zs) >= z0 and max(zs) <= z1
    changed = np.zeros((x1 - x0 + 1, z1 - z0 + 1), bool)
    changed |= city.ring[sl] == k
    mine = [lf for lf in leaves if in_rect(lf) and (lf.get("ring") == k or
                                                   lf["name"].startswith("landmark_"))]
    for lf in mine:
        if lf["kind"] in ("plot", "area"):
            changed[lf["x0"] - x0:lf["x1"] - x0 + 1, lf["z0"] - z0:lf["z1"] - z0 + 1] = True
    # the land the centre now takes, cleared of whatever the baseline built on it
    found, wsurf, wet = atlas.ground(x0, z0, x1 - x0 + 1, z1 - z0 + 1, _atlas_dir(rnd))
    air = vol.palette.index("air") if "air" in vol.palette else 0
    stone = len(vol.palette)
    vol.palette.append("stone")
    for (i, j) in zip(*np.nonzero(changed)):
        g = int(found[i, j]) - vol.y0
        vol.codes[i, :, j] = air
        vol.codes[i, :max(0, g + 1), j] = stone
    vol._tables = None
    target = city.target[sl]
    treat = city.treat[sl]
    pave = city.pave[sl]
    gp = DG.plan(found, wet, target, treat, pave=pave)
    vol, _ = DG.apply(vol, x0, z0, gp, materials=ground_materials(rnd, {}), only=changed)
    be = _RegionBackend(vol)
    parts = _order([dict(lf) for lf in mine])
    from .stages_build import annotate_gates
    annotate_gates(parts)
    voices: dict = {}

    def voice_of(lf):
        v = lf.get("voice")
        if v not in voices:
            roof = pipeline.voice_roof(v)
            if roof is not None and roof.get("chimney") is None:
                roof = dict(roof, chimney=False)
            voices[v] = (pipeline.voice_palette(v), roof)
        return voices[v]
    rows = []
    serial = [lf for lf in parts if lf["kind"] in ("edge", "point")
              and lf.get("role") != "compound"]
    last = [lf for lf in parts if lf["kind"] == "edge" and lf.get("role") == "compound"]
    rest = [lf for lf in parts if lf["kind"] not in ("edge", "point")]
    for lf in serial:
        rows.append(_one(rnd, be, lf, voice_of, "massing"))
    for lf in rest:
        voice_of(lf)
    n_workers = int(os.environ.get("ETHOSLM_DESIGN_WORKERS") or _cfg(rnd).get("workers") or 1)
    groups: dict = {}
    for lf in rest:
        groups.setdefault(lf.get("block") if lf.get("block") is not None
                          else lf["name"].rsplit("_", 1)[0], []).append(lf)
    chunks = _balance(list(groups.values()), max(1, n_workers * 2))
    if n_workers > 1 and len(chunks) > 1:
        from ..parallel import par_map
        global _CTX
        _CTX = (rnd, be.volume, voices, "massing")
        got = par_map(_chunk_worker, chunks, n=n_workers)
        _CTX = None
        for res in got:
            be._vol = be._vol.overlay(res["blocks"])
            rows.extend(res["rows"])
    else:
        for lf in rest:
            rows.append(_one(rnd, be, lf, voice_of, "massing"))
    for lf in last:
        rows.append(_one(rnd, be, lf, voice_of, "massing"))
    vol = be.volume
    rec = {"centre": [city.cx, city.cz], "radius": city.R, "monument": mon}

    def level(x, z):
        i, j = int(x) - X0, int(z) - Z0
        if 0 <= i < city.W and 0 <= j < city.W and city.target[i, j] > -30000:
            return int(city.target[i, j])
        return None
    cams = [c for c in composition_cameras(city.design, rec, level)
            if c["name"] != "city_overview"]
    views = render_views(vol, cams, out_dir, size=(720, 405), only=only)
    sheet = contact_sheet(views, os.path.join(out_dir, "contact.png"))
    failed = [r for r in rows if r.get("status") != "built"]
    out = {"views": views, "contact": sheet, "leaves": len(mine),
           "built": len(rows) - len(failed), "failed": [r.get("part") for r in failed][:20],
           "context": base or "the ground as found", "context_regions": len(used),
           "seconds": round(time.perf_counter() - t0, 1)}
    json.dump(out, open(os.path.join(out_dir, "massing.json"), "w"), indent=1, default=str)
    return out


# ---------------------------------------------------------------- delivery

#: blocks whose connection states only a running server computes (fences, walls, panes,
#: stairs): re-placed with block updates on after the bulk pass, as `Builder.flush` does
CONNECTIVE = ("fence", "wall", "pane", "bars", "stairs", "chest", "vine", "door")


def _inherited_written(rnd, snap, world_dir) -> dict:
    """A fresh `written.json`, or -- where the round delivers into a copy of a save
    another state wrote (`flags.design.inherits`, that state's directory) -- that
    state's record, each region pointing at the diff actually standing in the save."""
    cfg = _cfg(rnd)
    rec = {"regions": {}, "snapshot": os.path.relpath(snap, ROOT),
           "world": os.path.relpath(world_dir, ROOT)}
    src = cfg.get("inherits")
    if src:
        prev = _jload(os.path.join(ROOT, src, "written.json")) or {}
        for rid, r in (prev.get("regions") or {}).items():
            path = r.get("path") or os.path.join(src, "regions", rid, "diff.npz")
            rec["regions"][rid] = {**r, "path": path, "inherited_from": src}
        rec["inherits"] = src
    return rec


def _diff_cells(path, box=None) -> dict:
    """A diff as `{(x, y, z): block}`, only inside `box` `(x0, z0, x1, z1)` if given."""
    if not path or not os.path.exists(os.path.join(ROOT, path)):
        return {}
    xyz, codes, pal = load_diff(os.path.join(ROOT, path))
    if box is not None:
        m = (xyz[:, 0] >= box[0]) & (xyz[:, 0] <= box[2]) & (xyz[:, 2] >= box[1]) & \
            (xyz[:, 2] <= box[3])
        xyz, codes = xyz[m], codes[m]
    return {(int(x), int(y), int(z)): pal[int(c)] for (x, y, z), c in
            zip(xyz.tolist(), codes.tolist())}


def replacement(rnd, rid: str, old_path: str | None) -> tuple:
    """What writing region `rid`'s current diff over `old_path` (the diff standing in
    the save) must place: every cell either touches, at its value in the city as it
    will stand -- the ground as found, then every region's current diff in the order
    they are assembled. A cell only the old design changed goes back to the ground as
    found (or to a neighbour's work standing there); a smaller new design is never laid
    over leftover old buildings. Returns `([((x, y, z), block), ...], restored)`."""
    new = _diff_cells(os.path.relpath(rnd.rel("regions", rid, "diff.npz"), ROOT))
    old = _diff_cells(old_path)
    cells = set(new) | set(old)
    if not cells:
        return [], 0
    val = {}
    only_old = [c for c in cells if c not in new]
    if only_old:
        xs = [c[0] for c in only_old]
        ys = [c[1] for c in only_old]
        zs = [c[2] for c in only_old]
        x0, z0 = min(xs), min(zs)
        vol = base_volume(rnd, x0, z0, max(xs) - x0 + 1, max(zs) - z0 + 1,
                          min(ys), max(ys) + 1)
        for (x, y, z) in only_old:
            val[(x, y, z)] = str(vol.palette[int(vol.codes[x - vol.x0, y - vol.y0,
                                                           z - vol.z0])])
    # neighbours' current work standing in the cells the old design alone had changed
    bx0 = min(c[0] for c in cells)
    bx1 = max(c[0] for c in cells)
    bz0 = min(c[2] for c in cells)
    bz1 = max(c[2] for c in cells)
    base = rnd.rel("regions")
    for other in sorted(os.listdir(base)):
        rj = _jload(os.path.join(base, other, "region.json"))
        if not rj or not rj.get("ok"):
            continue
        b = rj.get("bounds")
        if b and (b[2] < bx0 or b[0] > bx1 or b[3] < bz0 or b[1] > bz1):
            continue
        if other == rid:
            src = new
        else:
            src = _diff_cells(os.path.relpath(os.path.join(base, other, "diff.npz"), ROOT),
                              (bx0, bz0, bx1, bz1))
        for c in cells:
            v = src.get(c)
            if v is not None:
                val[c] = v
    restored = sum(1 for c in only_old if c not in new)
    return sorted(val.items()), restored


def _write_delivery_record(path: str, record: dict) -> None:
    """Publish a complete receipt, including an unfinished write's touched footprint."""
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(record, f, indent=1)
    os.replace(tmp, path)


def _pending_delivery(rnd, rid: str, items: list) -> str:
    """Keep every cell a write might touch so retries can restore obsolete cells."""
    path = rnd.rel("regions", rid, "pending.diff.npz")
    palette = sorted({value for _, value in items})
    index = {value: i for i, value in enumerate(palette)}
    with open(path + ".tmp", "wb") as f:
        np.savez_compressed(f, xyz=np.array([p for p, _ in items], dtype=np.int32)
                            .reshape(-1, 3),
                            codes=np.array([index[v] for _, v in items], dtype=np.int32),
                            palette=np.array(palette, dtype=str))
    os.replace(path + ".tmp", path)
    return os.path.relpath(path, ROOT)


def stage_region_write(rnd, be, results: dict) -> dict:
    """Write the built regions into the **delivery save** (`flags.design.world`), never
    the shared one: a separate world on the same seed, served by its own server with the
    GDMC-HTTP interface. The save is snapshotted before the first write; every region's
    diff is written whole, and `written.json` records what went in, from which diff."""
    cfg = _cfg(rnd)
    world_dir = _world(rnd)
    rec_p = rnd.rel("written.json")
    previous = _jload(rec_p)
    if previous and (not previous.get("world") or os.path.realpath(
            os.path.join(ROOT, previous["world"])) != os.path.realpath(world_dir)):
        return {"status": "error", "stop": True,
                "error": "this run's delivery receipt belongs to another save; "
                         "use its original save or a separate run for another destination"}
    if os.path.realpath(world_dir) == os.path.realpath(os.path.join(ROOT, "run", "server",
                                                                    "world")):
        return {"status": "error", "stop": True,
                "error": "region_write delivers into a separate save. "
                         "the shared one"}
    from gdpc.interface import placeBlocks
    from gdpc import Block
    from .. import world as world_mod
    host = cfg.get("host", world_mod.HOST)
    try:
        import requests
        requests.get(host + "/version", timeout=5).raise_for_status()
    except Exception as e:                         # noqa: BLE001 -- named
        return {"status": "error", "stop": True,
                "error": f"no GDMC-HTTP server at {host} ({type(e).__name__}): start the "
                         f"delivery save's server with the interface enabled "
                         f"(`scripts/server.sh start` with ETHOSLM_SERVER_DIR naming its "
                         f"server directory)"}
    snap = os.path.join(ROOT, "run", "snapshots", f"{rnd.name}.before-write")
    if not os.path.exists(snap):
        shutil.copytree(world_dir, snap, ignore=shutil.ignore_patterns("session.lock"))
    written = previous or _inherited_written(rnd, snap, world_dir)
    want = cfg.get("write") or sorted(os.listdir(rnd.rel("regions")))
    out = {"written": [], "kept": [], "restored": 0}
    missing = [rid for rid in want if not _jload(rnd.rel("regions", rid, "region.json"))
               or not os.path.exists(rnd.rel("regions", rid, "diff.npz"))]
    if missing or not want:
        return {"status": "error", "stop": True,
                "error": f"no complete built region set to deliver; missing: {missing}"}
    written["complete"] = False
    _write_delivery_record(rec_p, written)
    for rid in want:
        dp = rnd.rel("regions", rid, "diff.npz")
        rj = _jload(rnd.rel("regions", rid, "region.json"))
        if not rj or not os.path.exists(dp):
            continue
        dg = hashlib.sha256(open(dp, "rb").read()).hexdigest()[:16]
        was = written["regions"].get(rid) or {}
        if was.get("diff") == dg and not was.get("failed") and \
                was.get("status", "complete") == "complete":
            out["kept"].append(rid)
            continue
        t0 = time.perf_counter()
        items, restored = replacement(rnd, rid, was.get("path"))
        # Keep the union of old and new cells even if a write fails or the next design
        # changes. It is the footprint a retry must repair, not a claim all cells stand.
        attempt = {"diff": dg, "path": _pending_delivery(rnd, rid, items),
                   "status": "pending", "cells": len(items), "replaced": was.get("diff")}
        written["regions"][rid] = attempt
        _write_delivery_record(rec_p, written)
        blocks = {}
        ok = bad = 0
        pairs = []
        for (x, y, z), name in items:
            b = blocks.get(name)
            if b is None:
                b = blocks[name] = Block(name if ":" in name.split("[")[0]
                                         else "minecraft:" + name)
            pairs.append(((x, y, z), b))
        try:
            for i in range(0, len(pairs), 50_000):
                batch = pairs[i:i + 50_000]
                response = list(placeBlocks(batch, doBlockUpdates=False, host=host))
                good = sum(bool(row[0]) for row in response)
                ok += good
                bad += max(0, len(batch) - good)
            conn = [it for it in pairs if any(k in str(it[1].id) for k in CONNECTIVE)]
            for i in range(0, len(conn), 20_000):
                batch = conn[i:i + 20_000]
                response = list(placeBlocks(batch, doBlockUpdates=True, host=host))
                bad += max(0, len(batch) - sum(bool(row[0]) for row in response))
        except Exception as e:                     # noqa: BLE001 -- receipt keeps retry state
            attempt["error"] = f"{type(e).__name__}: {e}"
            bad += 1
        if bad:
            attempt.update(placed=ok, failed=bad)
            _write_delivery_record(rec_p, written)
            return {**out, "status": "error", "stop": True, "failed": [rid],
                    "error": f"region {rid}: {bad} writes unconfirmed; retry delivery",
                    "record": os.path.relpath(rec_p, ROOT)}
        # the diff now standing in the save, kept beside the region: the next
        # replacement restores what this one wrote and the next does not
        keep = rnd.rel("regions", rid, "written.diff.npz")
        shutil.copyfile(dp, keep)
        written["regions"][rid] = {"diff": dg, "path": os.path.relpath(keep, ROOT),
                                   "status": "complete",
                                   "cells": len(items), "restored": restored,
                                   "placed": ok, "failed": bad,
                                   "replaced": was.get("diff"),
                                   "seconds": round(time.perf_counter() - t0, 1),
                                   "t": time.strftime("%Y-%m-%dT%H:%M:%S")}
        _write_delivery_record(rec_p, written)
        out["written"].append(rid)
        out["restored"] += restored
        print(f"   wrote {rid}: {ok:,} placed ({restored:,} restored to the ground as "
              f"found), {bad} failed", flush=True)
    try:
        from gdpc.interface import runCommand
        flushed = list(runCommand("save-all flush", host=host))
        if not flushed or not all(row[0] for row in flushed):
            raise RuntimeError("server did not confirm save-all flush")
    except Exception as e:                         # noqa: BLE001 -- retry without rewriting
        return {**out, "status": "error", "stop": True,
                "error": f"blocks sent, but save flush is unconfirmed: {e}; retry delivery",
                "record": os.path.relpath(rec_p, ROOT)}
    written["complete"] = True
    _write_delivery_record(rec_p, written)
    return {**out, "record": os.path.relpath(rec_p, ROOT)}
