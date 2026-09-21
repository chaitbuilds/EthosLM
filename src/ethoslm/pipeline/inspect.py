"""Look at the assembled built world, and measure what a reader would point at.

The closure round. Two things were missing between "the parts stood" and "the place is
the place": a picture of the **built** world a judge could look at -- the preview draws
the plan and five type cards, and the review said plainly that a plan map and isolated
building images cannot establish composition or access -- and a **measurement** of the
composition that a reading's findings can be verified against, so that "the village is
in two pieces" is a number before and after a revision and not a sentence twice.

    measure(plan, site)      deterministic composition measures off the plan's leaves
    draw_built(...)          the cheapest renderer that shows the built geometry: the
                             isometric of the whole site, a top-down of the built
                             volume, and street/approach elevations about the centre
    stage_inspect            after the construction check and before qualification:
                             draw, then ask the judge for a reading of what was built,
                             bound to the candidate it looked at

No server, no model call for the drawing, and every picture is a pure function of the
built volume.
"""
from __future__ import annotations

import json
import os
import time

from .. import pipeline as _pipeline

#: The measures a reading may name, each with the direction "better" is. Registered
#: here, before any reading cited one.
MEASURES = {
    "clusters": ("down", "how many separate groups the buildings stand in, a lane apart "
                         "or less counting as one group"),
    "cluster_gap": ("down", "the largest gap in columns between two groups"),
    "square_scale": ("down", "the largest open area against the median building "
                             "footprint, as a ratio"),
    "open_to_built": ("down", "open-ground leaves against building leaves, as a ratio"),
    "undeveloped_share": ("down", "the share of district ground nobody owns"),
    "plots": ("up", "how many buildings there are"),
}

#: Two plots closer than this, edge to edge, stand in one group. A lane and its verges.
CLUSTER_GAP = 10


def _rects(parts: list, kind: str) -> list:
    out = []
    for p in parts:
        if p.get("kind", "plot") != kind or p.get("x1") is None:
            continue
        r = _pipeline.part_rect(p)
        out.append((p.get("name"), [int(v) for v in r]))
    return out


def _gap(a, b) -> int:
    dx = max(0, max(a[0], b[0]) - min(a[2], b[2]))
    dz = max(0, max(a[1], b[1]) - min(a[3], b[3]))
    return max(dx, dz)


def measure(plan: dict | None, site: dict | None = None,
            arrangements: dict | None = None) -> dict:
    """The composition measures, off the plan that exists. Deterministic."""
    parts = _pipeline.plan_parts(plan or {})
    plots = _rects(parts, "plot")
    areas = _rects(parts, "area")
    out: dict = {"plots": len(plots), "areas": len(areas)}
    # clusters: union-find over plots within CLUSTER_GAP of each other
    parent = list(range(len(plots)))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i
    for i in range(len(plots)):
        for j in range(i + 1, len(plots)):
            if _gap(plots[i][1], plots[j][1]) <= CLUSTER_GAP:
                a, b = find(i), find(j)
                if a != b:
                    parent[a] = b
    groups: dict = {}
    for i in range(len(plots)):
        groups.setdefault(find(i), []).append(i)
    out["clusters"] = len(groups)
    gap = 0
    keys = list(groups)
    for i in range(len(keys)):
        for j in range(i + 1, len(keys)):
            g = min(_gap(plots[a][1], plots[b][1])
                    for a in groups[keys[i]] for b in groups[keys[j]])
            gap = max(gap, g)
    out["cluster_gap"] = int(gap)
    areas_cols = [(r[2] - r[0] + 1) * (r[3] - r[1] + 1) for _n, r in areas]
    plot_cols = sorted((r[2] - r[0] + 1) * (r[3] - r[1] + 1) for _n, r in plots)
    median = plot_cols[len(plot_cols) // 2] if plot_cols else 0
    out["square_scale"] = (round(max(areas_cols) / float(median), 2)
                           if areas_cols and median else None)
    out["open_to_built"] = (round(sum(areas_cols) / float(sum(plot_cols)), 3)
                            if plot_cols else None)
    rows = (arrangements or {}).get("districts") or []
    und = [float(r.get("undeveloped_share") or 0) for r in rows
           if r.get("undeveloped_share") is not None]
    out["undeveloped_share"] = round(sum(und) / len(und), 3) if und else None
    return out


def compare(before: dict | None, after: dict | None, cited=()) -> dict:
    """Which cited measures moved the way "better" is, and which moved the other way."""
    better, worse, same = [], [], []
    for m in cited or []:
        if m not in MEASURES or not before or not after:
            continue
        a, b = before.get(m), after.get(m)
        if a is None or b is None:
            continue
        want = MEASURES[m][0]
        if a == b:
            same.append(m)
        elif (b < a) == (want == "down"):
            better.append(m)
        else:
            worse.append(m)
    return {"better": better, "worse": worse, "same": same}


# ------------------------------------------------------------------- the pictures

INSPECT_BRIEF = """# Read the built place

> {sentence}

{intent}

This place has been **built**: every leaf of the plan was constructed offline, the
construction check ran over what stands, and these pictures are drawn from the built
volume itself, not from the plan.

{views}

## What was measured

{measures}

## What is asked

Write a page, as a person walking this place: **what is wrong that no check
reports**, and what is right. The whole first -- does it read as one place gathered
the way the sentence says, or as pieces; is the open ground somebody's; do the streets
and approaches lead anywhere -- then the buildings and the voice from the street views.

Then write a second file, `{findings}`, with the same findings as data. **Every finding
names its subjects and says whether the place is finished without it.** A finding is
`material` when the request, its architecture or its usability is not met without a
change -- an oversized anchor, working land too thin for the place's purpose, a lost
required feature, no frontage where the request implies one -- and `optional` when it
is a suggestion the place is finished without. A material finding carries either a
contextual `target` (a measure from the list below with the direction and the value the
programme implies for THIS place -- not a universal direction: a farming village may
need MORE open land, a market square sized to its houses, a dense ring its share) or a
qualitative `acceptance` condition a later reading can decide. `owner` is who can act:
`layout` (anchors, rings, land regions, lots), `fabric` (a district's character:
block, lots, open share, storeys), `voice`, `build` (a type), `capability`. `action`
names the bounded action, and **the list below is the one the improvement stage can
actually execute** -- it is read from `pipeline.improve.owner_actions`, so an action
offered here is an action that exists:

{actions}

...or an object `{{"type": "character", "characters": {{district: character}}}}` /
`{{"type": "voice", "voice": name}}`.

```json
{{"findings": [{{"id": "b1", "about": "composition|fabric|access|voice|scale|function",
                "says": "one sentence", "subjects": ["part or region names"],
                "material": true, "measure": "one of {measure_names} or null",
                "owner": "layout|fabric|voice|scale|build|capability",
                "target": {{"measure": "square_scale", "direction": "down", "value": 4.0,
                           "from": ["what the programme implies it from"]}},
                "acceptance": "a condition a later reading can decide, or null",
                "action": "shrink_anchor"}}],
  "closed": ["ids of the findings of the earlier reading below that this built place no longer shows"],
  "right": ["what is right, briefly"]}}
```

{earlier}
"""


def _above(vol, keep_below: int = 3):
    """The volume with everything more than `keep_below` blocks under its own surface
    made air, so an elevation is a street and not a cliff of subsoil."""
    import numpy as np
    from .. import observe, offline
    surface = offline.surface_heights(vol)
    floor = int(np.median(surface)) - keep_below
    pal = list(vol.palette)
    air = next((i for i, n in enumerate(pal) if str(n).split("[")[0].endswith("air")), None)
    if air is None:
        pal.append("minecraft:air")
        air = len(pal) - 1
    codes = vol.codes.copy()
    cut = max(0, floor - vol.y0)
    codes[:, :cut, :] = air
    return observe.Volume(vol.x0, vol.y0, vol.z0, codes, pal)


def _draw_built(rnd, built, plan: dict, site: dict | None) -> dict:
    import cv2
    from .. import preview as preview_mod
    d = rnd.rel("inspection")
    os.makedirs(d, exist_ok=True)
    out: dict = {"images": {}, "seconds": {}}
    t0 = time.perf_counter()
    parts = _pipeline.plan_parts(plan)
    rects = [r for _n, r in _rects(parts, "plot") + _rects(parts, "area")
             + _rects(parts, "edge")]
    if not rects:
        return out
    x0 = min(r[0] for r in rects) - 12
    z0 = min(r[1] for r in rects) - 12
    x1 = max(r[2] for r in rects) + 12
    z1 = max(r[3] for r in rects) + 12
    whole = _above(preview_mod.crop(built, x0, z0, x1, z1, pad=0), keep_below=6)
    img = preview_mod.preview(whole, scale=2)
    p = os.path.join(d, "overall.png")
    cv2.imwrite(p, img[:, :, ::-1])
    out["images"]["overall"] = p
    out["seconds"]["overall"] = round(time.perf_counter() - t0, 2)
    t1 = time.perf_counter()
    top = preview_mod.top_down(whole, scale=3)
    p = os.path.join(d, "top.png")
    cv2.imwrite(p, top[:, :, ::-1])
    out["images"]["top"] = p
    out["seconds"]["top"] = round(time.perf_counter() - t1, 2)
    # street views: about the largest open area (the square, the court), from each side
    areas = _rects(parts, "area")
    t2 = time.perf_counter()
    if areas:
        _n, sq = max(areas, key=lambda nr: (nr[1][2] - nr[1][0]) * (nr[1][3] - nr[1][1]))
        cx, cz = (sq[0] + sq[2]) // 2, (sq[1] + sq[3]) // 2
        reach = 44
        for facing, rect in (("south", (cx - reach, sq[1] - 4, cx + reach, cz)),
                             ("north", (cx - reach, cz, cx + reach, sq[3] + 4)),
                             ("east", (cx, cz - reach, sq[2] + 4, cz + reach)),
                             ("west", (sq[0] - 4, cz - reach, cx, cz + reach))):
            sub = _above(preview_mod.crop(built, *rect, pad=2), keep_below=2)
            img = preview_mod.elevation(sub, facing=facing, scale=3)
            p = os.path.join(d, f"street_{facing}.png")
            cv2.imwrite(p, img[:, :, ::-1])
            out["images"][f"street_{facing}"] = p
        out["street_about"] = {"area": _n, "rect": sq}
    out["seconds"]["streets"] = round(time.perf_counter() - t2, 2)
    # the approach: the elevation of the whole from the side the arterial enters
    lay = (plan.get("layout") or {})
    side = str(lay.get("axis_side") or "south")
    if side not in ("north", "south", "east", "west"):
        side = "south"
    t3 = time.perf_counter()
    img = preview_mod.elevation(_above(whole, keep_below=2), facing=side, scale=2)
    p = os.path.join(d, f"approach_{side}.png")
    cv2.imwrite(p, img[:, :, ::-1])
    out["images"]["approach"] = p
    out["seconds"]["approach"] = round(time.perf_counter() - t3, 2)
    out["seconds"]["all"] = round(time.perf_counter() - t0, 2)
    out["bounds"] = [x0, z0, x1, z1]
    return out


def stage_inspect(rnd, be, results: dict) -> dict:
    """Draw the built world and ask the judge to read it, bound to this candidate."""
    from .. import contracts, deps
    from .stages_measure import built_volumes
    plan = rnd.plan()
    if not plan:
        return {"skipped": "no plan.json: nothing was built to inspect"}
    built, _base = built_volumes(rnd)
    if built is None:
        return {"status": "error", "stop": True,
                "error": "no world_built.npz: nothing has been built, so there is no "
                         "assembled world to inspect"}
    d = rnd.rel("inspection")
    os.makedirs(d, exist_ok=True)
    here = deps.candidate_id(rnd, plan=plan)
    rec_p = os.path.join(d, "views.json")
    rec = json.load(open(rec_p)) if os.path.exists(rec_p) else {}
    # **Of this candidate and of this built volume.** A rebuilt world under the same
    # plan is a different thing to look at, so the inspection is keyed on both.
    built_digest = deps.content_print(rnd.rel("world_built.npz"))
    if rec.get("built_digest") and rec["built_digest"] != built_digest:
        rec["candidate"] = f"{rec.get('candidate')}@{rec['built_digest'][:8]}"
    if rec.get("candidate") and rec["candidate"] != here:
        # the candidate moved under this inspection, in any state: everything drawn and
        # read of it is about a design that is gone
        for f in sorted(os.listdir(d)):
            if not f.startswith("stale."):
                os.replace(os.path.join(d, f), os.path.join(d, f"stale.{rec['candidate']}.{f}"))
        print(f"   inspect: the inspection on disk is of candidate {rec['candidate']} and "
              f"this is {here}; it is set aside and the built world is drawn and read "
              f"again", flush=True)
        rec = {}
    site = _pipeline.settlement_site(rnd)
    if not rec.get("images"):
        drawn = _draw_built(rnd, built, plan, site)
        arr_p = rnd.rel("arrangements.json")
        rec = {"candidate": here, "built_digest": built_digest, **drawn,
               "overall": drawn["images"].get("overall"),
               "street": [v for k, v in drawn["images"].items() if k.startswith("street_")],
               "approach": drawn["images"].get("approach"),
               "measures": {**measure(plan, site, json.load(open(arr_p))
                                       if os.path.exists(arr_p) else None),
                            **section_measures(rnd)},
               "t": time.strftime("%Y-%m-%dT%H:%M:%S")}
        json.dump(rec, open(rec_p, "w"), indent=1)
        print(f"   inspect: {len(rec['images'])} view(s) of the built world drawn in "
              f"{drawn['seconds'].get('all')}s; measures {rec['measures']}", flush=True)
    reading_p = os.path.join(d, "reading.md")
    findings_p = os.path.join(d, "findings.json")
    if not os.path.exists(reading_p):
        brief_p = os.path.join(d, "reading_prompt.md")
        it = contracts.load(rnd, "intent") or {}
        views = "\n".join(f"{i + 1}. `{os.path.basename(p)}` -- {k.replace('_', ' ')}"
                          for i, (k, p) in enumerate(rec["images"].items()))
        earlier = ""
        pv = rnd.rel("preview", "preview.json")
        prev = json.load(open(pv)) if os.path.exists(pv) else {}
        rows = ((prev.get("applied") or {}).get("caused_by_findings")
                or (prev.get("reading_findings") or {}).get("findings") or [])
        if rows:
            earlier = ("## The earlier reading's findings\n\n"
                       + "\n".join(f"- `{f.get('id')}` {f.get('says')}" for f in rows))
        from .improve import owner_actions
        acts = "\n".join(
            f"- `{o}`: " + ", ".join(f"`{a}`" for a in owner_actions(o))
            for o in ("layout", "fabric", "voice", "build", "scale")
            if owner_actions(o))
        names = dict(MEASURES)
        names.update({k: ("", d) for k, d in SECTION_MEASURES.items()})
        open(brief_p, "w").write(INSPECT_BRIEF.format(
            sentence=rnd.sentence, intent=(plan.get("intent") or ""),
            views=views, findings=findings_p, actions=acts,
            measures="\n".join(f"- `{k}`: {v} -- {names.get(k, ('', ''))[1]}"
                               for k, v in (rec.get("measures") or {}).items()),
            measure_names=", ".join(f"`{m}`" for m in
                                    list(MEASURES) + sorted(
                                        k for k in (rec.get("measures") or {})
                                        if k.startswith("section."))),
            earlier=earlier))
        json.dump({"candidate": here, "request": brief_p, "write": reading_p,
                   "findings": findings_p}, open(os.path.join(d, "reading.job.json"), "w"),
                  indent=1)
        return {"inspection": {
            "status": "needs_model", "role": "judge", "request": brief_p,
            "write": reading_p, "candidate": here,
            "images": list(rec["images"].values())[:8],
            "note": "the reading of the built world: what a person walking it would see "
                    "wrong that no check reports, and the same as data"}}
    found = json.load(open(findings_p)) if os.path.exists(findings_p) else {"findings": [],
                                                                            "closed": []}
    rec["reading"] = {"path": reading_p, "findings": found.get("findings") or [],
                      "closed": found.get("closed") or [], "right": found.get("right") or [],
                      "structured": os.path.exists(findings_p)}
    json.dump(rec, open(rec_p, "w"), indent=1)
    with __import__("contextlib").suppress(ValueError):
        deps.stamp(rnd, "inspection", outputs=["inspection/views.json"], plan=plan,
                   note=f"{len(rec['reading']['findings'])} finding(s)")
    return {"candidate": here, "views": len(rec["images"]),
            "findings": len(rec["reading"]["findings"]),
            "closed": rec["reading"]["closed"], "measures": rec.get("measures"),
            "structured": rec["reading"]["structured"], "written": rec_p}

def stage_section(rnd, be, results: dict) -> dict:
    """Write what the registered section actually demonstrates, measured.

        The composition round, and it exists because the design round's `section.json` was
        **hand-authored**: nothing in the repository wrote it, and the acceptance gate that
        read it compared the length of its `demonstrated` list with the length of the round
        file's `must_demonstrate` list. Five of five passed while the round's own report said
        the contrasting fabrics could not be seen. A claim a stage did not measure is not
        evidence, so the record is derived here, from this build's own artifacts, every time
        the section is rebuilt -- which is also why this stage is in the improve stage's
        rebuild list and not only in the main order.

        A round with no `flags.section` skips: there is no registered section to measure.
        
    """
    from .. import section as section_mod
    reg = (rnd.flags.get("section") or {})
    if not reg:
        return {"skipped": "this round registers no section"}
    if not os.path.exists(rnd.rel("parts.json")):
        return {"skipped": "nothing is built yet: no parts.json to measure a section on"}
    rec = section_mod.record(rnd.state, registered=reg, of=rnd.name)
    p = rnd.rel("section.json")
    json.dump(rec, open(p, "w"), indent=1)
    print(f"   section {rec['registered'].get('id')}: "
          f"demonstrated {rec['demonstrated']}, failed {rec['failed']}, "
          f"unmeasured {rec['unmeasured']} "
          f"({rec['reported_not_targeted']['parts']} part(s) standing)", flush=True)
    for r in rec["relationships"]:
        print(f"      {r['status']:13s} {r['id']}: {str(r.get('how'))[:150]}", flush=True)
    return {"written": p, "demonstrated": rec["demonstrated"],
            "failed": rec["failed"], "unmeasured": rec["unmeasured"],
            "parts": rec["reported_not_targeted"]["parts"]}

#: The section's own measures, and what "better" is for each. A finding about the
#: section's fabric has to be able to cite a number that is **about the section** and
#: that an action can move: the six composition measures above are whole-place numbers,
#: so a row citing one of them closed on the whole place's reading and a finding about
#: one district's fabric had nothing of its own to name. These are added to the measures
#: dict wherever a round registers a section.
SECTION_MEASURES = {
    "section.contrast.held": "how many of the three built contrast measures differ by "
                             "the registered ratio (up is better)",
    "section.built_cover": "what share of the section's district ground carries emitted "
                           "building mass",
    "section.median_footprint": "the median built footprint of the section's buildings, "
                                "in columns",
    "section.structures": "how many complete buildings stand in the section",
}


def section_measures(rnd) -> dict:
    """The registered section's own measures, flattened for a finding to cite.

        `{"section.<side>.built_cover": ..., "section.contrast.held": n, ...}` where the
        round registers a section and something is built; `{}` otherwise. A measure a
        finding may name has to exist in this dict, because `obligation.close` looks the
        row's own measure up in it -- which is why an `emitted.<feature>` row could never
        close and a per-district finding closed on a whole-place number.
        
    """
    from .. import section as section_mod
    reg = rnd.flags.get("section") or {}
    if not reg or not os.path.exists(rnd.rel("parts.json")):
        return {}
    try:
        rec = section_mod.record(rnd.state, registered=reg, of=rnd.name)
    except Exception:                       # noqa: BLE001 -- a measure, not a stage
        return {}
    out: dict = {}
    con = next((r for r in rec["relationships"] if r["id"] == "contrast"), {})
    m = con.get("measured") or {}
    out["section.contrast.held"] = len(m.get("held") or [])
    tot_built = tot_ground = 0
    foots, structures = [], 0
    for s in m.get("sides") or []:
        side = str(s.get("side"))
        for key in ("built_cover", "median_footprint", "median_neighbour_gap",
                    "court_share", "structures"):
            if s.get(key) is not None:
                out[f"section.{side}.{key}"] = s[key]
        tot_built += int(s.get("built_columns") or 0)
        tot_ground += int(s.get("ground_columns") or 0)
        structures += int(s.get("structures") or 0)
        if s.get("median_footprint"):
            foots.append(float(s["median_footprint"]))
    if tot_ground:
        out["section.built_cover"] = round(tot_built / tot_ground, 4)
    if foots:
        out["section.median_footprint"] = round(sum(foots) / len(foots), 1)
    out["section.structures"] = structures
    return out

def stage_material(rnd, be, results: dict) -> dict:
    """The material pass, applied to a finished volume beside the built one.

        Off unless the round asks (`flags.material`). The composition round's bounded visual
        experiment judged the restrained recipe now on disk `enable_it, bounded`, on the two
        voices this city builds in: the positive control moves the textured frame 155 of 255
        on `wall` and 129 on `footing` where the flat display moves 0 and 6; the contextual
        arm makes 4,785 substitutions against the design round's 25,211, coheres at 0.172
        against a random arm's 0.027, lands on its condition 1.00 against 0.154, and leaves
        the courts and the market floors **byte-identical** because the types that draw them
        now declare their figures and the pass refuses a declared figure.

        What this writes is `world_finished.npz`. `world_built.npz` stays the structural
        record every check is taken against -- the construction check, the predicates, the
        occupancy, the section's own measurements -- so enabling this cannot make a physical
        check pass. A view may read the finished volume; nothing else does.
        
    """
    if not rnd.flags.get("material"):
        return {"skipped": "this round does not ask for the material pass"}
    from .. import material as material_mod, offline as offline_mod, surfaces as surf
    sp, bp = rnd.rel("surfaces.json"), rnd.rel("world_built.npz")
    if not (os.path.exists(sp) and os.path.exists(bp)):
        return {"skipped": "no surface record or no built volume"}
    doc = surf.read(sp)
    voices = sorted({str(r.get("voice_name") or "") for r in doc.get("parts") or []
                     if r.get("voice_name")})
    recipes = {}
    for v in voices:
        try:
            recipes[v] = material_mod.recipe_for(v)
        except Exception as e:                    # noqa: BLE001 -- reported per voice
            print(f"   material: {v} has no usable recipe ({type(e).__name__}: {e})",
                  flush=True)
    if not recipes:
        return {"skipped": f"none of {voices or 'the record\'s voices'} carries a recipe"}
    built = offline_mod.load_volume(bp)
    setting = {"condition": str(rnd.flags.get("material_condition") or "maintained"),
               "weather_from": material_mod.WEATHER_FROM}
    fin, rec = material_mod.apply(built, doc, {"by_voice": recipes}, setting,
                                  seed=int(rnd.flags.get("seed") or 1),
                                  mode="contextual")
    out_p = rnd.rel("world_finished.npz")
    offline_mod.save_volume(fin, out_p)
    print(f"   material: {rec.get('substituted')} substitution(s) over "
          f"{len(recipes)} voice(s), {rec.get('figure_refused')} declared figure "
          f"cell(s) refused, {rec.get('stale_skipped')} stale cell(s) left alone "
          f"-> {os.path.basename(out_p)}", flush=True)
    return {"written": out_p, "voices": voices,
            "substituted": rec.get("substituted"),
            "figure_refused": rec.get("figure_refused"),
            "stale_skipped": rec.get("stale_skipped"),
            "condition": setting["condition"]}
