"""The second whole-place layout policy: a settlement laid out along a shoreline.

It does not prove that *anything else* does, and the architecture audit's third finding
is exactly that: there was one whole-place layout with its own arithmetic, and
everywhere else the relation solver cut four strips round a centre. A place asked to
follow the water got a square town beside some.

So there are two policies, and they are deliberately built on **one** mechanism:
`ShoreSolver` is `placesolve.Solver` with its district step replaced. Every leaf part is
still placed by its relation, every candidate still goes through the same vetoes, the
document is still the one `place_failures` validates and `district_compile` lays out. A
switch between two independently written end-to-end pipelines would prove nothing about
generality; sharing the seam is the point.

What the shoreline policy actually adds:

  - the **anchor is derived from the ground**. `regions.shore_anchor` walks the land
    columns that touch water and returns the path they make. There is no fixed line and
    no fallback rectangle: a site with no water cannot carry a shoreline place and the
    policy says so by name rather than laying one out anyway.
  - **no obligatory centre.** `spec.core` falls back to the first defining part when
    nothing declares `centre`, and the relation solver cuts its strips round whatever
    that is -- a hidden centre in a layout that has none. Here the core is None unless
    the spec actually names one.
  - the districts are a **band following the path**, tiled against its mask
    (`regions.tile`), so a bay is a bay.
  - the **frontage faces the water**: the shore road runs between the band and the
    water, and every district is tiled longer along the shore than across it, so the
    compiler's own rule -- streets run along a district's longer side -- puts the lots'
    fronts on a street parallel to the water. That is a geometric fact the resolution
    record asserts, not a word in a brief.
"""
from __future__ import annotations

import math

from . import pipeline, placeregion as regions, placeplan, placesolve, spec as spec_mod

#: How wide, in columns, the road along the water is. A lane wide enough for the
#: circulation pass to make it a street of the district grid.
SHORE_ROAD = 5

#: How far inland the shore road runs from the water's own line, so the band's first row
#: of lots fronts it rather than standing in the water.
SHORE_SETBACK = 4

#: The least a shoreline district is on a side. The same floor the ring layout uses.
SHORE_MIN = 24

#: **Registered.** How much deeper a row of shore districts is made per unit of the
#: shore's own slope across the grid, capped here. A shore running along an axis gets
#: one district's depth; one at 45 degrees gets two, because a 28-deep band about a
#: 45-degree line holds no 28x28 axis-aligned rectangle and a district is a rectangle.
SHORE_SLOPE_MAX = 2.0

#: **Registered before the shoreline case ran.** How deep the band of land behind the
#: water may be negotiated to, as a share of the site, when the settlement the sentence
#: asks for does not fit in the band `regions.SHORE_DEPTH` gives. This is the **bound on
#: an inferred choice**: how deep a shore settlement runs is the library's inference and
#: is revisable inside a recorded bound; how many houses an explicit sentence asked for
#: is not. Past this the place is short and says so by name -- a cap is not a capacity.
SHORE_DEPTH_MAX = 0.62


def wants_frontage(spec: dict) -> str | None:
    """What the **sentence** says the buildings face, or None. `water` or `street`.

        Read through `ethoslm.intent`, off the sentence, like every other word-driven choice
        in this project. This is what turns "with homes facing the water" into a constraint
        the compiler obeys, instead of a sentence the report quotes.
        
    """
    from . import intent as intent_mod
    text = " ".join([str((spec or {}).get("sentence") or ""),
                     str((spec or {}).get("invariants") or "")])
    for r in intent_mod.read(text)["requirements"]:
        if r["kind"] == "orientation" and r["status"] != "unsupported":
            return str(r["wants"].get("faces"))
    return None


def wants_shoreline(spec: dict) -> bool:
    """Does this request ask for a place laid out along the water?

        Read off the **sentence**, through `ethoslm.intent`, and never off a place's name: the
        same rule every other word-driven choice in this project obeys. A spec that declares
        rings is concentric whatever else the sentence says -- rings are the stronger
        statement and the two cannot both be the whole-place organisation.
        
    """
    from . import intent as intent_mod
    if spec_mod.rings(spec):
        return False
    text = " ".join([str(spec.get("sentence") or ""),
                     str(spec.get("invariants") or "")])
    rec = intent_mod.read(text)
    return any(r["id"] == "layout/shoreline" and r["status"] != "unsupported"
               for r in rec["requirements"])


def _path_near(anchor: dict, rect, pad: int = 128) -> dict:
    """The shore anchor, cut down to the path within `pad` columns of `rect`.

        A district carries this so the compiler can ask which way the water lies from each
        of its blocks. Cut down rather than whole because the whole path of a site's shore
        is a long list to copy onto every district, and a point 300 columns away has no
        bearing on which wall a door goes in.
        
    """
    x0, z0, x1, z1 = [int(v) for v in rect]
    path = [[int(a), int(b)] for a, b in
            ((p[0], p[1]) for p in (anchor or {}).get("path") or [])
            if x0 - pad <= a <= x1 + pad and z0 - pad <= b <= z1 + pad]
    return {k: v for k, v in (anchor or {}).items() if k != "path"} | \
        {"path": path or [[int(p[0]), int(p[1])] for p in
                          (anchor or {}).get("path") or []]}


def _slope_of(anchor: dict) -> float:
    """The shore's median slope across the grid: how far it moves off its own axis per
    column along it. Zero for a shore running north-south or east-west."""
    path = [(p[0], p[1]) for p in (anchor or {}).get("path") or []]
    if len(path) < 2:
        return 0.0
    along_x = anchor.get("along") == "x"
    ds = []
    for a, b in zip(path, path[1:]):
        run = abs(b[0] - a[0]) if along_x else abs(b[1] - a[1])
        rise = abs(b[1] - a[1]) if along_x else abs(b[0] - a[0])
        if run:
            ds.append(rise / float(run))
    if not ds:
        return 0.0
    ds.sort()
    return float(ds[len(ds) // 2])


class ShoreSolver(placesolve.Solver):
    """`placesolve.Solver` with the shoreline's district step and no hidden centre."""

    def __init__(self, spec, site, plateau, decls, voice, vol=None, seed: int = 1,
                 caps: dict | None = None, intent: dict | None = None):
        super().__init__(spec, site, plateau, decls, voice, vol=vol, seed=seed,
                         caps=caps, intent=intent)
        # **No obligatory centre.** `spec.core` answers with the first defining part
        # where nothing declares `centre`, which is a reasonable default for a place
        # built round something and a fiction for one built along something.
        if not any(p["relation"] == "centre" for p in spec["defining_parts"]):
            self.core = None
        self.anchor = None
        self.wet = self.heights = None
        self.band = None
        if vol is not None:
            try:
                self.wet, self.heights = regions.water_mask(vol, self.X, self.Z, self.S)
                self.anchor = regions.shore_anchor(self.wet, self.X, self.Z)
            except Exception as e:             # noqa: BLE001 -- reported as a failure
                self.fails.append({"part": "place", "check": "shoreline",
                                   "why": f"the site's ground could not be read for a "
                                          f"shoreline: {e}"})

    # --- the whole ------------------------------------------------------------

    def solve(self) -> tuple:
        if self.wet is None:
            return None, [{"part": "place", "check": "shoreline",
                           "why": "a shoreline place is laid out against the water in "
                                  "the ground and this round has no cached volume to "
                                  "read it from"}]
        if self.anchor is None:
            share = float(self.wet.mean())
            return None, [{"part": "place", "check": "shoreline",
                           "why": f"no shoreline can be derived from this site: "
                                  f"{share:.1%} of its columns are water, against the "
                                  f"{regions.SHORE_MIN_WATER:.0%} a shore needs on "
                                  f"either side of the line. The request is refused "
                                  f"rather than laid out along a line nothing chose",
                           "water_share": round(share, 4)}]
        # **The band is as deep as a whole number of district rows.** Found by running
        # it: a band of 65 columns against a 28-column floor and a 6-column lane tiled
        # into one row of 32 and a 27-column remainder that is nobody's -- which is the
        # same defect the ring layout had at its chamfered corners, a size smaller. The
        # depth is snapped down to `rows x least + gaps` so the band divides exactly.
        least = max(SHORE_MIN, self.dmin)
        gap = placesolve.LANE_GAP
        self.least = least
        # **A row is deeper where the shore runs across the grid.** Found by running the
        # policy on the square the search chose: that shore runs diagonally, and a band
        # 28 columns deep about a 45-degree line contains no 28x28 axis-aligned
        # rectangle at all -- so a site with a perfectly good shore produced no district
        # and the run stopped. A district is a rectangle because the compiler lays out
        # rectangles; what has to give is the **depth of the row**, which is an inferred
        # number. The shore's own median slope says how much: a shore along an axis
        # needs one district's depth, a shore at 45 degrees needs two, and the districts
        # then step along it like houses on a curving bank.
        self.slope = _slope_of(self.anchor)
        self.row_depth = int(round(least * (1.0 + min(self.slope, SHORE_SLOPE_MAX))))
        self.rows_max = max(1, int((self.S * SHORE_DEPTH_MAX + gap)
                                   // (self.row_depth + gap)))
        rows = max(1, (max(self.row_depth, int(self.S * regions.SHORE_DEPTH)) + gap)
                   // (self.row_depth + gap))
        # **Two rows at least where the districts gather round something in the band**
        # (the closure round): a ribbon one row deep flanks a square on two sides and
        # never reaches behind it, and "around" is at least three sides. ...**and as
        # deep as the bound admits**: a diagonal shore's ribbons fit where the staircase
        # leaves room, and the square stands among them. The band's depth is an inferred
        # choice the district step deepens anyway when it is short.
        core = self.core
        if core is not None and self._gathered_about(core):
            rows = max(rows, 2, self.rows_max)
        self.rows = min(rows, self.rows_max)
        self.negotiated: list = []
        self.band = self._band(self.rows)
        if not self.band.any():
            return None, [{"part": "place", "check": "shoreline",
                           "why": "the band of land along this site's shore holds no "
                                  "dry column"}]
        return super().solve()

    def _band(self, rows: int):
        """The band `rows` district-rows deep, and the depth it works out at.

                The depth is a whole number of rows and their lanes. Found by running it: a
                band of 65 columns against a 28-column floor and a 6-column lane tiled into one
                row of 32 and a 27-column remainder that is nobody's -- the same defect the ring
                layout had at its chamfered corners, a size smaller.
                
        """
        gap = placesolve.LANE_GAP
        self.band_depth = rows * self.row_depth + (rows - 1) * gap
        return regions.shore_band(self.anchor, self.wet, self.X, self.Z,
                                  self.band_depth)

    # --- one part --------------------------------------------------------------

    def _gathered_about(self, d: dict) -> bool:
        """Is this part the object of an `around` relation whose subjects are the
        districts? The review's counterexample in one question: the square a village is
        gathered around belongs *in* the band the village is built in."""
        return any(r["relation"] == "around" and r.get("object_part") == d["name"]
                   and r.get("subject_groups") for r in self.relations)

    def centre_point(self) -> tuple | None:
        """The column a centre part the districts gather round stands at: the middle of
        the shore path, set inland by the shore road and the setback so the part stands
        in the band, on the landward side of the road along the water. None with no
        anchor. `stage_plateau` may cut its level ground here rather than at the site's
        middle (adapter request A8)."""
        if self.anchor is None:
            return None
        path = self.anchor["path"]
        mx, mz = path[len(path) // 2]
        inland = {"north": 1, "south": -1, "west": 1, "east": -1}[
            self.anchor["water_side"]]
        step = SHORE_SETBACK + SHORE_ROAD + self.least // 2
        if self.anchor["along"] == "x":
            return int(mx), int(mz + inland * step)
        return int(mx + inland * step), int(mz)

    def _cands_centre(self, d, name, k) -> tuple:
        """The centre of a shoreline place: at the site's middle as the parent has it,
        **unless the districts are gathered around it** (the closure round), in which
        case it stands in the band, at the shore's middle, where the ribbons the tiling
        cuts on either side of it and behind it are what "gathered around" means on a
        shore. The saved shore village put its square at the site's centre, forty
        columns inland of both its districts, and spent twelve repair passes moving it
        without ever being able to move the houses."""
        if not self._gathered_about(d) or spec_mod.compound(d):
            return super()._cands_centre(d, name, k)
        spots = self._centre_spots(d)
        if not spots:
            return super()._cands_centre(d, name, k)
        was = (self.cx, self.cz)
        cands, why = [], None
        seen = set()
        try:
            for at in spots:
                self.cx, self.cz = at
                self._anchor_least = self._gathering_least(at)
                got, why = super()._cands_centre(d, name, k)
                for c in got:
                    if tuple(c["rect"]) in seen:
                        continue
                    seen.add(tuple(c["rect"]))
                    c["leaf"]["notes"] = ("the thing the districts gather round, in the "
                                          "band among the ribbons the shore tiles into; "
                                          "solved")
                    cands.append(c)
        finally:
            self.cx, self.cz = was
        return cands, why

    def _gathering_least(self, at: tuple) -> tuple | None:
        """The least side a centre at `at` may have for the band's ribbons to stand
        within the gathering's reach of it: `intent.AROUND_REACH` times the anchor's
        half-diagonal has to reach the farthest ribbon's far corner. The expression
        round: an anchor sized from the programme is smaller than the band, and the
        ribbons stand where the shore puts them."""
        import math as _m
        from .intent import AROUND_REACH
        cols = getattr(self, "_band_cols", None)
        if cols is None or not len(cols):
            return None
        # the whole band: the ribbons the final cut lays lie inside it, wherever the
        # centre ends up, so the band's far edge is what the reach has to cover
        dx = cols[:, 0] - float(at[0])
        dz = cols[:, 1] - float(at[1])
        far = float(((dx * dx + dz * dz) ** 0.5).max())
        need_side = int(_m.ceil(far * _m.sqrt(2.0) / AROUND_REACH)) + 2
        return (need_side, f"the band's farthest column stands {far:.0f} from this spot "
                           f"and the gathering's reach is AROUND_REACH {AROUND_REACH:g} x "
                           f"the anchor's half-diagonal: at least {need_side} a side")

    def _centre_spots(self, d: dict) -> list:
        """Where a centre the districts gather round may stand on a shore: inside the
        band, among the ribbons the band tiles into **before** the centre is placed,
        nearest their middle first. The middle of the shore path is not it: on a
        diagonal shore the ribbons that fit lie off the path's middle, and a square at
        the middle had two of them on one side and none behind it."""
        import numpy as np
        if self.anchor is None or getattr(self, "band", None) is None:
            return []
        rows = max(self.rows, 2, int(getattr(self, "rows_max", 2)))
        try:
            mask, rects = self._cut(rows, placed=[])
        except Exception:                      # noqa: BLE001 -- no tiling, no spot
            return []
        if not rects:
            return []
        got = self._type_of(d)
        # sized from the programme where the rule sizes it (the expression round) -- and
        # never so small that the ribbons the band tiles into fall out of the relation's
        # reach (`intent.AROUND_REACH` times the anchor's half-diagonal)
        self._ribbon_rects = list(rects)
        _ii, _jj = np.nonzero(np.asarray(mask, bool))
        self._band_cols = np.stack([_ii + self.X, _jj + self.Z], axis=1).astype(float)
        sized = self._sized(d, got[1], got[1].get("kind", d["kind"])) if got else None
        side = (int(sized["side"]) if sized else
                placesolve._clean_side(got[1]) if got else self.least)
        area = sum((r[2] - r[0] + 1) * (r[3] - r[1] + 1) for r in rects)
        cx = sum((r[0] + r[2]) / 2.0 * (r[2] - r[0] + 1) * (r[3] - r[1] + 1)
                 for r in rects) / area
        cz = sum((r[1] + r[3]) / 2.0 * (r[2] - r[0] + 1) * (r[3] - r[1] + 1)
                 for r in rects) / area
        m = np.asarray(mask, bool)
        half = side // 2
        # **First, on the landward side of the shore road, at the depth a part of this
        # size stands** (the expression round): a centre sized from the programme is
        # smaller than the band, and set at the ribbons' centroid it stood between the
        # rows with neither in reach; on the road's side of the band the first row
        # flanks it and the second stands behind it, which is what gathered means here.
        path = self.anchor["path"]
        inland = {"north": 1, "south": -1, "west": 1, "east": -1}[self.anchor["water_side"]]
        depth = SHORE_SETBACK + SHORE_ROAD + half

        def inside(x, z):
            for (ax, az) in ((x - half, z - half), (x + half, z - half),
                             (x - half, z + half), (x + half, z + half)):
                i, j = ax - self.X, az - self.Z
                if not (0 <= i < m.shape[0] and 0 <= j < m.shape[1]) or not m[i, j]:
                    return False
            return True
        spots = []
        step = max(4, side // 4)
        for r in range(0, 6):
            ring = [(0, 0)] if r == 0 else [
                (dx * step * r, dz * step * r)
                for dx in (-1, 0, 1) for dz in (-1, 0, 1) if dx or dz]
            for dx, dz in ring:
                x, z = int(round(cx + dx)), int(round(cz + dz))
                if inside(x, z) and (x, z) not in spots:
                    spots.append((x, z))
            if len(spots) >= 5:
                break
        # ...and, after the ribbons' middle, the road's side of the band at the depth a
        # part of this size stands, so a small centre has the first row beside it and
        # the second behind
        mid = len(path) // 2
        for off in (0, 1, -1, 2, -2):
            i = mid + off * max(1, step)
            if not (0 <= i < len(path)):
                continue
            px, pz = path[i]
            x, z = ((int(px), int(pz + inland * depth)) if self.anchor["along"] == "x"
                    else (int(px + inland * depth), int(pz)))
            if inside(x, z) and (x, z) not in spots:
                spots.append((x, z))
        return spots[:7]

    def _cands_beside(self, d, name, k) -> tuple:
        """`beside_the_centre`, in a place that has no centre.

                A shoreline settlement has a **middle** even where it has no centre: the point
                on the shore its band is longest about. A green "beside the centre" of a village
                strung along the water goes there, on the landward side of the shore road, and
                the request is answered rather than refused for a centre nobody asked for.
                
        """
        if self.core is not None and self.by_name.get(self.core.get("name")) is not None:
            return super()._cands_beside(d, name, k)
        if self.anchor is None:
            return super()._cands_beside(d, name, k)
        got = self._type_of(d)
        if got is None:
            return [], (f"{name}: no committed type of this place's form builds a "
                        f"{d['family']} ({d['kind']})")
        tname, decl = got
        kind = decl.get("kind", d["kind"])
        path = self.anchor["path"]
        mx, mz = path[len(path) // 2]
        inland = {"north": 1, "south": -1, "west": 1, "east": -1}[
            self.anchor["water_side"]]
        sized = self._sized(d, decl, kind)
        side = int(sized["side"]) if sized else placesolve._clean_side(decl)
        step = SHORE_SETBACK + SHORE_ROAD + side // 2
        if self.anchor["along"] == "x":
            mz += inland * step
        else:
            mx += inland * step
        rect = (int(mx), int(mz), int(mx), int(mz))
        return self._around(rect, d, name, tname, decl, kind), None

    # --- the districts ---------------------------------------------------------

    def _deepen_inland(self, rect, part: dict, n: int, others: list) -> tuple:
        """`rect` grown away from the water, four columns at a time, until the density
        band's ceiling holds `n` houses or the site, a placed part or another ribbon
        stops it. Returns the rectangle reached."""
        side = self.anchor["water_side"]
        g = placesolve.LANE_GAP
        x0, z0, x1, z1 = [int(v) for v in rect]
        bx0, bz0, bx1, bz1 = self._bounds()
        blocked = list(others)
        for leaf in self.placed + self.compounds:
            rs = (pipeline.part_rects({**leaf, "name": leaf.get("name")})
                  if leaf.get("kind") else
                  [(leaf["x0"], leaf["z0"], leaf["x1"], leaf["z1"])])
            blocked.extend(rs)
        standing = {"parts": [dict(p) for p in self.placed], "arterials": {"cells": []}}
        for _ in range(64):
            band = placeplan.count_band({"name": "probe", "x0": x0, "z0": z0,
                                         "x1": x1, "z1": z1}, part, standing, self.decls)
            if band["hi"] >= n:
                break
            nx0, nz0, nx1, nz1 = x0, z0, x1, z1
            if side == "south":
                nz0 -= 4
            elif side == "north":
                nz1 += 4
            elif side == "east":
                nx0 -= 4
            else:
                nx1 += 4
            if not (bx0 <= nx0 and nx1 <= bx1 and bz0 <= nz0 and nz1 <= bz1):
                break
            if any(placesolve._overlaps((nx0, nz0, nx1, nz1),
                                        placesolve._grow(o, g)) for o in blocked):
                break
            x0, z0, x1, z1 = nx0, nz0, nx1, nz1
        return (x0, z0, x1, z1)

    def _blocked_mask(self, m, placed=None):
        """The parts already placed, and a lane round each, are not district ground."""
        g = placesolve.LANE_GAP
        for leaf in (placed if placed is not None else self.placed + self.compounds):
            rs = (pipeline.part_rects({**leaf, "name": leaf.get("name")})
                  if leaf.get("kind") else
                  [(leaf["x0"], leaf["z0"], leaf["x1"], leaf["z1"])])
            for (a0, b0, a1, b1) in rs:
                i0, j0 = max(0, a0 - g - self.X), max(0, b0 - g - self.Z)
                m[i0:max(0, a1 + g - self.X) + 1, j0:max(0, b1 + g - self.Z) + 1] = False
        return m

    def _cut(self, rows: int, groups: list | None = None, placed=None) -> tuple:
        """`(mask, rectangles)` for a band `rows` deep.

                **A row at a time, and each row exactly one district deep.** Tiling the whole
                band at once produced districts nearly square -- 70 along the shore and 69
                inland -- whose streets then ran inland and whose lots faced along the coast,
                which is the opposite of what was asked for. A shoreline district is a *ribbon*:
                the band is cut into rows parallel to the water, each `least` deep, and each row
                is tiled along its own length. The compiler's own rule (streets run along a
                district's longer side) then puts every front on a street parallel to the water,
                by construction. `placed` overrides the parts blocked out, so the centre can ask
                what the band tiles into before it chooses where to stand.
                
        """
        import numpy as np
        g = placesolve.LANE_GAP
        least = self.least
        along_x = self.anchor["along"] == "x"
        whole = self._blocked_mask(np.asarray(self._band(rows), bool).copy(), placed)
        keep_rects = []
        for k in range(rows):
            near = k * (self.row_depth + g)
            far = near + self.row_depth
            inner = (np.asarray(regions.shore_band(self.anchor, self.wet, self.X,
                                                   self.Z, near), bool)
                     if near else np.zeros_like(whole))
            outer = np.asarray(regions.shore_band(self.anchor, self.wet, self.X,
                                                  self.Z, far), bool)
            row = self._blocked_mask((outer & ~inner) & whole, placed)
            if not row.any():
                continue
            reg = regions.region(f"shore_row_{k}", row, self.X, self.Z,
                                 policy="shoreline", anchor=self.anchor,
                                 surface="built",
                                 notes=f"row {k + 1} of the shore band, "
                                       f"{self.row_depth} columns deep, {near} "
                                       f"behind the water line")
            for (x0, z0, x1, z1) in regions.tile(reg, minimum=least, gap=g,
                                                 most=regions.TILE_MAX):
                w, d = x1 - x0 + 1, z1 - z0 + 1
                if (w >= d) == along_x and w >= least and d >= least:
                    keep_rects.append((x0, z0, x1, z1))
        self.band_tiling = "rows"
        if not keep_rects and rows > 1:
            # **A band that follows the shore's own depth** (the closure round's held-
            # out hamlet). On a shore that runs at a slope, each row is a diagonal
            # ribbon whose axis-aligned rectangles are shallower than the row -- 21 and
            # 26 columns deep in rows of 38 on a 156 site, under the 28 a district needs
            # -- so a row at a time found nothing where the band as a whole holds a
            # 116x40 ribbon. The whole band is tiled instead, and only rectangles longer
            # along the shore than across it are kept, which is the rule that puts the
            # streets along the water and the lots' fronts on them.
            reg = regions.region("shore_band", whole, self.X, self.Z, policy="shoreline",
                                 anchor=self.anchor, surface="built",
                                 notes=f"the whole band, {self.band_depth} columns deep, "
                                       f"tiled at once because its rows hold no district")
            for (x0, z0, x1, z1) in regions.tile(reg, minimum=least, gap=g,
                                                 most=regions.TILE_MAX):
                w, d = x1 - x0 + 1, z1 - z0 + 1
                if (w >= d) == along_x and w >= least and d >= least:
                    keep_rects.append((x0, z0, x1, z1))
            if keep_rects:
                self.band_tiling = "whole"
        return whole, keep_rects

    def _districts(self, groups: list) -> None:
        """The band along the shore, tiled, and one route down it.

                The parent cuts four strips out of the box round a centre. Here the ground is a
                mask and the districts are its tiling, which is the operation both policies now
                share (`regions.tile`).
                
        """
        if not groups:
            return
        g = placesolve.LANE_GAP
        least = self.least
        along_x = self.anchor["along"] == "x"
        want = int(self.spec.get("structures") or 0)

        def _blocked(m):
            """The parts already placed, and a lane round each, are not district
            ground."""
            for leaf in self.placed + self.compounds:
                rs = (pipeline.part_rects({**leaf, "name": leaf.get("name")})
                      if leaf.get("kind") else
                      [(leaf["x0"], leaf["z0"], leaf["x1"], leaf["z1"])])
                for (a0, b0, a1, b1) in rs:
                    i0, j0 = max(0, a0 - g - self.X), max(0, b0 - g - self.Z)
                    m[i0:max(0, a1 + g - self.X) + 1,
                      j0:max(0, b1 + g - self.Z) + 1] = False
            return m

        def cut(rows: int) -> tuple:
            return self._cut(rows, groups)

        def _unused_cut(rows: int) -> tuple:
            """`(mask, rectangles)` for a band `rows` deep.

                        **A row at a time, and each row exactly one district deep.** Tiling the
                        whole band at once produced districts nearly square -- 70 along the shore
                        and 69 inland -- whose streets then ran inland and whose lots faced along
                        the coast, which is the opposite of what was asked for. A shoreline
                        district is a *ribbon*: the band is cut into rows parallel to the water,
                        each `least` deep, and each row is tiled along its own length. The
                        compiler's own rule (streets run along a district's longer side) then puts
                        every front on a street parallel to the water, by construction.
                        
            """
            import numpy as np
            whole = _blocked(np.asarray(self._band(rows), bool).copy())
            keep_rects = []
            for k in range(rows):
                near = k * (self.row_depth + g)
                far = near + self.row_depth
                inner = (np.asarray(regions.shore_band(self.anchor, self.wet, self.X,
                                                       self.Z, near), bool)
                         if near else np.zeros_like(whole))
                outer = np.asarray(regions.shore_band(self.anchor, self.wet, self.X,
                                                      self.Z, far), bool)
                row = _blocked((outer & ~inner) & whole)
                if not row.any():
                    continue
                reg = regions.region(f"shore_row_{k}", row, self.X, self.Z,
                                     policy="shoreline", anchor=self.anchor,
                                     surface="built",
                                     notes=f"row {k + 1} of the shore band, "
                                           f"{self.row_depth} columns deep, {near} "
                                           f"behind the water line")
                for (x0, z0, x1, z1) in regions.tile(reg, minimum=least, gap=g,
                                                     most=regions.TILE_MAX):
                    w, d = x1 - x0 + 1, z1 - z0 + 1
                    if (w >= d) == along_x and w >= least and d >= least:
                        keep_rects.append((x0, z0, x1, z1))
            return whole, keep_rects

        standing = {"parts": [dict(p) for p in self.placed], "arterials": {"cells": []}}

        def band_of(r, p):
            return placeplan.count_band({"name": "probe", "x0": r[0], "z0": r[1],
                                         "x1": r[2], "z1": r[3]}, p, standing, self.decls)

        def holds(rects: list) -> int:
            """The most the ribbons hold at the group's word (the band's ceiling at the
            smallest lot), from the one density definition -- the closure round; the
            block-share estimate asked a sloped shore's ribbon for six of ten."""
            n = 0
            for r in rects:
                p = groups[0]
                area = (r[2] - r[0] + 1) * (r[3] - r[1] + 1)
                cap = int(area * placesolve.DISTRICT_FILL // spec_mod.columns_per_plot(p))
                n += max(1, min(cap, band_of(r, p)["hi"]))
            return n

        # **The band's depth is negotiated, inside a bound that is written down.** A
        # shore settlement is a ribbon and the site's footprint was derived for a square
        # place, so the band `SHORE_DEPTH` gives may hold fewer houses than the kind's
        # band asks for. How deep the ribbon runs is an *inferred* choice and may be
        # revised; how many houses an explicit sentence asked for is not. So the depth
        # grows a row at a time up to `SHORE_DEPTH_MAX` and every step is recorded --
        # and where the deepest admitted band is still short, the place is short and
        # `intent.coverage` says so. A cap is not a capacity.
        mask, fitted = cut(self.rows)
        while want and holds(fitted) < want and self.rows < self.rows_max:
            self.rows += 1
            m2, f2 = cut(self.rows)
            self.negotiated.append(
                {"what": "band_depth", "to_rows": self.rows,
                 "depth": int(self.band_depth), "holds": holds(f2), "wanted": want,
                 "bound": {"SHORE_DEPTH_MAX": SHORE_DEPTH_MAX,
                           "rows_max": self.rows_max},
                 "why": "the band as inferred holds fewer structures than the place "
                        "asks for; its depth is an inferred choice and is deepened "
                        "inside the registered bound"})
            if holds(f2) <= holds(fitted):
                break
            mask, fitted = m2, f2
        self.band = mask
        if not fitted:
            self.fails.append({"part": groups[0]["name"], "check": "district",
                               "why": f"the shore band holds no rectangle "
                                      f"{least} on a side for a district"})
            return
        # ordered along the shore, so `shore_1` is at one end and `shore_n` at the other
        fitted.sort(key=lambda r: (r[0] if along_x else r[1], r[1], r[0]))
        sectors = [(f"shore_{i + 1}", r) for i, r in enumerate(fitted)]

        assign: dict = {}
        remaining = list(sectors)
        for p in groups:
            if p["relation"] == "quarter":
                for _k in range(int(p["count"])):
                    if remaining:
                        assign[remaining.pop(0)[0]] = p
        through = [p for p in groups if p["relation"] in ("throughout", "along", "edge")]
        for i, s in enumerate(remaining):
            if through:
                assign[s[0]] = through[i % len(through)]
        if not assign:
            for i, s in enumerate(sectors):
                assign[s[0]] = groups[i % len(groups)]

        # **The frontage the sentence asked for, as a side of the world per district.**
        # Local, from the shore path beside each district and not from the anchor's one
        # global `water_side`: on a bay the water is east of one district and north of
        # the next, and a single label would put half the doors in the wrong wall.
        self.faces_want = wants_frontage(self.spec)
        rows = []
        for label, r in sectors:
            p = assign.get(label)
            if p is None:
                continue
            area = (r[2] - r[0] + 1) * (r[3] - r[1] + 1)
            per = spec_mod.columns_per_plot(p)
            cap = int(area * placesolve.DISTRICT_FILL // per)
            band = band_of(r, p)
            rows.append({"label": label, "rect": r, "part": p, "area": area,
                         "cap": cap, "band": band,
                         "from_ground": max(1, min(cap, band["mid"])),
                         "most": max(1, min(cap, band["hi"]))})
        declared = int(self.spec.get("structures") or 0)
        want = declared or sum(row["from_ground"] for row in rows)
        # **A sector the ground fits fewer than two houses in is not a district.** The
        # same rule the relation solver has (`placesolve.DISTRICT_MIN_STRUCTURES`),
        # which this step did not have: a staircase of districts down a diagonal shore
        # leaves steps that hold one house or none, and a district holding one house is
        # held to a district's count and cover and can meet neither. They become left-
        # over ground, which is what they are.
        small = [r for r in rows
                 if r["from_ground"] < placesolve.DISTRICT_MIN_STRUCTURES]
        if small and len(small) < len(rows):
            self.left_over = [{"label": r["label"], "rect": list(r["rect"]),
                               "area": r["area"], "structures": int(r["from_ground"]),
                               "why": f"under {placesolve.DISTRICT_MIN_STRUCTURES} "
                                      f"structures at this density"} for r in small]
            rows = [r for r in rows if r not in small]
            sectors = [(l, rc) for l, rc in sectors
                       if any(r["label"] == l for r in rows)]
        # **the sentence's number over the ribbons, capped at each ribbon's ceiling**:
        # what the ceilings cannot hold is short and reported, never laid denser
        counts = placesolve._largest_remainder(
            [row["from_ground"] for row in rows], want,
            caps=[row["most"] for row in rows])
        self.count_short = max(0, want - sum(counts))
        # **An exact count is laid whole**: what the ribbons' ceilings cannot hold at
        # their first depth goes on the ribbons by area anyway, and where that puts a
        # district over its word's ceiling the record says so (`over_ceiling`) and the
        # layout owner's extent action grows it inland before any number moves. The
        # held-out hamlet: ten houses capped at eight by a 40-deep ribbon the ladder
        # then grew to hold them.
        if self.exact and self.count_short:
            more = placesolve._largest_remainder([row["area"] for row in rows],
                                                 int(self.count_short), floor=0)
            counts = [c + m for c, m in zip(counts, more)]
            self.count_over_ceiling = True
            self.count_short = 0
        # **...and the ribbon is deepened inland until its word's ceiling holds its
        # count**, where the site has the ground: a ribbon is as deep as the houses on
        # it need to be low, not as deep as one row. Inland only -- never toward the
        # water -- and never into another ribbon or a placed part.
        if self.exact:
            for row, n in zip(rows, counts):
                extra0 = 2 * len(((row["part"].get("character") or {}).get("landmarks")) or [])
                if n + extra0 <= row["most"]:
                    continue
                # a landmark is laid on a lot of its own, larger than a house's
                extra = 2 * len(((row["part"].get("character") or {}).get("landmarks")) or [])
                grown = self._deepen_inland(row["rect"], row["part"], n + extra,
                                            [r["rect"] for r in rows if r is not row])
                if grown != tuple(row["rect"]):
                    row["deepened"] = {"from": list(row["rect"]), "to": list(grown),
                                       "for": int(n)}
                    row["rect"] = tuple(grown)
                    row["band"] = band_of(grown, row["part"])
                    row["most"] = max(1, row["band"]["hi"])
                    row["area"] = (grown[2] - grown[0] + 1) * (grown[3] - grown[1] + 1)
            self.count_over_ceiling = any(n > row["most"] for row, n in zip(rows, counts))
        sectors = [(row["label"], tuple(row["rect"])) for row in rows]
        for row, n in zip(rows, counts):
            p, r = row["part"], row["rect"]
            role = p.get("role") or spec_mod.read_role(None, p, p["name"])
            faces = (water_direction(self.anchor, (r[0] + r[2]) // 2,
                                     (r[1] + r[3]) // 2)
                     if self.faces_want == "water" else None)
            # the shore beside *this* district, so the compiler can ask the question
            # again per block; a district is a rectangle and a shore is not
            near = _path_near(self.anchor, r)
            self.districts.append({
                "name": f"{p['name']}_{row['label']}", "x0": r[0], "z0": r[1],
                "x1": r[2], "z1": r[3],
                "structures": int(max(1, min(n, row["cap"]))),
                # the four columns of this region, fixed at resolution
                # (`placeregion.column_record`)
                **regions.column_record(
                    (r[0], r[1], r[2], r[3]),
                    why=[f"sector {row['label']} of the shore band of `{p['name']}`, "
                         f"fixed when the place was solved"]),
                "defines": p["name"],
                **({"exact": True} if self.exact else {}),
                "count_band": {k: row["band"][k] for k in ("lo", "mid", "hi", "usable")},
                **({"faces": faces, "faces_anchor": near} if faces else {}),
                **({"voice": p["voice"]} if p.get("voice") else {}),
                "purpose": (f"{p.get('notes') or p['name']} A "
                            f"{p.get('density') or 'medium'}, {role} district on the "
                            f"shore; its streets run along the water and its lots front "
                            f"them."),
                "notes": (f"Sector {row['label']} of the band of land within "
                          f"{self.band_depth} columns of the shoreline, on the "
                          f"{'north/south' if along_x else 'west/east'} grain the water "
                          f"itself makes. Laid out against the ground's own water line, "
                          f"not against a centre.")})
        covered = sum(row["area"] for row in rows)
        band_cols = int(self.band.sum())
        self.district_record = {
            "policy": "shoreline",
            "anchor": {k: v for k, v in self.anchor.items() if k != "path"},
            "anchor_path": self.anchor["path"],
            "band_depth": int(self.band_depth), "band_columns": band_cols,
            "rows": int(self.rows), "rows_max": int(self.rows_max),
            "tiling": getattr(self, "band_tiling", "rows"),
            "row_depth": int(self.row_depth), "slope": round(self.slope, 3),
            "negotiated": list(self.negotiated),
            "sectors": [{"label": row["label"], "rect": list(row["rect"]),
                         "group": row["part"]["name"],
                         **({"deepened": row["deepened"]} if row.get("deepened") else {})}
                        for row in rows],
            "structures": {"declared": declared,
                           "laid": sum(int(d["structures"]) for d in self.districts),
                           "caps": [row["cap"] for row in rows],
                           "bands": [row["band"] for row in rows],
                           "short": int(getattr(self, "count_short", 0)),
                           "laid_over_ceiling": bool(getattr(self, "count_over_ceiling", False)),
                           "exact": self.exact},
            "coverage": round(covered / float(band_cols), 4) if band_cols else 0.0,
            "faces": "water",
            "relations": self._relations_predicted(),
            "left_over": list(self.left_over)}

    # --- the document -----------------------------------------------------------

    def _place_doc(self) -> dict:
        doc = super()._place_doc()
        doc["layout"]["case"] = "shoreline"
        doc["layout"]["policy"] = "shoreline"
        doc["layout"]["anchor"] = {k: v for k, v in (self.anchor or {}).items()
                                   if k != "path"}
        doc["layout"]["anchor_path"] = (self.anchor or {}).get("path")
        doc["layout"]["faces"] = "water"
        doc["layout"]["registered"]["SHORE_DEPTH"] = regions.SHORE_DEPTH
        doc["layout"]["registered"]["SHORE_MIN_WATER"] = regions.SHORE_MIN_WATER
        doc["layout"]["registered"]["SHORE_ROAD"] = SHORE_ROAD
        doc["intent"] = (
            f"A {self.spec['kind']} laid out along the shoreline the ground itself "
            f"makes: {len(self.districts)} district(s) in a band up to "
            f"{getattr(self, 'band_depth', 0)} columns deep behind the water line, each "
            f"longer along the shore than across it so its streets run beside the water "
            f"and its lots front them. "
            + ("No wall: the request says the place is unwalled. "
               if not self.wall else "")
            + ("No centre: nothing is declared at one, and the layout is against the "
               "water rather than around a point. "
               if self.core is None else "")
            + (f"{self.spec['invariants']}" if self.spec.get("invariants") else ""))
        return doc


def shoreline_layout(spec: dict, site: dict, plateau: dict | None, decls: dict,
                     voice: str, vol=None, seed: int = 1,
                     caps: dict | None = None, intent: dict | None = None) -> tuple:
    """The shoreline policy, in the shape every policy answers in: `(place, fails)`."""
    return ShoreSolver(spec, site, plateau, decls, voice, vol=vol, seed=seed,
                       caps=caps, intent=intent).solve()


def clamp_to_land(place: dict | None, rect) -> list | None:
    """A rectangle of a shoreline place, cut back so it stops the shore road and setback
    short of the water: the shore path is the water's edge, and ground claimed beyond it
    is water. `rect` unchanged for a place that is not on a shore; None where nothing
    of it is left on land."""
    lay = (place or {}).get("layout") or {}
    path = lay.get("anchor_path") or []
    anchor = lay.get("anchor") or {}
    if lay.get("policy") != "shoreline" or not path or not anchor.get("water_side"):
        return list(rect)
    x0, z0, x1, z1 = [int(v) for v in rect]
    keep = SHORE_SETBACK + SHORE_ROAD
    side = anchor["water_side"]
    along_x = anchor.get("along") == "x"
    pts = [(int(a[0]), int(a[1])) for a in path]
    if along_x:
        near = [pz for px, pz in pts if x0 - 8 <= px <= x1 + 8] or [pz for _px, pz in pts]
        if side == "south":
            z1 = min(z1, min(near) - keep - 1)
        else:
            z0 = max(z0, max(near) + keep + 1)
    else:
        near = [px for px, pz in pts if z0 - 8 <= pz <= z1 + 8] or [px for px, _pz in pts]
        if side == "east":
            x1 = min(x1, min(near) - keep - 1)
        else:
            x0 = max(x0, max(near) + keep + 1)
    if x1 < x0 or z1 < z0:
        return None
    return [x0, z0, x1, z1]


def centre_point(spec: dict, site: dict, vol, intent: dict | None = None) -> tuple | None:
    """Where a shoreline place's centre part stands when its districts gather round it,
    for the plateau stage to level: `(x, z)` or None. See `ShoreSolver.centre_point`."""
    try:
        solver = ShoreSolver(spec, site, None, {}, "", vol=vol, intent=intent)
    except Exception:                          # noqa: BLE001 -- no shore, no point
        return None
    core = solver.core
    if solver.anchor is None or core is None or not solver._gathered_about(core):
        return None
    solver.least = max(SHORE_MIN, solver.dmin)
    return solver.centre_point()


def faces_water(place: dict) -> tuple:
    """Do the districts of this plan actually face the water? `(bool, why)`.

        Asked of the geometry and not of the prose: every district must be longer along the
        shore's own axis than across it, because the compiler runs its streets along a
        district's longer side and puts the lots' fronts on them. A layout that merely
        *says* the houses face the water fails this.

        **This is a necessary condition and it is not the requirement.** The integration
        review's second finding, and it is the sharpest thing in the review: the saved
        village passed this check with nine of its fifteen homes fronting *away* from the
        water it recorded. Both facts were true. A ribbon district has its streets parallel
        to the shore, and a lot on the landward side of such a street faces inland -- the
        aspect ratio says where the streets run and says nothing at all about which way a
        door is. `fronts_facing_water` below is the requirement; this is a property of the
        tiling.
        
    """
    lay = (place or {}).get("layout") or {}
    anchor = lay.get("anchor") or {}
    if lay.get("policy") != "shoreline" or not anchor:
        return (False, "this plan was not laid out by the shoreline policy")
    along_x = anchor.get("along") == "x"
    bad = []
    for d in place.get("districts") or []:
        w, h = d["x1"] - d["x0"] + 1, d["z1"] - d["z0"] + 1
        if (w >= h) != along_x:
            bad.append(d["name"])
    return (not bad,
            f"every district's long side runs along the shore ({anchor.get('along')})"
            if not bad else
            f"{len(bad)} district(s) are deeper than they are long, so their streets "
            f"run inland and their lots face along the coast: {bad[:4]}")


#: The four side names, as the step you take walking out of a front door on that side.
#: `front: "north"` means the street this lot fronts is to its north, so a person
#: leaving it walks north -- `-z`. The plan, the compiler's `_Frame.front` and this all
#: mean the same thing by the word and it is written down here because the one time they
#: did not, fifteen houses faced the wrong way and every check passed.
STEP = {"north": (0, -1), "south": (0, 1), "west": (-1, 0), "east": (1, 0)}

OPPOSITE = {"north": "south", "south": "north", "west": "east", "east": "west"}


def water_direction(anchor: dict, x: int, z: int) -> str | None:
    """Which way the water lies from column `(x, z)`: `north`, `south`, `west` or `east`.

        **Local, from the shore's own path.** The anchor carries one `water_side` for the
        whole site, which is a fact about the site's grain and not about any particular
        house: on a bay, a headland or any shore that curves, the water is in a different
        direction from each arm of it. So the direction here is the step from this column
        to the nearest column of the shore path, reduced to its dominant axis -- which is
        the same answer as `water_side` on a straight shore and a different one on a curved
        shore, which is the entire point.
        
    """
    path = [(int(p[0]), int(p[1])) for p in (anchor or {}).get("path") or []]
    if not path:
        side = (anchor or {}).get("water_side")
        return side if side in STEP else None
    px, pz = min(path, key=lambda p: (p[0] - x) ** 2 + (p[1] - z) ** 2)
    dx, dz = px - int(x), pz - int(z)
    if dx == 0 and dz == 0:
        side = (anchor or {}).get("water_side")
        return side if side in STEP else None
    if abs(dx) >= abs(dz):
        return "east" if dx > 0 else "west"
    return "south" if dz > 0 else "north"


def fronts_facing_water(anchor: dict, parts: list) -> dict:
    """Which way every plot in `parts` actually fronts, against the water beside it.

        Returns `{"toward", "along", "away", "unknown", "rows"}`. A front is

          `toward`   the door opens on the side the water is on;
          `along`    the door opens along the shore -- neither at the water nor away from
                     it, which is what a lot on a cross-street does and is not a defect;
          `away`     the door opens on the side **opposite** the water. This is the one a
                     request for homes facing the water is broken by, and it is the one the
                     saved village had nine of.

        No threshold and no share: "most of them face the water" is a number somebody would
        have to choose, and the geometric statement that needs no choosing is that a place
        whose homes face the water has **no home fronting away from it**.
        
    """
    out = {"toward": 0, "along": 0, "away": 0, "unknown": 0, "rows": []}
    for p in parts:
        if p.get("kind", "plot") != "plot":
            continue
        front = p.get("front")
        if front not in STEP:
            out["unknown"] += 1
            out["rows"].append({"part": p.get("name"), "front": front,
                                "water": None, "class": "unknown"})
            continue
        try:
            cx = (int(p["x0"]) + int(p["x1"])) // 2
            cz = (int(p["z0"]) + int(p["z1"])) // 2
        except (KeyError, TypeError, ValueError):
            out["unknown"] += 1
            continue
        towards = water_direction(anchor, cx, cz)
        if towards is None:
            out["unknown"] += 1
            continue
        kind = ("toward" if front == towards else
                "away" if front == OPPOSITE[towards] else "along")
        out[kind] += 1
        out["rows"].append({"part": p.get("name"), "front": front, "water": towards,
                            "class": kind, "at": [cx, cz]})
    return out
