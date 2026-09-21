"""One spatial interface: a region is a mask, a boundary, an anchor, routes and a level.

(`ethoslm.regions` is a different module and always was: it reads the world save's region
files. This one is about the ground a *design* owns, which is why it is named for the
place rather than for the world.)

Until this module the whole-place geometry was concentric arithmetic. That is not the
same as "only rings work" -- the relation solver lays ringless places perfectly well --
but it *is* true that the only shared vocabulary between a layout and the compiler was a
**rectangle**. The walls were drawn round and the districts inside them were still four
axis-aligned strips clipped at the ends, so the lower ring covered 54.5% of its own
annulus against a 60% bar, and every corner between the chamfer and the strips was
ground nobody owned.

So a region is written down, once:

    mask      a boolean grid over the site, `True` where the region owns the ground.
              Every policy produces one; nothing downstream cares how.
    boundary  the closed path round it, where it has one (a ring wall's line).
    anchor    what it is laid out *against*: the centre of a concentric place, the
              **shoreline** of a coastal one. `{"kind", "path", "faces"}`.
    routes    the polylines through it -- the shore road, the arterials.
    level     the terrace level its ground is brought to, or None.
    surface   what the ground it does not build on is for.

and two operations both policies share: `tile`, which cuts a region into the rectangles
a district compiler can lay out, and `frontage`, which says which way the buildings in
it face.

`tile` is the part that matters. It takes the mask -- **not** a rectangle -- so a ring
with chamfered corners, a band following a river and a district with a lake in it are
all the same problem, and a region that is not a rectangle stops being ground nobody
owns.
"""
from __future__ import annotations

import math

import numpy as np


#: How coarse the rectangle search is, in columns. A district boundary is not meaningful
#: to the column, the search is quadratic in the grid, and four columns is under a
#: lane's width -- so the tiling is done on a grid of this and mapped back. Registered
#: before the coverage numbers that test it.
TILE_STEP = 4

#: The least a tiled rectangle may be on a side before it is not a district. Below this
#: the compiler has nowhere to put a street, and `concentric_layout` has always used the
#: same floor (`2 * SITE_INSET + 24`).
TILE_MIN = 24

#: How many rectangles one region is cut into at most. A district is one compiler call
#: and a place of five hundred of them is not a place.
TILE_MAX = 12


# ------------------------------------------------------------------ the record

def region(name: str, mask, x0: int, z0: int, *, policy: str = "", role=None,
           defines=None, boundary=None, inner=None, anchor=None, level=None,
           routes=None, surface: str = "built", density=None, voice=None,
           requirement=None, notes: str = "") -> dict:
    """One region, as the record everything downstream reads.

        `mask` is a boolean array indexed `[x - x0, z - z0]`; `x0, z0` is its world origin.
        Held as an array on the live record and dropped when the record is written to
        disk -- `serialise` keeps the boundary, the rectangles and the counts, which is
        what a reader of `resolution.json` wants and a 512x512 bitmap is not.
        
    """
    mask = np.asarray(mask, bool)
    xs, zs = np.nonzero(mask)
    rect = ([int(x0 + xs.min()), int(z0 + zs.min()),
             int(x0 + xs.max()), int(z0 + zs.max())] if xs.size else None)
    return {"name": name, "policy": policy, "role": role, "defines": defines,
            "mask": mask, "origin": [int(x0), int(z0)], "rect": rect,
            "boundary": boundary, "inner": inner, "anchor": anchor, "level": level,
            "routes": list(routes or []), "surface": surface, "density": density,
            "voice": voice, "requirement": requirement, "lots": None,
            "columns": int(mask.sum()), "notes": notes}


def serialise(r: dict) -> dict:
    """The region without its bitmap: what goes in `resolution.json`."""
    from . import contracts
    out = {k: v for k, v in r.items() if k in contracts.REGION_FIELDS}
    out["name"] = r["name"]
    return out


def columns(r: dict) -> int:
    return int(np.asarray(r["mask"], bool).sum())


# ----------------------------------------------------- the four columns of a region The
# design round's second contract. The expression round's defect, in one line: the ring
# layout shortened a dense sector to what its count needs and called the remainder
# `surface: open`, and `intent.lot_cover` then dropped every open region from its
# denominator -- so the ground the requirement was about shrank as the allocation
# shrank, and "27.5% of a dense ring" was 27.5% of the part of the ring that happened to
# have houses on it. A measurement whose denominator moves with the answer is not a
# measurement. So a region carries **four** counts and no consumer may substitute one
# for another: scope_columns the ground the requirement is about. Fixed when the design
# is resolved and independent of every later allocation: an inferred open remainder is
# still the ring's ground and is still in here. developable_columns of that, what the
# compiler may build on -- the arterial's band and the standing parts' clearances
# removed. allocated_columns the lots the compiler drew. A lot is ground promised to a
# building; it is not a building. built_columns the footprint that actually stands.
# Larger empty lots raise `allocated_columns` and leave this where it was, which is the
# whole point of keeping them apart. Only ground an **explicit** requirement asks to be
# open leaves the scope, and then it says which requirement asked and how much it took
# (`open_requested`). An inferred remainder records where it was cut from (`scope_of`)
# and stays in.

#: The fields this contract writes on a region record. Registered so a consumer can
#: assert that it read the column it meant to read.
COLUMN_FIELDS = ("rect_columns", "scope_columns", "developable_columns",
                 "allocated_columns", "built_columns", "open_requested", "scope_of",
                 "columns_from")


def rect_columns(rect) -> int:
    """The columns of an inclusive `(x0, z0, x1, z1)`."""
    x0, z0, x1, z1 = (int(v) for v in rect)
    return (abs(x1 - x0) + 1) * (abs(z1 - z0) + 1)


def column_record(rect, *, developable=None, allocated=None, built=None,
                  open_requested=None, scope_of=None, why=()) -> dict:
    """The four columns of one region, as the record every consumer reads.

        `open_requested` is the **requirement id** of an explicit requirement that asks this
        ground to be open -- and only then: naming a remainder `open` is an allocation
        decision and is not independent justification for removing it from the requested
        scope. Ground an explicit requirement asks to be open leaves the scope and the
        record says how much left and on whose authority.
        
    """
    total = rect_columns(rect)
    src = list(why)
    scope = total
    if open_requested:
        scope = 0
        src.append(f"requirement {open_requested} asks this ground to be open: {total} "
                   f"column(s) leave the scope of the density it is measured against")
    elif scope_of:
        src.append(f"an inferred remainder of `{scope_of}`: it is still that region's "
                   f"ground and stays inside the scope")
    return {"rect_columns": int(total), "scope_columns": int(scope),
            "developable_columns": None if developable is None else int(developable),
            "allocated_columns": None if allocated is None else int(allocated),
            "built_columns": None if built is None else int(built),
            "open_requested": str(open_requested) if open_requested else None,
            "scope_of": str(scope_of) if scope_of else None,
            "columns_from": src}


def denominator(regions) -> dict:
    """The ground a scoped measurement is over: `{"columns", "of", "regions", "why"}`.

        `developable_columns` where every region in scope records one -- the compiler cannot
        build on an arterial's band and a floor asked of ground nobody may build on is a
        floor about somebody else's decision -- and `scope_columns` otherwise. Regions an
        explicit requirement asked to be open are already out, by their own record.
        
    """
    scope = dev = 0
    n, all_dev = 0, True
    for r in regions or []:
        sc = r.get("scope_columns")
        if sc is None:
            sc = rect_columns(r["rect"]) if r.get("rect") else 0
        if not sc:
            continue
        n += 1
        scope += int(sc)
        if r.get("developable_columns") is None:
            all_dev = False
        else:
            dev += min(int(r["developable_columns"]), int(sc))
    use = dev if (all_dev and dev) else scope
    return {"columns": int(use), "of": "developable_columns" if (all_dev and dev)
            else "scope_columns", "regions": n, "scope_columns": int(scope),
            "why": ("the ground the requirement is about, as it was fixed at "
                    "resolution: an inferred open remainder is inside it")}


# ------------------------------------------------------------------ the tiling

def _largest_rect(grid) -> tuple:
    """The largest all-true axis-aligned rectangle in a boolean grid.

        The histogram method, per row, with the usual stack. Returns
        `(i0, j0, i1, j1, area)` in grid indices, or `None` where the grid is empty.
        
    """
    rows, cols = grid.shape
    heights = np.zeros(cols, int)
    best = None
    for i in range(rows):
        heights = np.where(grid[i], heights + 1, 0)
        stack: list = []
        for j in range(cols + 1):
            h = int(heights[j]) if j < cols else 0
            start = j
            while stack and stack[-1][1] >= h:
                s, sh = stack.pop()
                area = sh * (j - s)
                if sh and (best is None or area > best[4]):
                    best = (i - sh + 1, s, i, j - 1, area)
                start = s
            stack.append((start, h))
    return best


def tile(r: dict, *, step: int = TILE_STEP, minimum: int = TILE_MIN,
         gap: int = 6, most: int = TILE_MAX, cover: float = 0.0) -> list:
    """Cut a region into the rectangles a district compiler can lay out.

        Greedy largest-rectangle-first on a coarse grid, each taken rectangle cleared with a
        `gap` margin so two districts never share a column and the lanes have somewhere to
        run. Stops at `most` rectangles, or when the largest that remains is under
        `minimum` on a side, or when `cover` of the region's columns are covered.

        Deterministic: the same mask gives the same rectangles in the same order, which is
        what lets a plan be replayed.
        
    """
    mask = np.asarray(r["mask"], bool)
    x0, z0 = r["origin"]
    free = mask.copy()
    g = mask[::step, ::step].copy()
    pad = max(1, int(math.ceil(gap / float(step))))
    need = max(1, int(math.ceil(minimum / float(step))))
    total = float(mask.sum()) or 1.0
    out, taken = [], 0
    for _ in range(most):
        got = _largest_rect(g)
        if got is None:
            break
        i0, j0, i1, j1, _area = got
        if (i1 - i0 + 1) < need or (j1 - j0 + 1) < need:
            break
        # Back to world columns, **grown by a cell in every direction before the fine
        # trim**. Found by running the shore case: a band 28 columns deep whose first
        # owned column is not a multiple of `step` came back 27 deep -- one short of the
        # floor -- and the district was thrown away for an artefact of the search grid.
        # The coarse rectangle is a *seed*; the rectangle that is kept is the largest
        # wholly-free one in the window around it, at column resolution.
        wx0 = max(0, i0 * step - (step - 1))
        wz0 = max(0, j0 * step - (step - 1))
        wx1 = min(mask.shape[0] - 1, i1 * step + 2 * step - 2)
        wz1 = min(mask.shape[1] - 1, j1 * step + 2 * step - 2)
        sub = free[wx0:wx1 + 1, wz0:wz1 + 1]
        in_box = _largest_rect(sub) if sub.size else None
        if in_box is None:
            g[i0:i1 + 1, j0:j1 + 1] = False
            continue
        a0, b0, a1, b1, _ = in_box
        rx0, rz0 = x0 + wx0 + a0, z0 + wz0 + b0
        rx1, rz1 = x0 + wx0 + a1, z0 + wz0 + b1
        if (rx1 - rx0 + 1) < minimum or (rz1 - rz0 + 1) < minimum:
            g[i0:i1 + 1, j0:j1 + 1] = False
            continue
        out.append((rx0, rz0, rx1, rz1))
        taken += (rx1 - rx0 + 1) * (rz1 - rz0 + 1)
        # cleared from both grids, with the lane round it, so the next rectangle never
        # touches this one and the fine trim cannot grow back into it
        free[max(0, rx0 - x0 - gap):rx1 - x0 + 1 + gap,
             max(0, rz0 - z0 - gap):rz1 - z0 + 1 + gap] = False
        g[max(0, i0 - pad):i1 + 1 + pad, max(0, j0 - pad):j1 + 1 + pad] = False
        g &= free[::step, ::step]
        if cover and taken / total >= cover:
            break
    return out


def mask_of_rects(rects: list, x0: int, z0: int, w: int, d: int):
    """The mask a list of rectangles covers. What a policy hands `region` when its
    districts are already rectangles."""
    m = np.zeros((w, d), bool)
    for (a0, b0, a1, b1) in rects:
        m[max(0, a0 - x0):max(0, a1 - x0 + 1), max(0, b0 - z0):max(0, b1 - z0 + 1)] = True
    return m


# --------------------------------------------------------------- the ring case

def annulus_mask(x0: int, z0: int, S: int, cx: int, cz: int, inner: int, outer: int,
                 *, chamfer_in: int = 0, chamfer_out: int = 0):
    """The ground between two concentric boundaries, square or chamfered.

        The chamfer is the octagon's corner cut, in columns, and it is applied to **both**
        boundaries: a ring inside a round wall is round on its outside, and a ring outside a
        round wall is round on its inside. That second half is the one the strip layout
        never had, and it is why an octagonal place had square districts in a round wall.
        
    """
    xs = np.arange(x0, x0 + S)[:, None] - cx
    zs = np.arange(z0, z0 + S)[None, :] - cz
    ax, az = np.abs(xs), np.abs(zs)
    out = (np.maximum(ax, az) <= outer)
    if chamfer_out:
        out &= (ax + az) <= (2 * outer - chamfer_out)
    inside = (np.maximum(ax, az) <= inner)
    if chamfer_in:
        inside &= (ax + az) <= (2 * inner - chamfer_in)
    return out & ~inside


def ring_sectors(cx: int, cz: int, inner: int, outer: int, lo: int, hi: int,
                 *, chamfer: int = 0, gap: int = 6, dmin: int = 24,
                 sector_max: int = 256) -> list:
    """The districts of one annulus: four strips, and the four corner sectors a
        chamfered ring has room for.

        The four strips are exactly what `placeplan.concentric_layout` has always cut, so a
        square ring tiles the way it always did and every registered number about a square
        place is unmoved. What is new is the **corners**: where the ring is chamfered the
        strips stop short of the cut, and the triangles between them held nothing. Each one
        takes the largest square that fits inside the chamfer, which is what the coverage
        bar was short of.

        Returns `[(label, (x0, z0, x1, z1))]`.
        
    """
    cc = int(chamfer or 0)
    strips = {
        "north": (cx - hi + cc, cz - hi, cx + hi - cc, cz - lo),
        "south": (cx - hi + cc, cz + lo, cx + hi - cc, cz + hi),
        "west": (cx - hi, cz - lo + gap, cx - lo, cz + lo - gap),
        "east": (cx + lo, cz - lo + gap, cx + hi, cz + lo - gap),
    }
    out = []
    for sname in ("north", "south", "west", "east"):
        x0, z0, x1, z1 = strips[sname]
        if x1 - x0 + 1 < dmin or z1 - z0 + 1 < dmin:
            continue
        along_x = (x1 - x0) >= (z1 - z0)
        length = (x1 - x0 + 1) if along_x else (z1 - z0 + 1)
        pieces = max(1, int(math.ceil(length / float(sector_max))))
        while pieces > 1 and (length - gap * (pieces - 1)) // pieces < dmin:
            pieces -= 1
        plen = (length - gap * (pieces - 1)) // pieces
        for i in range(pieces):
            start = (x0 if along_x else z0) + i * (plen + gap)
            end = start + plen - 1 if i < pieces - 1 else (x1 if along_x else z1)
            rect = ((start, z0, end, z1) if along_x else (x0, start, x1, end))
            if pieces == 1:
                label = sname
            elif pieces == 2:
                label = sname + ("_west" if along_x else "_north") if i == 0 \
                    else sname + ("_east" if along_x else "_south")
            else:
                label = f"{sname}_{i + 1}"
            out.append((label, rect))
    if cc:
        # **The corners of a round ring.** Each chamfer cuts a right triangle of legs
        # `cc` off the ring's outer corner; the ground between that cut, the two strips
        # and the ring's inner boundary is a quadrilateral, and the largest axis-aligned
        # rectangle inside it is what a district can use. Taken square, at the corner,
        # inset by the lane the strips keep.
        side = max(0, int((cc - gap) * 0.5))
        room = hi - lo - gap
        side = min(side, room)
        if side >= dmin:
            for label, (sx, sz) in (("north_west", (-1, -1)), ("north_east", (1, -1)),
                                    ("south_west", (-1, 1)), ("south_east", (1, 1))):
                # the corner square sits inside the chamfer line x+z = 2*hi - cc
                ax0 = cx + sx * (hi - cc - side) if sx < 0 else cx + sx * (hi - cc)
                az0 = cz + sz * (hi - cc - side) if sz < 0 else cz + sz * (hi - cc)
                x0 = min(ax0, ax0 + sx * side)
                z0 = min(az0, az0 + sz * side)
                out.append((f"corner_{label}", (int(x0), int(z0),
                                                int(x0 + side), int(z0 + side))))
    return out


# ----------------------------------------------------------- the shoreline case

#: How wide a band of land a shoreline settlement is laid out in, as a share of the
#: site. Registered before the shoreline case ran: a settlement that follows a shore is
#: a ribbon, and a ribbon that is a third of the site deep is a town that happens to be
#: near water.
SHORE_DEPTH = 0.34

#: How much water a site needs before `shore_anchor` will call an edge a shoreline.
#: Under this the "shore" is a pond and the anchor would be noise.
SHORE_MIN_WATER = 0.02


def water_mask(vol, x0: int, z0: int, S: int):
    """`(wet, height)` over the site's columns, off the cached volume.

        `observe.ground_heights` already computes exactly this for the router -- the top of
        the ground, or the water surface where that is higher, and a flag for which. Reading
        it here rather than re-deriving it is deliberate: a shoreline the layout believes in
        and the lanes do not is worse than no shoreline.
        
    """
    from . import observe
    h, wet = observe.ground_heights(vol)
    i, j = x0 - vol.x0, z0 - vol.z0
    return (np.asarray(wet[i:i + S, j:j + S], bool),
            np.asarray(h[i:i + S, j:j + S], int))


def shore_anchor(wet, x0: int, z0: int, *, step: int = 8) -> dict | None:
    """The shoreline: the land columns that touch water, as a path in world columns.

        **Derived from the ground and from nothing else.** No fixed straight line, no hidden
        centre: the land/water boundary is walked at `step` columns, the largest connected
        run is taken, and the path is what the districts are laid against and the buildings
        face.

        None where the site has too little water to have a shore (`SHORE_MIN_WATER`), which
        is a refusal a shoreline request has to be able to get.
        
    """
    wet = np.asarray(wet, bool)
    if wet.mean() < SHORE_MIN_WATER or (~wet).mean() < SHORE_MIN_WATER:
        return None
    land = ~wet
    touch = np.zeros_like(land)
    touch[:-1, :] |= land[:-1, :] & wet[1:, :]
    touch[1:, :] |= land[1:, :] & wet[:-1, :]
    touch[:, :-1] |= land[:, :-1] & wet[:, 1:]
    touch[:, 1:] |= land[:, 1:] & wet[:, :-1]
    # **The largest connected run, and not every wet column on the site.** Found by
    # running it on a real square: the site the search chose has a lake, a river and two
    # ponds, and taking the median of *all* the shore cells in each slice jumped between
    # them -- 292 columns between two neighbouring samples -- so the "shoreline" was a
    # line no water actually has. A shore is one body of water's edge.
    from scipy import ndimage
    labels, n = ndimage.label(touch, structure=np.ones((3, 3), bool))
    if n > 1:
        sizes = ndimage.sum(touch, labels, range(1, n + 1))
        touch = labels == (int(np.argmax(sizes)) + 1)
    xs, zs = np.nonzero(touch)
    if xs.size < 8:
        return None
    # The shore's own axis: the principal direction of the boundary cells. A coast
    # running north-south is walked in z and one running east-west in x, so the path is
    # a function of the long axis and never doubles back on itself.
    dx, dz = float(xs.std()), float(zs.std())
    along_x = dx >= dz
    key, other = (xs, zs) if along_x else (zs, xs)
    lo, hi = int(key.min()), int(key.max())
    pts = []
    for v in range(lo, hi + 1, max(1, step)):
        sel = other[(key >= v) & (key < v + max(1, step))]
        if sel.size:
            pts.append((v, int(round(float(np.median(sel))))))
    if len(pts) < 3:
        return None
    path = [[int(x0 + a), int(z0 + b)] if along_x else [int(x0 + b), int(z0 + a)]
            for a, b in pts]
    # which side of the path the water is on, so "facing the water" is a direction
    mid = pts[len(pts) // 2]
    mi, mj = (mid[0], mid[1]) if along_x else (mid[1], mid[0])
    probe = 6
    lo_wet = wet[max(0, mi - (0 if along_x else probe)):mi + 1,
                 max(0, mj - (probe if along_x else 0)):mj + 1].mean() if True else 0
    if along_x:
        below = wet[mi, max(0, mj - probe):mj].mean() if mj else 0.0
        above = wet[mi, mj + 1:mj + 1 + probe].mean()
        side = "north" if below > above else "south"
    else:
        below = wet[max(0, mi - probe):mi, mj].mean() if mi else 0.0
        above = wet[mi + 1:mi + 1 + probe, mj].mean()
        side = "west" if below > above else "east"
    return {"kind": "shoreline", "path": path, "along": "x" if along_x else "z",
            "water_side": side, "faces": "water",
            "water_share": round(float(wet.mean()), 4),
            "columns": int(xs.size),
            "note": "the land columns that touch water, walked along the shore's own "
                    "long axis; the layout is fitted to this and not to a line"}


def shore_band(anchor: dict, wet, x0: int, z0: int, depth: int):
    """The land within `depth` columns of the shore: where a shoreline place stands.

        A band and not a rectangle: the mask follows the path, so a bay is a bay and the
        ground behind a headland is not claimed.
        
    """
    wet = np.asarray(wet, bool)
    S = wet.shape
    m = np.zeros(S, bool)
    path = [(p[0] - x0, p[1] - z0) for p in anchor["path"]]
    along_x = anchor["along"] == "x"
    inland = {"north": 1, "south": -1, "west": 1, "east": -1}[anchor["water_side"]]
    for k, (a, b) in enumerate(path):
        nxt = path[min(k + 1, len(path) - 1)]
        span = range(min(a, nxt[0]), max(a, nxt[0]) + 1) if along_x else \
            range(min(b, nxt[1]), max(b, nxt[1]) + 1)
        for t in span:
            if along_x:
                i = t
                j0 = b if inland > 0 else b - depth
                sl = slice(max(0, j0), min(S[1], j0 + depth + 1))
                if 0 <= i < S[0]:
                    m[i, sl] = True
            else:
                j = t
                i0 = a if inland > 0 else a - depth
                sl = slice(max(0, i0), min(S[0], i0 + depth + 1))
                if 0 <= j < S[1]:
                    m[sl, j] = True
    return m & ~wet


def frontage_facing(anchor: dict | None) -> str | None:
    """Which way the buildings in a region face: the water where there is a shore."""
    return (anchor or {}).get("faces")
