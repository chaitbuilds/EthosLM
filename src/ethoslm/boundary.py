"""**Explicit boundary geometry**: the outline a place, a ring or a compound is enclosed by.

The design synthesis round. A boundary was a half-side and a word: `placeplan` drew
squares, and a "round" ring clipped its corners by `RING_CHAMFER`, so a round city read
square at city scale and left unowned wedges at the corners. Here the outline is a
record the design states and every consumer shares -- the wall's path, the gates on it,
the land each ring owns and the ring roads that follow it are all derived from one
`Outline`, so they cannot disagree.

    {"shape": "circle"}                                   radius = the extent's radius
    {"shape": "ellipse", "aspect": 0.8, "rotation": 20}   minor/major, degrees
    {"shape": "superellipse", "n": 3.0}                   n=2 circle, larger n squarer
    {"shape": "polygon", "points": [[1,0],[0.3,0.9],...], "round": 0.1}
                                                          unit coordinates about the
                                                          centre, star-shaped; `round`
                                                          fillets corners (share of R)

Bearings are compass degrees clockwise from north; north is -z, east +x. Every outline
is star-shaped about its centre and is sampled once as a polar table, which is what
`radius_at`, the concentric `scaled` rings, the wall polylines and the gate points use.
Land masks are rasters from the same table, and an inset by a distance is exact on the
raster (a Euclidean distance transform), so a ring's owned land and its wall's footprint
agree to the column.

A wall on the block lattice runs along x, along z or at 45 degrees (`Builder._decide_edge`).
`polyline` turns any outline into such a path: consecutive vertices sampled on the curve
every `step` blocks of arc, each pair joined by one diagonal and one axial run ordered to
bulge outward, so a circle of radius 500 reads as a smooth round wall within a block or two
of the true curve. A gate needs an axial run: `polyline(gates=...)` pins a straight run of
the gate's width centred on each gate bearing.
"""
from __future__ import annotations

import math

import numpy as np

#: samples of the polar table
POLAR_N = 2880
SHAPES = ("circle", "ellipse", "superellipse", "polygon")


def bearing_vec(bearing: float) -> tuple:
    """The unit (dx, dz) of a compass bearing: 0 north (-z), 90 east (+x)."""
    b = math.radians(float(bearing))
    return (math.sin(b), -math.cos(b))


def bearing_of(dx: float, dz: float) -> float:
    """The compass bearing, 0..360, of a direction."""
    return math.degrees(math.atan2(dx, -dz)) % 360.0


BEARINGS = {"north": 0.0, "north_east": 45.0, "east": 90.0, "south_east": 135.0,
            "south": 180.0, "south_west": 225.0, "west": 270.0, "north_west": 315.0}


def bearing(value) -> float:
    """A bearing from a number or a compass word."""
    if isinstance(value, (int, float)):
        return float(value) % 360.0
    v = str(value).strip().lower().replace("-", "_").replace(" ", "_")
    if v in BEARINGS:
        return BEARINGS[v]
    return float(v) % 360.0


class Outline:
    """A star-shaped closed outline about `centre`, as a polar table of radii."""

    def __init__(self, centre, radii: np.ndarray, record: dict | None = None):
        self.centre = (float(centre[0]), float(centre[1]))
        self.radii = np.asarray(radii, np.float64)
        self.record = dict(record or {})
        #: bearings a wall path must have a vertex at (a polygon's corners)
        self.corners: list = []

    # ----------------------------------------------------------- construction
    @classmethod
    def from_record(cls, rec: dict | None, centre, radius: float) -> "Outline":
        """The outline a design record states, at `radius` (the extent's half-size)."""
        rec = dict(rec or {"shape": "circle"})
        shape = str(rec.get("shape") or "circle")
        if shape not in SHAPES:
            raise ValueError(f"boundary shape {shape!r} is not one of {SHAPES}")
        th = np.linspace(0.0, 2 * math.pi, POLAR_N, endpoint=False)  # bearing, radians
        rot = math.radians(float(rec.get("rotation") or 0.0))
        R = float(radius)
        if shape == "circle":
            r = np.full(POLAR_N, R)
        elif shape in ("ellipse", "superellipse"):
            a = R
            b = R * float(rec.get("aspect") or 1.0)
            n = 2.0 if shape == "ellipse" else float(rec.get("n") or 2.0)
            t = th - rot
            # in the outline's own frame: u along the bearing-0 axis rotated, v across
            c, s = np.abs(np.cos(t)), np.abs(np.sin(t))
            r = 1.0 / np.power(np.power(c / b, n) + np.power(s / a, n), 1.0 / n)
        else:
            pts = [(float(p[0]), float(p[1])) for p in rec.get("points") or []]
            if len(pts) < 3:
                raise ValueError("a polygon boundary needs at least three points")
            r = _polygon_polar(pts, th, rot) * R
            fillet = float(rec.get("round") or 0.0)
            if fillet > 0:
                r = _fillet_polar(r, fillet * R)
            else:
                # the corners are vertices of every wall path drawn on this outline
                c, s_ = math.cos(rot), math.sin(rot)
                out = cls(centre, r, {**rec, "shape": shape, "radius": R})
                out.corners = [bearing_of(x * c - z * s_, x * s_ + z * c) for x, z in pts]
                return out
        return cls(centre, r, {**rec, "shape": shape, "radius": R})

    def scaled(self, f: float) -> "Outline":
        """The same outline at `f` of its size about the same centre (a concentric ring)."""
        o = Outline(self.centre, self.radii * float(f), {**self.record, "scaled": float(f)})
        o.corners = list(self.corners)
        return o

    def offset(self, d: float) -> "Outline":
        """The outline moved `d` blocks outward (negative inward), along its normal."""
        r = self.radii
        dr = np.gradient(np.concatenate([r[-2:], r, r[:2]]))[2:-2] / (2 * math.pi / POLAR_N)
        # radial distance that moves the curve by d along its normal
        cosang = r / np.sqrt(r * r + dr * dr)
        o = Outline(self.centre, np.maximum(1.0, r + d / np.maximum(0.2, cosang)),
                    {**self.record, "offset": float(d)})
        o.corners = list(self.corners)
        return o

    # ----------------------------------------------------------- queries
    def radius_at(self, bearing_deg) -> np.ndarray | float:
        b = np.asarray(bearing_deg, np.float64) % 360.0
        pos = b / 360.0 * POLAR_N
        i0 = np.floor(pos).astype(int) % POLAR_N
        i1 = (i0 + 1) % POLAR_N
        f = pos - np.floor(pos)
        out = self.radii[i0] * (1 - f) + self.radii[i1] * f
        return float(out) if np.ndim(out) == 0 else out

    def point_at(self, bearing_deg: float, frac: float = 1.0) -> tuple:
        """The point on (or at `frac` of the way out to) the outline at a bearing."""
        r = self.radius_at(bearing_deg) * float(frac)
        dx, dz = bearing_vec(bearing_deg)
        return (self.centre[0] + r * dx, self.centre[1] + r * dz)

    def extent(self) -> tuple:
        """The bounding box `(x0, z0, x1, z1)`, integer, inclusive."""
        th = np.linspace(0.0, 360.0, POLAR_N, endpoint=False)
        dx, dz = np.sin(np.radians(th)), -np.cos(np.radians(th))
        xs = self.centre[0] + self.radii * dx
        zs = self.centre[1] + self.radii * dz
        return (int(math.floor(xs.min())), int(math.floor(zs.min())),
                int(math.ceil(xs.max())), int(math.ceil(zs.max())))

    def area(self) -> float:
        """Enclosed area in columns (polar integral)."""
        return float(0.5 * np.sum(self.radii ** 2) * (2 * math.pi / POLAR_N))

    def perimeter(self) -> float:
        th = np.linspace(0.0, 2 * math.pi, POLAR_N + 1)
        r = np.concatenate([self.radii, self.radii[:1]])
        x, z = r * np.sin(th), -r * np.cos(th)
        return float(np.sum(np.hypot(np.diff(x), np.diff(z))))

    def ratio(self, x, z) -> np.ndarray:
        """|p - c| / r(bearing of p): < 1 inside, 1 on, > 1 outside. Vectorised."""
        dx = np.asarray(x, np.float64) - self.centre[0]
        dz = np.asarray(z, np.float64) - self.centre[1]
        b = np.degrees(np.arctan2(dx, -dz)) % 360.0
        return np.hypot(dx, dz) / np.maximum(1e-6, self.radius_at(b))

    def mask(self, x0: int, z0: int, w: int, h: int) -> np.ndarray:
        """`[w, h]` bool, indexed `[x - x0, z - z0]`: columns whose centre is inside."""
        xs = np.arange(x0, x0 + w, dtype=np.float64)[:, None] + 0.5
        zs = np.arange(z0, z0 + h, dtype=np.float64)[None, :] + 0.5
        return self.ratio(xs, zs) <= 1.0

    def signed_distance(self, x0: int, z0: int, w: int, h: int) -> np.ndarray:
        """Approximate signed distance in blocks, negative inside, `[w, h]`: exact on
        the raster (Euclidean distance transform of the mask and of its complement)."""
        from scipy import ndimage
        m = self.mask(x0, z0, w, h)
        inside = ndimage.distance_transform_edt(m)
        outside = ndimage.distance_transform_edt(~m)
        return np.where(m, -inside + 0.5, outside - 0.5)

    # ----------------------------------------------------------- the wall's path
    def polyline(self, *, step: float = 18.0, gates=(), gate_run: int = 13,
                 frac: float = 1.0) -> list:
        """A closed vertex list `[(x, z), ...]` (first == last) following the outline
                (at `frac` of its size) in runs along x, along z or at 45 degrees only.

                Vertices are integer points on the curve about every `step` blocks of arc; each
                consecutive pair is joined by a diagonal and an axial run whose order bulges
                outward. `gates` are bearings: at each a straight axial run of `gate_run` blocks
                is pinned, centred on the gate, so a gate type stands on a straight wall.

        """
        rr = self.radii * float(frac)
        cx, cz = self.centre
        per = float(np.sum(np.hypot(*_polar_xy(rr))))
        n = max(8, int(round(per / max(4.0, step))))
        bearings = list(np.linspace(0.0, 360.0, n, endpoint=False))
        gates = sorted(float(g) % 360.0 for g in (gates or ()))
        # drop sample bearings too close to a gate span; the span supplies its own
        spans = []
        for g in gates:
            r_g = float(np.interp(g, np.linspace(0, 360, POLAR_N, endpoint=False), rr,
                                  period=360))
            half_deg = math.degrees((gate_run / 2 + 2) / max(1.0, r_g))
            spans.append((g, half_deg))
        bearings = sorted(set(bearings) | {float(c) % 360.0 for c in self.corners})
        keep = [b for b in bearings
                if all(_ang(b, g) > h + math.degrees(step / 2 / max(1.0, float(
                    self.radius_at(b) * frac))) for g, h in spans)]
        pts = []            # (bearing, (x, z), pinned-gate-id or None)
        for b in keep:
            r = float(self.radius_at(b)) * frac
            dx, dz = bearing_vec(b)
            pts.append((b, (int(round(cx + r * dx)), int(round(cz + r * dz))), None))
        for gi, (g, _h) in enumerate(spans):
            r = float(self.radius_at(g)) * frac
            dx, dz = bearing_vec(g)
            gx, gz = cx + r * dx, cz + r * dz
            half = gate_run // 2
            # the gate's wall runs along the axis nearer the tangent
            tx, tz = -dz, dx
            if abs(tx) >= abs(tz):          # wall runs along x at this gate
                a = (int(round(gx)) - half, int(round(gz)))
                c = (int(round(gx)) + half, int(round(gz)))
            else:
                a = (int(round(gx)), int(round(gz)) - half)
                c = (int(round(gx)), int(round(gz)) + half)
            ba, bc = bearing_of(a[0] - cx, a[1] - cz), bearing_of(c[0] - cx, c[1] - cz)
            pts.append((ba, a, gi))
            pts.append((bc, c, gi))
        pts.sort(key=lambda t: t[0])
        # de-duplicate coincident vertices
        verts, pins = [], []
        for b, p, gi in pts:
            if verts and verts[-1] == p:
                continue
            verts.append(p)
            pins.append(gi)
        if verts[0] == verts[-1]:
            verts.pop()
            pins.pop()
        out = [verts[0]]
        m = len(verts)
        # a one-block wobble between neighbouring samples is rounding, not the curve:
        # the next vertex is snapped onto the run (never a pinned gate vertex)
        for i in range(m - 1):
            a, c = verts[i], verts[i + 1]
            dx, dz = c[0] - a[0], c[1] - a[1]
            if pins[i + 1] is None and min(abs(dx), abs(dz)) == 1 and \
                    max(abs(dx), abs(dz)) >= 4:
                verts[i + 1] = (c[0], a[1]) if abs(dx) > abs(dz) else (a[0], c[1])
        for i in range(m):
            a, c = verts[i], verts[(i + 1) % m]
            if pins[i] is not None and pins[i] == pins[(i + 1) % m]:
                out.append(c)                 # the pinned straight gate run
                continue
            out.extend(_octilinear(a, c, (cx, cz))[1:])
        # collapse collinear consecutive runs
        return _collinear_merge(out)

    def gate_points(self, gates, frac: float = 1.0) -> list:
        """For each bearing: `{"bearing", "x", "z", "facing", "along"}` -- the gate cell
        on the wall's pinned run (see `polyline`) and the outward facing."""
        cx, cz = self.centre
        out = []
        for g in gates:
            g = bearing(g)
            r = float(self.radius_at(g)) * frac
            dx, dz = bearing_vec(g)
            gx, gz = int(round(cx + r * dx)), int(round(cz + r * dz))
            tx, tz = -dz, dx
            along = "x" if abs(tx) >= abs(tz) else "z"
            if along == "x":
                facing = "south" if dz > 0 else "north"
            else:
                facing = "east" if dx > 0 else "west"
            out.append({"bearing": g, "x": gx, "z": gz, "facing": facing, "along": along})
        return out

    def to_json(self) -> dict:
        return {**self.record, "centre": [round(self.centre[0], 2), round(self.centre[1], 2)],
                "area": round(self.area()), "perimeter": round(self.perimeter()),
                "extent": list(self.extent())}


# ----------------------------------------------------------------- helpers

def _polar_xy(r: np.ndarray) -> tuple:
    th = np.linspace(0.0, 2 * math.pi, len(r), endpoint=False)
    x, z = r * np.sin(th), -r * np.cos(th)
    return np.diff(np.concatenate([x, x[:1]])), np.diff(np.concatenate([z, z[:1]]))


def _ang(a: float, b: float) -> float:
    d = abs((a - b) % 360.0)
    return min(d, 360.0 - d)


def _polygon_polar(pts: list, th: np.ndarray, rot: float) -> np.ndarray:
    """Radius along each bearing to a star-shaped polygon given in unit coordinates
    (x east, z south), rotated by `rot` radians clockwise."""
    c, s = math.cos(rot), math.sin(rot)
    P = [(x * c - z * s, x * s + z * c) for x, z in pts]
    dx, dz = np.sin(th), -np.cos(th)
    best = np.full(len(th), np.inf)
    for (ax, az), (bx, bz) in zip(P, P[1:] + P[:1]):
        # solve t*d = a + u*(b - a), t > 0, 0 <= u <= 1
        ex, ez = bx - ax, bz - az
        den = dx * ez - dz * ex
        with np.errstate(divide="ignore", invalid="ignore"):
            t = (ax * ez - az * ex) / den
            u = (ax * dz - az * dx) / den
        ok = (np.abs(den) > 1e-12) & (t > 0) & (u >= -1e-9) & (u <= 1 + 1e-9)
        best = np.where(ok & (t < best), t, best)
    if not np.all(np.isfinite(best)):
        raise ValueError("the polygon is not star-shaped about its centre")
    return best


def _fillet_polar(r: np.ndarray, rad: float) -> np.ndarray:
    """Round a polar outline's corners: erode then dilate its raster by `rad`."""
    from scipy import ndimage
    R = float(r.max())
    size = int(2 * R + 8)
    o = Outline((size / 2, size / 2), r)
    m = o.mask(0, 0, size, size)
    inside = ndimage.distance_transform_edt(m)
    er = inside > rad
    grown = ndimage.distance_transform_edt(~er) <= rad
    # read the radii back off the rounded raster
    th = np.linspace(0.0, 2 * math.pi, len(r), endpoint=False)
    out = np.empty_like(r)
    for i, t in enumerate(th):
        dx, dz = math.sin(t), -math.cos(t)
        lo, hi = 0.0, R + 2
        for _ in range(24):
            mid = (lo + hi) / 2
            x, z = int(size / 2 + mid * dx), int(size / 2 + mid * dz)
            if 0 <= x < size and 0 <= z < size and grown[x, z]:
                lo = mid
            else:
                hi = mid
        out[i] = lo
    return out


def _octilinear(a: tuple, c: tuple, centre: tuple) -> list:
    """`[a, (bend), c]`: a to c by one 45-degree run and one axial run, ordered so the
    bend lies farther from `centre` (the outline is convex-ish; the path hugs it)."""
    (ax, az), (cx_, cz_) = a, c
    dx, dz = cx_ - ax, cz_ - az
    if dx == 0 or dz == 0 or abs(dx) == abs(dz):
        return [a, c]
    d = min(abs(dx), abs(dz))
    sx, sz = (1 if dx > 0 else -1), (1 if dz > 0 else -1)
    diag = (sx * d, sz * d)
    # bend after the diagonal, or after the axial run
    b1 = (ax + diag[0], az + diag[1])
    b2 = (cx_ - diag[0], cz_ - diag[1])
    ox, oz = centre
    f1 = math.hypot(b1[0] - ox, b1[1] - oz)
    f2 = math.hypot(b2[0] - ox, b2[1] - oz)
    return [a, b1 if f1 >= f2 else b2, c]


def _collinear_merge(path: list) -> list:
    """Drop vertices between two runs in the same direction."""
    def unit(p, q):
        dx, dz = q[0] - p[0], q[1] - p[1]
        return ((dx > 0) - (dx < 0), (dz > 0) - (dz < 0))
    out = [path[0]]
    for i in range(1, len(path) - 1):
        if unit(out[-1], path[i]) == unit(path[i], path[i + 1]):
            continue
        out.append(path[i])
    out.append(path[-1])
    if out[0] != out[-1]:
        out.append(out[0])
    return out


def polyline_cells(path: list) -> list:
    """The lattice columns of a polyline, in order (axial or 45-degree runs only)."""
    cells = []
    for (ax, az), (bx, bz) in zip(path, path[1:]):
        n = max(abs(bx - ax), abs(bz - az))
        sx, sz = (bx > ax) - (bx < ax), (bz > az) - (bz < az)
        for i in range(n):
            cells.append((ax + sx * i, az + sz * i))
    if path:
        cells.append(tuple(path[-1]))
    return cells


def check_polyline(path: list) -> list:
    """Every segment that is neither axial nor 45 degrees, as `[(a, b), ...]`."""
    bad = []
    for a, b in zip(path, path[1:]):
        dx, dz = b[0] - a[0], b[1] - a[1]
        if dx and dz and abs(dx) != abs(dz):
            bad.append((a, b))
    return bad
