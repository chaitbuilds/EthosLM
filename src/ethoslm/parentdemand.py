"""**What a neighbourhood's forms need of the parent that cuts its land.** The parent
composition round.

The design resolution round made one form plan per type the answer every layer reads
(`ethoslm.formplan`), and stopped at the lot: the parent still spread a strip's count over
its pieces by feasible columns and cut the strip where its ground broke, and the forms
entered only when each piece was compiled. A sum of lot areas says nothing about whether
the arrangement those lots need -- a row of shops, the clearance behind it, two rows of
courtyard houses facing each other across their lane -- fits the ground it was given. The
design resolution section showed the consequence: a lane piece 76 wide on a strip 51
deep holds its houses back to back with a lane on each outer edge, whatever the count.

This module turns the adopted forms into the **section and module** each street
arrangement needs, in the composer's own units (`streetplan`: the lane, the clearance
between rows, the edge margin, the principal street's verge), so the parent can:

  * say which arrangements fit a strip and, for those that do not, what is short and
    which parent decision would supply it (`sections`);
  * cut pieces that hold **whole modules** of the arrangement that fits, give the ground
    left over an owner as landscape rather than residue, and state the programme each
    piece holds by those modules (`module_pieces`);
  * hand the adopted arrangement to the compiler, which composes it and reports when it
    could not (`district_compile` reads `sector.module`).

Nothing here names a place. The numbers are the forms' (`formplan.least_lot`) and the
composer's (`streetplan`); a strip is a rectangle, its ground a per-slice profile.
"""
from __future__ import annotations

from . import formplan

#: The street arrangements the parent can ask a strip for, by name. `across` -- lanes
#: run across the strip from its principal street, two rows of houses facing each other
#: on each lane (the lane grain the composer calls `lanes along z` with `lanes: between`
#: on an east-west strip); `along` -- one lane runs along the strip behind the principal
#: street's row, with a row of houses facing it from each side.
ARRANGEMENTS = ("across", "along")


def forms(ch: dict, house_type: str | None, shop_type: str | None = "shop_house",
          *, inset: int = 1) -> dict | None:
    """The lots the adopted forms need, asked of the types' own plans exactly as the
    compiler asks them (`district_compile`, the street composition): the dwelling's
    least lot between two party walls and at a row's free end, and the trade form's
    least depth. None where the dwelling publishes no form plan."""
    if not house_type or formplan.plan_fn(house_type) is None:
        return None
    fs = {u: dict(v) for u, v in ((ch or {}).get("forms") or {}).items()}
    dw = dict(fs.get("dwelling") or {})
    if (ch or {}).get("court_least") and "court" not in dw:
        dw["court"] = int(ch["court_least"])
    lo, hi = formplan.FLANKS["north"]
    mid = formplan.least_lot(house_type, dw, attached=(lo, hi), inset=inset) or {}
    end = formplan.least_lot(house_type, dw, attached=(lo,), inset=inset) or {}
    if not mid.get("lot") or not end.get("lot"):
        return None
    out = {"house": {"type": house_type, "params": dw, "width": int(mid["lot"][0]),
                     "depth": int(mid["lot"][1]),
                     "end_extra": max(0, int(end["lot"][0]) - int(mid["lot"][0])),
                     "plan": (mid.get("plan") or {}).get("why")},
           "shop": None}
    tr = dict(fs.get("trade") or {})
    if shop_type and formplan.plan_fn(shop_type) is not None and tr:
        for sd in (12, 13, 11, 14):
            ws = [w for w in range(7, 11)
                  if formplan.admits(shop_type, w, sd, tr, configs_=[(lo, hi)])]
            if ws:
                out["shop"] = {"type": shop_type, "params": tr, "width": int(ws[0]),
                               "depth": int(sd)}
                break
    return out


def _consts():
    from . import streetplan as sp
    from .district_compile import EDGE_MARGIN, LANDMARK_AREA_SIDE
    return {"lane": sp.LANE, "clear": sp.CLEAR, "margin": EDGE_MARGIN,
            "verge": 1, "walk": sp.MARKET_WALK, "market": LANDMARK_AREA_SIDE}


def row_run(f: dict, n: int) -> int:
    """The length a run of `n` houses along one street needs: `n` lots between party
    walls and the inset of the run's two free ends."""
    if n <= 0:
        return 0
    return n * int(f["house"]["width"]) + 2 * int(f["house"]["end_extra"])


def across_width(f: dict, rows: int) -> int:
    """How wide a piece must be (along the strip) to hold `rows` rows of houses in the
    `across` arrangement, as the composer sweeps it with `lanes: between`: a row backing
    onto the piece's edge, its lane, the row facing it; then, for each further pair, the
    clearance behind, a row and its lane, and the row facing it. An odd last row faces a
    lane laid along the far edge. Edge margins included."""
    c = _consts()
    D, L, C = int(f["house"]["depth"]), c["lane"], c["clear"]
    if rows <= 0:
        return 0
    w = D + L + (D if rows >= 2 else 0)
    k = 2
    while k < rows:
        w += C + D + L
        k += 1
        if k < rows:
            w += D
            k += 1
    return w + 2 * c["margin"]


def sections(f: dict, depth: int) -> dict:
    """**Which arrangements a strip `depth` deep can hold, and what each is short of.**

        `depth` is the strip's buildable depth across it: from the first column past the
        principal street's verge to the last before the edge margin. For each arrangement the
        section it needs (a shop row where the trade form is owed, the clearance behind it,
        the rows of houses and their lane), how many houses deep one module is, and -- where
        it does not fit -- by how much and which parent decision would supply that depth.

    """
    c = _consts()
    D, W = int(f["house"]["depth"]), int(f["house"]["width"])
    e = int(f["house"]["end_extra"])
    shop_d = int((f.get("shop") or {}).get("depth") or 0)
    front = (shop_d + c["clear"]) if shop_d else 0
    out = {}
    # across: rows of houses stand along the lanes, so the strip's depth holds as many
    # houses along each lane as the house's width divides it into
    left = depth - front
    per = 0
    while row_run(f, per + 1) <= left:
        per += 1
    out["across"] = {
        "section": ([f"shop row {shop_d}", f"clearance {c['clear']}"] if shop_d else [])
        + [f"houses {W} wide along the lane x {per}"],
        "needs": front + row_run(f, 1), "has": int(depth),
        "fits": per >= 1, "houses_per_row": int(per),
        "module": {"rows_2": across_width(f, 2), "rows_3": across_width(f, 3),
                   "rows_4": across_width(f, 4)},
        "why": (f"{per} house(s) of {W} (+{e} at each free end) along each lane in the "
                f"{left} columns behind the "
                + (f"shop row ({shop_d} + {c['clear']} clear)" if shop_d else "street")
                + "; each lane serves the two rows facing each other across it")}
    # along: shops (or houses) on the principal street, then a row facing a lane along
    # the strip and a row facing it from the far side
    need = front + D + c["lane"] + D
    short = max(0, need - depth)
    out["along"] = {
        "section": ([f"shop row {shop_d}", f"clearance {c['clear']}"] if shop_d else [])
        + [f"houses {D} deep", f"lane {c['lane']}", f"houses {D} deep"],
        "needs": int(need), "has": int(depth), "fits": short == 0, "short": int(short),
        "houses_per_row": None,
        "why": (f"two rows of houses {D} deep facing each other across a lane along the "
                f"strip need {need} columns across it"
                + (f"; this strip has {depth}: {short} short" if short else "")),
        **({"supply": (f"{short} more columns across the strip: the principal street's "
                       f"line or the strip's inner inset, which are the ring's and its "
                       f"walls' clearances to give, or a shallower house form")}
           if short else {})}
    return out


def buildable_depth(rect, road_line: int | None, *, along_x: bool, side: str) -> int:
    """The strip's buildable depth across it: from past the principal street's verge (or
    the edge margin where no street runs along it) to the far edge margin."""
    c = _consts()
    x0, z0, x1, z1 = rect
    lo, hi = (z0, z1) if along_x else (x0, x1)
    if road_line is not None:
        if side in ("north", "west"):
            lo = max(lo + c["margin"], int(road_line) + 1 + c["verge"])
        else:
            hi = min(hi - c["margin"], int(road_line) - 1 - c["verge"])
    else:
        lo += c["margin"]
    hi2 = hi - c["margin"] if road_line is None or side in ("north", "west") else hi
    return int(hi2 - lo + 1)


def market_zone(f: dict) -> int:
    """The length along the strip the arrival's market takes with the fronts that face
    it: the market floor, its walk and a shop row on its far side, plus edge margins."""
    c = _consts()
    shop_d = int((f.get("shop") or {}).get("depth") or 0)
    return c["market"] + c["walk"] + max(shop_d, 1) + 2 * c["margin"]


def whole_runs(prof_row, across: int, bar: float = 0.97) -> list:
    """Runs of slices whose ground carries a building across the whole strip (at least
    `bar` of `across` feasible), as `[(a, b)]` inclusive -- where a row of houses whose
    depth runs across the strip can stand."""
    runs, cur = [], None
    for i, v in enumerate(prof_row):
        if float(v) >= bar * across:
            cur = [i, i] if cur is None else [cur[0], i]
        else:
            if cur:
                runs.append(tuple(cur))
            cur = None
    if cur:
        runs.append(tuple(cur))
    return runs




def refit(alt: dict, districts: list, rects: list, profs: dict, f: dict, *, depth: int,
          ref: int, gap: int, along_x: bool, access=None, open_share: float = 0.2,
          negotiate_share: float = 0.7, step: int = 4) -> dict | None:
    """**A strip's cut, revised by what its neighbourhood's modules need.**

        `alt` is an arrangement the ground negotiation measured (`placeplan.negotiate_strip`:
        pieces cut where the ground breaks, each at the level its own feasible columns
        prefer). The ground alone decides each piece's level, so the arrival -- the founded
        piece nearest the ring's gate, which holds the market -- and the homes beside it are
        levelled apart, and nothing asks whether the lanes' rows still fit at those levels.

        Here the pieces beside the arrival are asked, at each offered level, how many rows of
        houses in the `across` arrangement their whole ground holds (`across_width`); the
        arrival is asked at which levels its market zone (`market_zone`) stays whole from the
        end it is entered at. Among the pairs of levels that keep **every module the ground
        admits anywhere**, the pair with the least step between the arrival and its homes is
        taken, and a homes piece whose run cannot hold its module where it stands is moved
        along its run to where it can. The programme follows the modules: a homes piece is
        asked for its rows' houses and the shops its principal street's run holds
        (`programme`), not a share of the strip's count spread by feasible columns.

        Returns the revised alternative (`modules` says what moved and why), or None where
        nothing changed.

    """
    per = int((sections(f, depth).get("across") or {}).get("houses_per_row") or 0)
    if per < 1:
        return None
    c = _consts()
    shop = f.get("shop") or {}
    pieces = [dict(p) for p in alt["pieces"]]
    if access is None:
        return None
    ax, az = int(access[0]), int(access[-1])

    def _dist(p):
        x0, z0, x1, z1 = p["rect"]
        return max(x0 - ax, 0, ax - x1) + max(z0 - az, 0, az - z1)
    live = [i for i, p in enumerate(pieces)
            if not p.get("open") and float(p.get("share") or 0) >= negotiate_share]
    if not live:
        # no founded piece is entered from the gate: the modules have no neighbourhood
        # to be continuous with, and the ground cut stands
        return None
    arrival = min(live, key=lambda i: _dist(pieces[i]))
    src = pieces[arrival]["from"]
    k = next((j for j, d in enumerate(districts) if d["name"] == src), None)
    if k is None:
        return None
    prof, r = profs[k], rects[k]
    base = r[0] if along_x else r[1]
    across = (r[3] - r[1] + 1) if along_x else (r[2] - r[0] + 1)

    def span(p):
        x0, z0, x1, z1 = p["rect"]
        return ((x0 if along_x else z0) - base, (x1 if along_x else z1) - base)
    # the homes: the carrying pieces of the arrival's source beside it, in order away
    # from the gate
    a_lo, a_hi = span(pieces[arrival])
    x0a, z0a, x1a, z1a = pieces[arrival]["rect"]
    entered_lo = (abs(x0a - ax) <= abs(x1a - ax)) if along_x else (abs(z0a - az) <= abs(z1a - az))
    homes = [i for i, p in enumerate(pieces)
             if i != arrival and p["from"] == src and not p.get("open")
             and float(p.get("share") or 0) >= negotiate_share]
    if not homes:
        return None
    zone = market_zone(f)

    def arrival_levels():
        """Levels at which the market zone is whole from the arrival's entered end."""
        got = []
        for L, row in prof.items():
            for (ra, rb) in whole_runs(row, across, bar=0.9):
                ra, rb = max(ra, a_lo), min(rb, a_hi)
                # whole from within a lot's inset of the end it is entered at: the
                # market may stand a few columns in from the gate street's verge
                reach = int(f["house"]["width"]) // 2
                near = (ra - a_lo <= reach) if entered_lo else (a_hi - rb <= reach)
                if near and rb - ra + 1 >= zone:
                    got.append(int(L))
        return sorted(set(got))

    def modules_at(i, L):
        """`(rows, a, b)`: the most rows the piece's run holds at level L, and where."""
        lo, hi = span(pieces[i])
        best = (0, lo, hi)
        for (ra, rb) in whole_runs(prof[L], across):
            ra, rb = max(ra, lo), min(rb, hi)
            if rb - ra + 1 < across_width(f, 2):
                continue
            rows = 2
            while across_width(f, rows + 1) <= rb - ra + 1:
                rows += 1
            if rows > best[0]:
                best = (rows, ra, rb)
        return best
    La_opts = arrival_levels()
    if not La_opts:
        return None
    # every module the ground admits for each homes piece, at any offered level
    most = {i: max(modules_at(i, L)[0] for L in prof) for i in homes}
    if not any(most.values()):
        return None
    choice = None
    for La in La_opts:
        lv, worst = {}, 0
        for i in homes:
            opts = [L for L in prof if modules_at(i, L)[0] == most[i]]
            if not opts:
                continue
            Lh = min(opts, key=lambda L: (abs(L - La), abs(L - ref)))
            lv[i] = int(Lh)
            beside = (abs(span(pieces[i])[0] - a_hi) <= gap + 1
                      or abs(a_lo - span(pieces[i])[1]) <= gap + 1)
            if beside:
                worst = max(worst, abs(int(Lh) - int(La)))
        key = (worst, abs(La - int(pieces[arrival]["level"])))
        if choice is None or key < choice[0]:
            choice = (key, La, lv)
    (worst, _d), La, lv = choice
    out = [dict(p) for p in pieces]
    why = []
    if La != pieces[arrival]["level"]:
        why.append(f"arrival {pieces[arrival]['rect']} {pieces[arrival]['level']} -> {La}: "
                   f"the level nearest its homes at which its market zone ({zone} "
                   f"columns) stays whole from the gate")
    out[arrival].update(level=int(La), module={"role": "arrival", "arrangement": "anchor",
                                                "zone": int(zone)})
    for i, Lh in lv.items():
        rows, ra, rb = modules_at(i, Lh)
        p = out[i]
        run = (span(p)[1] - span(p)[0] + 1) - 2 * c["margin"]
        shops = (run // int(shop["width"])) if shop else 0
        was = p["level"]
        p.update(level=int(Lh),
                 feasible_columns=int(sum(prof[Lh][span(p)[0]:span(p)[1] + 1])),
                 module={"role": "homes", "arrangement": "across", "rows": int(rows),
                         "houses": int(rows * per), "shops": int(shops),
                         "run": [int(base + ra), int(base + rb)],
                         "why": (f"{rows} row(s) of {per} house(s) facing each other across "
                                 f"their lane(s) on the whole ground "
                                 f"{base + ra}..{base + rb} at {Lh} "
                                 f"(`across` module {across_width(f, rows)} columns); "
                                 f"{abs(Lh - La)} from the arrival at {La}")},
                 programme=int(rows * per + shops))
        p["share"] = round(p["feasible_columns"] / float(max(1, p["columns"])), 4)
        why.append(f"homes {p['rect']} {was} -> {Lh}: {rows} row(s) of {per}, the most "
                   f"its ground holds at any offered level, at the least step "
                   f"({abs(Lh - La)}) to the arrival")
    founded = sum(q["feasible_columns"] for q in out if not q.get("open"))
    usable = sum(q["feasible_columns"] * q["share"] for q in out if not q.get("open"))
    return {"arrangement": f"{alt['arrangement']}+modules", "pieces": out,
            "cuts": int(alt.get("cuts") or 0),
            "founded_columns": int(founded), "usable_columns": int(usable),
            "score": int(usable), "least_share": min(q["share"] for q in out),
            "modules": why or ["levels stood; the programme follows the modules"],
            "step": int(worst), "depth": int(depth)}


def strip_depth(rects: list, routes, *, along_x: bool) -> tuple:
    """`(depth, street_side)`: the buildable depth across a strip of pieces `rects` and
    the side its principal street runs along -- the band of road cells running the
    strip's length, where there is one -- in the compiler's own margins."""
    c = _consts()
    x0 = min(r[0] for r in rects)
    z0 = min(r[1] for r in rects)
    x1 = max(r[2] for r in rects)
    z1 = max(r[3] for r in rects)
    length = (x1 - x0 + 1) if along_x else (z1 - z0 + 1)
    road = {(int(a), int(b)) for (a, b) in (routes or ())}
    lo, hi = (z0, z1) if along_x else (x0, x1)
    rows = []
    for v in range(lo, hi + 1):
        n = sum(1 for u in range(x0 if along_x else z0, (x1 if along_x else z1) + 1)
                if ((u, v) if along_x else (v, u)) in road)
        if n >= 0.5 * length:
            rows.append(v)
    if not rows:
        return (hi - lo + 1 - 2 * c["margin"], None)
    if min(rows) - lo <= hi - max(rows):
        side = "north" if along_x else "west"
        return (hi - c["margin"] - (max(rows) + 1 + c["verge"]) + 1, side)
    side = "south" if along_x else "east"
    return ((min(rows) - 1 - c["verge"]) - (lo + c["margin"]) + 1, side)


#: The fewest houses a residential lane must hold on each side to be a lane of a quarter
#: rather than a court entrance: a lane holding one or two houses a side is what the
#: parent composition round delivered on a strip 51 deep (four houses on one lane), and
#: its reader called it a hamlet's fabric beside a market.
LANE_LEAST = 3

#: Where a ring's principal street runs across its strips, and so how many sides of it
#: the section is laid on: `edge` -- along the strip's inner edge, the neighbourhood's
#: section behind it on one side; `middle` -- down the strip's middle (where the
#: arterial joins district centres), a section on each side.
STREET_PLACES = ("edge", "middle")


def ring_section(f: dict, *, road: int, lane_least: int = LANE_LEAST,
                 place: str | None = None) -> dict:
    """**The depth across a ring strip that its complete neighbourhood needs.**

        The city attempt round. `sections` names what one strip of a given depth holds and
        what it is short of, but the parent that set the depth -- the ring's width in
        `placeplan.concentric_layout` -- never read it: a ring's width came from its share
        of the site, and the section's shortfall was a note. This turns the same form plans
        and composer units into the **least district depth** at which every strip of the
        ring holds the owed relationship: courtyard houses facing each other across a
        residential lane of at least `lane_least` houses a side, behind the principal
        street's own frontage.

        For each street arrangement (`along`, `across`) the depth one side of the principal
        street needs; for each street place (`STREET_PLACES`) the district depth that gives:
        the principal street's band (`road` columns, dilated a column), its verge and the
        edge margins. The chosen proposal is the shallowest; every other is recorded with
        its depth, so the choice is a comparison and not a constant. `place` pins the
        street place where a caller has decided it.

    """
    c = _consts()
    D, W = int(f["house"]["depth"]), int(f["house"]["width"])
    shop_d = int((f.get("shop") or {}).get("depth") or 0)
    front = (shop_d + c["clear"]) if shop_d else 0
    sides = {
        # a lane along the strip: the street's frontage, its clearance, a row backing
        # onto it facing the lane, and the row facing that across it
        "along": {"per_side": int(front + D + c["lane"] + D),
                  "lane_houses": None,
                  "why": f"frontage {shop_d or '-'} + clear {c['clear'] if shop_d else 0}"
                         f" + houses {D} + lane {c['lane']} + houses {D}"},
        # lanes across the strip from the street, `lane_least` houses along each side
        "across": {"per_side": int(front + row_run(f, lane_least)),
                   "lane_houses": int(lane_least),
                   "why": f"frontage {shop_d or '-'} + clear {c['clear'] if shop_d else 0}"
                          f" + {lane_least} houses {W} wide along each lane"
                          f" (+{f['house']['end_extra']} at each free end)"},
    }
    band = int(road) + 2                          # the arterial, dilated a column
    proposals = []
    for where in STREET_PLACES:
        if place and where != place:
            continue
        n_sides = 1 if where == "edge" else 2
        for arr, s in sides.items():
            one = s["per_side"] + c["verge"] + c["margin"]
            depth = band + n_sides * one + (c["margin"] if where == "edge" else 0)
            proposals.append({"street": where, "arrangement": arr, "sides": n_sides,
                              "per_side": s["per_side"], "depth": int(depth),
                              "why": (f"{n_sides} side(s) of the principal street "
                                      f"({band} with its dilation), each {s['why']}, "
                                      f"its verge {c['verge']} and margin {c['margin']}")})
    proposals.sort(key=lambda p: (p["depth"], p["street"] != "edge", p["arrangement"]))
    chosen = proposals[0] if proposals else None
    return {"chosen": chosen, "proposals": proposals, "lane_least": int(lane_least),
            "house": [W, D], "shop_depth": shop_d,
            "owes": ("courtyard houses facing each other across a residential lane of at "
                     f"least {lane_least} houses a side, behind the principal street's "
                     "frontage")}
