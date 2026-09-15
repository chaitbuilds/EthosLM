"""Frontage as a precondition, not a postmortem.

    check_door(x, y, z) -> can a person walk to this doorway from the lane, without
                           jumping, given everything I have decided so far?

The "given everything I have decided so far" is the part that makes it a precondition.
A model program's blocks are pending until the pass flushes, so a check that reads only
the world would judge the wall the pass built last time. The pending writes are overlaid
onto a box cut out of the pass's own volume, and the walk model runs on that.

Cost, measured on the settlement site: the volume is decoded once per pass (0.03 s) and
each check is a 25x25 slice, an overlay, a Nav build and a 0-jump flood -- a few
milliseconds. Cheap enough that a program can call it for every door it places, which is
the only version of this that is worth having. If a pass still learns about frontage
from a critique afterwards, the ordering change bought nothing."""
from __future__ import annotations

from . import observe


class Frontage:
    """The circulation network, as a question a running build pass can ask."""

    def __init__(self, vol: observe.Volume, network, radius: int = 12):
        self.vol = vol
        self.net = network
        self.radius = radius
        self.calls = {"check_door": 0, "nearest_lane": 0, "threshold": 0}
        self.results: list = []
        #: label -> what `site()` did to that structure's doorstep. The circulation pass
        #: levelled a doorstep on the ground as it found it; siting then raises that
        #: ground to a pad, and from that moment the doorstep is on the pad. The
        #: network's own record is left alone -- where the door goes is the circulation
        #: pass's decision and this is not entitled to overrule it -- but what
        #: `floor_from_threshold` answers has to be where the floor now is, or every
        #: building put on a pad has its ground storey buried in it.
        self.sited: dict = {}

    # --- what a model program sees ---------------------------------------
    def threshold(self, label: str) -> dict | None:
        """The threshold the circulation pass reserved for this structure.

                Position, and which way you face walking in off the lane. Put the door there.
                Nothing else in the shared state says anything about intent, and this exists
                only because the world cannot be asked which side was meant to be the front.
                
        """
        self.calls["threshold"] += 1
        t = self.net.threshold(label) if self.net else None
        if t is None:
            return None
        # Named so they cannot be misread: `floor_block_y` is where the floor block
        # goes, `stand_y` is the cell a person occupies standing on it. An earlier
        # version called the second one "floor_y" and it would have put every floor a
        # block high.
        return {"id": t.id, "lane": [t.x, t.y, t.z], "facing": t.facing,
                "door": list(t.door), "floor_block_y": t.y, "stand_y": t.y + 1,
                **self.sited.get(label, {})}

    def nearest_lane(self, x: int, z: int) -> dict | None:
        self.calls["nearest_lane"] += 1
        if not self.net:
            return None
        n = self.net.nearest(int(x), int(z))
        if n is None:
            return None
        return {"x": n[0], "z": n[1], "y": n[2], "distance": n[3]}

    def check_door(self, x: int, y: int, z: int, pending: dict | None = None) -> dict:
        """Is this doorway on the walkable network, with what the pass has built so far?

                Returns {"ok", "reason", "jumps", "stance_y", "nearest_lane"}. `ok` is False
                both when nobody can stand in the doorway at all and when they can but cannot
                get there from the lane without jumping -- the second is the headline defect and
                the one no check could see before the network existed.
                
        """
        self.calls["check_door"] += 1
        x, y, z = int(x), int(y), int(z)
        r = self.radius
        sub = self.vol.sub(x - r, z - r, 2 * r + 1, 2 * r + 1)
        if pending:
            sub.overlay({p: b for p, b in pending.items()
                         if abs(p[0] - x) <= r and abs(p[2] - z) <= r})
        nav = observe.Nav(sub)
        out = {"ok": False, "reason": "", "jumps": None, "stance_y": None,
               "nearest_lane": self.nearest_lane(x, z)}

        s = nav.stance_near(x, z, y, tol=2)
        if s is None:
            out["reason"] = ("nothing can stand in this doorway -- it is blocked, or "
                             "there is no floor at this height")
            self.results.append({"pos": [x, y, z], **out})
            return out
        out["stance_y"] = s / 2.0

        seeds = []
        for (lx, lz, ly) in (self.net.surface() if self.net else []):
            if abs(lx - x) > r or abs(lz - z) > r:
                continue
            ls = nav.stance_near(lx, lz, ly + 1, tol=1)
            if ls is not None:
                seeds.append((lx, lz, ls))
        if not seeds:
            # Widen once before giving up. A fixed radius turns "your door is a long way
            # from the lane" into "your door is not on the network", and a builder that
            # believes it moves a door that was fine -- a check limitation deforming a
            # design, which is the thing this whole layer exists to avoid.
            r2 = r * 3
            sub = self.vol.sub(x - r2, z - r2, 2 * r2 + 1, 2 * r2 + 1)
            if pending:
                sub.overlay({p: b for p, b in pending.items()
                             if abs(p[0] - x) <= r2 and abs(p[2] - z) <= r2})
            nav = observe.Nav(sub)
            s = nav.stance_near(x, z, y, tol=2)
            for (lx, lz, ly) in (self.net.surface() if self.net else []):
                if abs(lx - x) > r2 or abs(lz - z) > r2:
                    continue
                ls = nav.stance_near(lx, lz, ly + 1, tol=1)
                if ls is not None:
                    seeds.append((lx, lz, ls))
        if not seeds or s is None:
            out["reason"] = (f"no lane within {r * 3} blocks of this doorway; the "
                             f"nearest is {out['nearest_lane']}. That is a distance, "
                             f"not a defect -- the whole-town checks judge it.")
            self.results.append({"pos": [x, y, z], **out})
            return out

        walk = nav.flood(seeds, max_jumps=0)
        if (x, z, s) in walk:
            out["ok"] = True
            out["jumps"] = 0
            out["reason"] = "on the lane network, walkable"
        else:
            scramble = nav.flood(seeds)
            j = scramble.get((x, z, s))
            out["jumps"] = j
            out["reason"] = (f"reachable from the lane only by jumping {j} times"
                             if j else
                             "cannot be reached from the lane on foot at all")
        self.results.append({"pos": [x, y, z], **out})
        return out

    # --- harness side -----------------------------------------------------
    def report(self) -> dict:
        bad = [r for r in self.results if not r["ok"]]
        return {"calls": dict(self.calls), "checked": len(self.results),
                "failed_when_asked": len(bad),
                "examples": bad[:5]}
