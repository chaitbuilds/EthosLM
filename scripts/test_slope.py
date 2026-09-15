"""Slope fixture contract, or --recheck NAME for a full offline type report."""
import json
from pathlib import Path
import sys
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from ethoslm import observe, pipeline
from ethoslm.slopefixture import descriptor, make


def recheck(name, output):
    decl = pipeline.load_type(str(ROOT / "types" / (name + ".py")))
    rnd = pipeline.Round.load(str(ROOT / "rounds/types_d.json"))
    kind = decl["kind"]
    fixtures = (pipeline._fixtures(rnd) + [descriptor(decl)] if kind == "plot" else
                [f for f in pipeline.check_parts() if f["kind"] == kind])
    seeds = [1, 2] if kind == "plot" else [1, 2, 3, 4]
    result = pipeline.check_type(rnd, pipeline.OfflineBackend(rnd),
        str(ROOT / "types" / (name + ".py")), [], seeds, fixtures=fixtures)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    (output / (name + ".md")).write_text(result.pop("text"))
    for row in result["rows"]:
        report = row.pop("report")
        row["findings"] = [dict(code=f.code, message=f.message, pos=f.pos)
                           for f in report.findings]
    result.update(type=name, fixtures=fixtures, seeds=seeds)
    (output / (name + ".json")).write_text(json.dumps(result, indent=2) + "\n")
    print(name, result["instances"], result["errors"], result["entry_lines"], result["seconds"], flush=True)


class Slope(unittest.TestCase):
    def test_type_walkability_uses_the_contexts_lane_seeded_answer(self):
        class Context:
            from_outdoors = {(1, 2, 3)}
        self.assertEqual(pipeline._type_walkable(Context()), {(1, 2, 3)})

    def test_slope_and_baseline_are_valid_before_a_type_runs(self):
        decl = pipeline.load_type(str(ROOT / "types/cottage.py"))
        fixture = descriptor(decl)
        rnd, be = make(fixture["round"], ROOT)
        h, _ = observe.ground_heights(be.volume)
        plot = json.loads(Path(rnd.rel("plots.json")).read_text())[0]
        # Measure well inside the plot, away from the prepared public approach.
        profile = h[plot["x0"]+3:plot["x1"]-1, plot["z1"]-3]
        self.assertTrue(np.all(profile[2:] - profile[:-2] == 1), profile)
        ctx = pipeline.build_context(rnd, be, None)
        self.assertFalse(pipeline.standard_report(ctx, ctx.plots).findings)
        self.assertIn(fixture, pipeline._fixtures_for(rnd, {"name": "cottage"}))


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--recheck":
        recheck(sys.argv[2], sys.argv[3])
    else:
        unittest.main()
