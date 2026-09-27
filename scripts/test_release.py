"""Delivery receipts must describe confirmed writes of the current candidate.

Exercise the production writer in temporary saves; only server I/O and baseline terrain
are substituted. No Minecraft instance or recorded build is required.
"""
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from ethoslm import cli
from ethoslm.pipeline import stages_design as SD


def put(path, doc):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(doc))


class Release(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.save = self.root / "save"
        self.save.mkdir()
        (self.save / "level.dat").touch()
        self.state = self.root / "out" / "sample"
        self.rnd = SimpleNamespace(name="sample", flags={"design": {"world": str(self.save)}})
        self.rnd.rel = lambda *p: str(self.state.joinpath(*p))
        self.region = self.state / "regions" / "0_0"
        put(self.region / "region.json", {"ok": True, "bounds": [0, 0, 10, 10]})
        self.diff({(0, 64, 0): "stone"})
        self.results = {"regions": {"built": ["0_0"], "kept": [], "failed": [],
                                    "regions": {"0_0": {"failed": 0}}}}
        put(self.state / "round.json", {"results": self.results})
        put(self.state / "sample.json", {"flags": self.rnd.flags})
        self.world = {}
        self.reply = "ok"
        self.calls = []

        def place(items, **kw):
            self.calls.append((items, kw))
            for pos, block in items:
                self.world[tuple(pos)] = block.id
            if self.reply == "exception":
                raise ConnectionError("lost response after writes")
            if self.reply == "short":
                return []
            good = self.reply == "ok" or (self.reply == "updates" and not kw["doBlockUpdates"])
            return [(good, "response") for _ in items]

        def baseline(rnd, x, z, w, d, y0, y1):
            return SimpleNamespace(x0=x, y0=y0, z0=z, palette=["air"],
                                   codes=np.zeros((w, y1-y0, d), dtype=np.uint16))

        for obj, name, value in ((cli, "ROOT", str(self.root)), (SD, "ROOT", str(self.root)),
                                  (SD, "base_volume", baseline)):
            p = patch.object(obj, name, value)
            p.start()
            self.addCleanup(p.stop)
        for name, kw in (("requests.get", {}),
                         ("gdpc.interface.placeBlocks", {"side_effect": place}),
                         ("gdpc.interface.runCommand", {"return_value": [(True, "saved")]})):
            p = patch(name, **kw)
            m = p.start()
            self.addCleanup(p.stop)
            if name.endswith("runCommand"):
                self.flush = m

    def diff(self, cells):
        palette = sorted(set(cells.values()))
        np.savez_compressed(self.region / "diff.npz",
                            xyz=np.array(list(cells), dtype=np.int32).reshape(-1, 3),
                            codes=np.array([palette.index(v) for v in cells.values()]),
                            palette=np.array(palette))

    def write(self):
        return SD.stage_region_write(self.rnd, None, {})

    def receipt(self):
        return json.loads((self.state / "written.json").read_text())

    def status(self):
        return cli.state_of("sample")["state"]

    def test_receipt_tracks_content_and_destination(self):
        self.assertEqual(self.status(), "designed")
        self.assertEqual(self.write()["written"], ["0_0"])
        self.assertEqual(self.status(), "delivered")
        self.assertEqual(self.write()["kept"], ["0_0"])
        self.diff({(1, 64, 0): "stone"})
        self.assertEqual(self.status(), "designed")
        self.assertEqual(self.write()["written"], ["0_0"])
        self.assertEqual(self.status(), "delivered")
        put(self.state / "sample.json", {"flags": {"design": {"world": "another-save"}}})
        self.assertEqual(self.status(), "designed")
        self.rnd.flags["design"]["world"] = str(self.root / "another-save")
        self.assertTrue(self.write()["stop"])

    def test_failed_or_short_write_is_retried(self):
        for reply in ("failed", "short", "exception"):
            with self.subTest(reply=reply):
                self.diff({(0, 64, len(self.calls)): "stone"})
                self.reply = reply
                self.assertTrue(self.write()["stop"])
                self.assertFalse(self.receipt()["complete"])
                self.assertEqual(self.status(), "designed")
                self.reply = "ok"
                before = len(self.calls)
                self.assertEqual(self.write()["written"], ["0_0"])
                self.assertGreater(len(self.calls), before)
                self.assertEqual(self.status(), "delivered")

    def test_connective_updates_must_succeed(self):
        self.diff({(0, 64, 0): "oak_fence"})
        self.reply = "updates"
        self.assertTrue(self.write()["stop"])
        self.assertEqual(self.receipt()["regions"]["0_0"]["status"], "pending")
        self.reply = "ok"
        self.assertEqual(self.write()["written"], ["0_0"])
        self.assertEqual(self.status(), "delivered")

    def test_partial_changed_candidate_restores_all_touched_cells(self):
        self.diff({(9, 64, 0): "stone"})
        self.write()
        self.diff({(0, 64, 0): "bricks"})
        self.reply = "exception"
        self.assertTrue(self.write()["stop"])
        self.diff({(5, 64, 0): "sandstone"})
        self.reply = "ok"
        self.assertEqual(self.write()["restored"], 2)
        self.assertEqual(self.world[(9, 64, 0)], "minecraft:air")
        self.assertEqual(self.world[(0, 64, 0)], "minecraft:air")
        self.assertEqual(self.world[(5, 64, 0)], "minecraft:sandstone")
        self.assertEqual(self.status(), "delivered")

    def test_save_flush_is_required_and_can_be_retried(self):
        self.flush.return_value = [(False, "not saved")]
        self.assertTrue(self.write()["stop"])
        self.assertEqual(self.status(), "designed")
        before = len(self.calls)
        self.flush.return_value = [(True, "saved")]
        self.assertEqual(self.write()["kept"], ["0_0"])
        self.assertEqual(len(self.calls), before)
        self.assertEqual(self.status(), "delivered")

    def test_incomplete_region_set_is_not_delivered(self):
        self.rnd.flags["design"]["write"] = ["0_0", "1_0"]
        self.assertTrue(self.write()["stop"])
        self.assertEqual(self.calls, [])
        self.results["regions"]["regions"]["0_0"]["failed"] = 1
        put(self.state / "round.json", {"results": self.results})
        self.assertEqual(self.status(), "blocked")

    def test_doctor_explains_missing_terrain_directory(self):
        with patch.object(cli, "NEEDS", ()):
            self.assertIn("generate terrain", " ".join(cli.doctor(str(self.save), quiet=True)))
            (self.save / "region").mkdir()
            (self.save / "region" / "r.0.0.mca").touch()
            self.assertEqual(cli.doctor(str(self.save), quiet=True), [])


if __name__ == "__main__":
    unittest.main()
