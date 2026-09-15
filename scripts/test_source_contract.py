"""Source layout and documentation contracts, independent of cached worlds."""
import ast
from pathlib import Path
import re
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


class SourceContract(unittest.TestCase):
    def test_no_historical_identifiers_in_source_docstrings(self):
        pattern = re.compile(r"\brounds?[-_\s]*\d+|\bsessions?\b|\bEllul\b", re.I)
        bad = []
        for path in (ROOT / "src").rglob("*.py"):
            for node in ast.walk(ast.parse(path.read_text())):
                if isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
                    if pattern.search(ast.get_docstring(node) or ""):
                        bad.append(f"{path.relative_to(ROOT)}:{getattr(node, 'lineno', 1)}")
        self.assertEqual(bad, [])

    def test_public_stage_imports_keep_identity(self):
        from ethoslm import pipeline, place
        from ethoslm.pipeline import stages_plan, stages_build, stages_measure
        self.assertIs(place.stage_plan, stages_plan.stage_plan)
        self.assertIs(place.stage_parts, stages_build.stage_parts)
        self.assertIs(place.stage_place_check, stages_measure.stage_place_check)
        self.assertIs(pipeline.plan_failures, stages_plan.plan_failures)

    def test_stage_annotations_still_resolve(self):
        import typing
        from ethoslm import pipeline
        self.assertIs(typing.get_type_hints(pipeline.stage_waves)["rnd"], pipeline.Round)
        self.assertIs(typing.get_type_hints(pipeline.stage_type_check)["rnd"], pipeline.Round)

    def test_machine_paths_are_not_in_executable_sources(self):
        bad = []
        for base in (ROOT / "src", ROOT / "scripts"):
            for path in base.rglob("*"):
                if path.suffix not in (".py", ".sh") or path.name == Path(__file__).name:
                    continue
                if re.search(r"/(?:nix/store|home/[^/\s]+)/", path.read_text()):
                    bad.append(str(path.relative_to(ROOT)))
        self.assertEqual(bad, [])


if __name__ == "__main__":
    unittest.main()
