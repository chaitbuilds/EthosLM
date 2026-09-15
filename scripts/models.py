"""Which model does which job, as the router resolves it right now.

    python scripts/models.py            the route table: role, provider/model, ready or why not
    python scripts/models.py --ping     one tiny text request to every distinct ready route,
                                        so a stranger can see the wire work before spending
                                        a real call on it
    python scripts/models.py --json     the table as JSON

Exit status is 1 when a routed role cannot be called (a key missing, a provider or tier
unknown) or a ping fails, so a round's own command can refuse to start rather than fail
at its first model call an hour in. With nothing configured every role is `subagent`,
which is the default and is not an error.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from ethoslm import model                                                # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--ping", action="store_true")
    ap.add_argument("--json", action="store_true")
    a = ap.parse_args()
    try:
        r = model.Router()
    except (ValueError, OSError) as e:
        print(f"!! {e}")
        return 1
    rows = r.table()
    if a.json:
        print(json.dumps(rows, indent=1))
    else:
        print(f"models file: {model.models_path()}")
        for row in rows:
            mark = "  " if row["note"] in ("ready", "staged for the supervising agent") \
                else "!!"
            print(f"{mark} {row['role']:<7} {row['route']:<40} {row['note']}")
    bad = sum(1 for row in rows if row["note"] not in
              ("ready", "staged for the supervising agent"))
    if not a.ping:
        return 1 if bad else 0
    seen = set()
    for role in model.ROLES:
        rt = r.routes.get(role)
        if rt is None or rt.name in seen or rt.ready:
            continue
        seen.add(rt.name)
        t0 = time.perf_counter()
        try:
            ans = rt.transport()({"model": rt.model, "max_tokens": 32, "messages": [
                {"role": "user", "content": [{"type": "text", "text":
                    "Reply with the single word: ready"}]}]})
            text = "".join(b.get("text", "") for b in ans["content"]
                           if b.get("type") == "text").strip()
            print(f"   {rt.name}: {text!r} ({ans['usage']['input_tokens']} in, "
                  f"{ans['usage']['output_tokens']} out, "
                  f"{time.perf_counter() - t0:.1f} s)")
        except RuntimeError as e:
            print(f"!! {rt.name}: {e}")
            bad += 1
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
