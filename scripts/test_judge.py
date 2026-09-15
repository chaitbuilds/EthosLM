"""The judge's contract, checked with stub models -- no subagent, no image model.

What a stub can prove and a model cannot: that the bracket, the cache, the blinding
and the position swap are mechanism, not hope. The stubs are adversarial on purpose --
one judges by content, one answers purely by position (the failure mode the swap
exists to catch), one refuses to answer at all.

  1. A bracket costs k-1 comparisons, two calls each, and picks the content winner.
  2. A warm-cache re-run makes zero model calls and returns the identical winner.
  3. The cache is content-addressed: the same bytes under a different filename cost
     nothing.
  4. The model sees blinded copies -- a.png / b.png in a hash-named scratch dir with
     the right bytes -- and the prompt never contains a candidate's real name.
  5. A judge that answers from position produces ties, not winners: 100% tie rate,
     and the bracket still terminates deterministically.
  6. Eight copies of the same image tie at >80% (structurally, here: identical bytes
     share a key across orientations, so identical content can never be resolved).
  7. With no model attached, the bracket raises the full round of staged requests;
     fulfilling them by hand and re-running resolves from cache alone.
"""
import os
import shutil
import sys
import tempfile

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from ethoslm import judge, measure  # noqa: E402

# The stubs go through judge.py's real costing path; without this, every test run
# appends ~31 fake model_call rows to the log every economics claim reads.
measure.LOG = os.path.join(tempfile.mkdtemp(prefix="judge_test_measure_"),
                           "measurements.jsonl")


def make_images(d, sizes):
    out = []
    for i, n in enumerate(sizes):
        p = os.path.join(d, f"candidate_{i}.png")
        open(p, "wb").write(bytes([i]) * n)
        out.append(p)
    return out


def main():
    cases = []
    d = tempfile.mkdtemp(prefix="judge_test_")
    cache = os.path.join(d, "cache.jsonl")

    calls = {"n": 0}

    def by_content(a, b, prompt):
        """Prefers the bigger file. Position-invariant, like a real judge should be."""
        calls["n"] += 1
        return ("A" if os.path.getsize(a) >= os.path.getsize(b) else "B") + " -- bigger"

    imgs = make_images(d, [10, 40, 20, 30])
    w, log = judge.judge(imgs, "which is bigger?", ask=by_content, cache_path=cache)
    cases.append(("bracket picks the content winner",
                  w == imgs[1], f"winner {os.path.basename(w)}"))
    cases.append(("k-1 comparisons, two orientations each",
                  log["comparisons"] == 3 and calls["n"] == 6,
                  f"{log['comparisons']} comparisons, {calls['n']} calls"))
    cases.append(("content judge, no ties", log["ties"] == 0, ""))

    calls["n"] = 0
    w2, log2 = judge.judge(imgs, "which is bigger?", ask=by_content, cache_path=cache)
    cases.append(("warm cache: zero model calls, identical winner",
                  calls["n"] == 0 and w2 == w, f"{calls['n']} calls"))

    # content addressing: same bytes, different names, still zero calls
    copies = []
    for i, p in enumerate(imgs):
        c = os.path.join(d, f"renamed_{i}.png")
        shutil.copyfile(p, c)
        copies.append(c)
    calls["n"] = 0
    w3, _ = judge.judge(copies, "which is bigger?", ask=by_content, cache_path=cache)
    cases.append(("cache is content-addressed, not path-addressed",
                  calls["n"] == 0 and os.path.getsize(w3) == os.path.getsize(w),
                  f"{calls['n']} calls"))

    # blinding
    blind_ok = {"v": True}
    originals = {os.path.basename(p) for p in imgs}

    def paranoid(a, b, prompt):
        if os.path.basename(a) != "a.png" or os.path.basename(b) != "b.png":
            blind_ok["v"] = False
        if "judge_scratch" not in a or "judge_scratch" not in b:
            blind_ok["v"] = False
        if any(name in prompt for name in originals):
            blind_ok["v"] = False
        return by_content(a, b, prompt)

    cache2 = os.path.join(d, "cache2.jsonl")
    judge.judge(imgs, "which is bigger?", ask=paranoid, cache_path=cache2)
    cases.append(("the model sees a.png/b.png in a hash-named dir, "
                  "prompt names no candidate", blind_ok["v"], ""))

    # a scratch copy holds the candidate's actual bytes
    req = judge.stage(judge.request(imgs[0], imgs[1], "q", cache_path=cache2))
    same = (open(req["scratch_a"], "rb").read() == open(imgs[0], "rb").read()
            and open(req["scratch_b"], "rb").read() == open(imgs[1], "rb").read())
    cases.append(("blinded copies carry the right bytes", same, ""))

    # position bias -> ties, never winners
    def by_position(a, b, prompt):
        return "A, the first one looks better"

    cache3 = os.path.join(d, "cache3.jsonl")
    w4, log4 = judge.judge(imgs, "which reads better?", ask=by_position,
                           cache_path=cache3)
    cases.append(("a position-answering judge produces 100% ties",
                  log4["tie_rate"] == 1.0, f"tie rate {log4['tie_rate']}"))
    cases.append(("...and the bracket still terminates on the first candidate",
                  w4 == imgs[0], os.path.basename(w4)))

    # eight copies of the same image: the acceptance test
    same_img = [os.path.join(d, f"same_{i}.png") for i in range(8)]
    for p in same_img:
        open(p, "wb").write(b"\x89PNG identical" * 5)
    cache4 = os.path.join(d, "cache4.jsonl")
    w5, log5 = judge.judge(same_img, "which is better made?", ask=by_content,
                           cache_path=cache4)
    cases.append(("8 copies of one image: tie rate above 80%",
                  log5["tie_rate"] > 0.8,
                  f"tie rate {log5['tie_rate']:.2f} over {log5['comparisons']}"))

    # unparseable answer -> no preference -> tie
    cache5 = os.path.join(d, "cache5.jsonl")
    res = judge.compare(imgs[0], imgs[1], "q2",
                        ask=lambda a, b, p: "I really cannot tell.",
                        cache_path=cache5)
    cases.append(("an unparseable answer is a tie, not a crash", res == "tie", res))

    # odd field: a bye, and the right winner
    cache6 = os.path.join(d, "cache6.jsonl")
    odd = make_images(d, [10, 40, 20, 30, 50])
    w6, log6 = judge.judge(odd, "which is bigger?", ask=by_content, cache_path=cache6)
    cases.append(("odd field: bye advances, biggest still wins",
                  w6 == odd[4] and log6["comparisons"] == 4,
                  f"{log6['comparisons']} comparisons"))

    # the orchestrated path: no model attached
    cache7 = os.path.join(d, "cache7.jsonl")
    try:
        judge.judge(imgs[:2], "cold?", ask=None, cache_path=cache7)
        cases.append(("cold cache with no model raises JudgementNeeded", False, ""))
    except judge.JudgementNeeded as e:
        staged = all(os.path.exists(r["scratch_a"]) and os.path.exists(r["scratch_b"])
                     for r in e.requests)
        cases.append(("cold cache with no model raises staged requests",
                      len(e.requests) == 2 and staged, f"{len(e.requests)} requests"))
        for r in e.requests:
            judge.fulfil(r, "B -- answered by hand", 1.0, cache_path=cache7)
    w7, log7 = judge.judge(imgs[:2], "cold?", ask=None, cache_path=cache7)
    cases.append(("fulfilled by hand, the re-run resolves from cache alone",
                  log7["comparisons"] == 1 and log7["ties"] == 1, ""))

    fails = 0
    for label, ok, detail in cases:
        print(f"{'ok  ' if ok else 'FAIL'} {label}" + (f"  ({detail})" if detail else ""))
        fails += not ok
    print(f"\n{len(cases) - fails}/{len(cases)} judge cases pass")
    shutil.rmtree(d, ignore_errors=True)
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
