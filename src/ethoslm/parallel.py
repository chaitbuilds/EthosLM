"""Run independent instance checks across processes, in order, with the same answer.

The type layer's instruments are embarrassingly parallel and were entirely serial. The
conformance suite stands one hand-written type on 36 terrain fixtures at two seeds and
two storey counts -- 144 instances that share nothing, each of which sites its own
ground, builds, and lints what it built -- and it took a hundred seconds on one core.
A type's own checker is the same shape: one composed program per fixture round per seed,
and nothing that one of them does is visible to another.

Two rules, and they are what make this a harness change rather than a measurement one:

**The answer does not move.** Work items are submitted in order and collected in order,
so the rows come back in exactly the sequence a serial loop produces. Nothing here
reduces, sorts, or decides anything; it only moves where the work happens.

**One process, one item, no shared state.** `fork` is the start method, deliberately:
every fixture bank, block registry and cached volume this project loads is read-only
after import, so a forked child inherits them warm and pickles nothing but the item and
the result. `spawn` would re-import and re-read the terrain bank once per worker.

`ETHOSLM_WORKERS` sets the pool size; `ETHOSLM_WORKERS=1` runs the work inline, in this
process, with no pool at all -- which is the serial arm of every comparison and the
thing to reach for when a traceback is more useful than a speed-up.
"""
from __future__ import annotations

import os


def workers(default: int | None = None) -> int:
    """How many processes to use. `ETHOSLM_WORKERS`, else the machine, capped."""
    got = os.environ.get("ETHOSLM_WORKERS")
    if got:
        try:
            return max(1, int(got))
        except ValueError:
            pass
    if default is not None:
        return max(1, int(default))
    return max(1, min(8, (os.cpu_count() or 1)))


def par_map(fn, items, *, n: int | None = None, chunk: int = 1) -> list:
    """`[fn(i) for i in items]`, across processes, in the same order.

        Falls back to the plain list comprehension for one worker, for fewer items than
        workers, or where a pool cannot be made -- so a caller never has two code paths and
        the serial answer is always available by one environment variable.
        
    """
    items = list(items)
    k = workers(n)
    if k <= 1 or len(items) <= 1:
        return [fn(i) for i in items]
    import concurrent.futures as cf
    import multiprocessing as mp
    try:
        ctx = mp.get_context("fork")
    except ValueError:                          # noqa: BLE001 -- not on this platform
        return [fn(i) for i in items]
    with cf.ProcessPoolExecutor(max_workers=k, mp_context=ctx) as pool:
        return list(pool.map(fn, items, chunksize=max(1, int(chunk))))
