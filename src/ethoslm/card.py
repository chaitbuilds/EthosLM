"""One card composer, and the rule about what may not go on a card.

So: one `compose`, one layout vocabulary, and one refusal rule.

**The refusal rule.** `strict` refuses to compose a card that has a near-black panel in
it. A card with a hole in it is not evidence about a building, and the honest thing to
return is nothing at all -- with the reason recorded in `DAMAGE`, because a composer
that just returns None turns a broken camera into a missing row three files downstream.

**Byte-identity.** Every layout here reproduces its original composer's `hstack`/
`vstack` arithmetic exactly, gap for gap. It has to: the judge caches on
`sha256(image_a) + sha256(image_b) + question`, so a card that differs by one pixel is
an experiment that stops replaying. `scripts/test_pipeline.py` re-composes every
historical card and checks the digest against the one already on disk."""
from __future__ import annotations

import os

import numpy as np

#: The grey every card in this project has ever been matted on.
BG = 235
#: The gutter between panels, in pixels.
GAP = 8

#: Layouts, as rows of frame keys. `None` is a deliberate blank cell. `quad` is the
#: four-panel building card: eye-level at the door and the aerial on top, the two close
#: three-quarters underneath. Used by E1d, step 3 and step 4's check 2 -- the same
#: layout in all three, which was the one thing they did agree on.
LAYOUTS = {
    "quad": [["eye", "aerial"], ["ne", "sw"]],
    "pair": [["45", "225"]],
    "e3": [["a45", "a135", "eye0"], ["a225", "a315", "eye1"]],
}

#: Why the last `compose` for a tag returned None. Keyed by tag, cleared on success.
DAMAGE: dict = {}


def _read(path):
    import cv2
    return None if not path or not os.path.exists(path) else cv2.imread(path)


def compose(frames: dict, out_path: str, *, layout="quad", size=None,
            strict: bool = True, tag: str | None = None,
            blank_keys=("eye",), required=None, black_mean: float | None = None):
    """Lay `frames` ({key: png path}) out on one card. Returns the path, or None.

        `blank_keys` may be missing and are matted grey -- a structure with no reserved
        threshold has no eye-level shot, and that is a fact about the settlement rather
        than a broken camera. Everything else in the layout is `required`: absent, and the
        card is refused.

        `strict` additionally refuses a card any of whose panels is below `black_mean`
        (`render.BLACK_MEAN` by default).
        
    """
    import cv2
    from . import render
    rows = LAYOUTS[layout] if isinstance(layout, str) else layout
    keys = [k for row in rows for k in row if k]
    tag = tag if tag is not None else os.path.basename(out_path)
    thr = render.BLACK_MEAN if black_mean is None else black_mean
    need = set(keys) - set(blank_keys) if required is None else set(required)

    imgs, dark, missing = {}, [], []
    for k in keys:
        im = _read(frames.get(k))
        imgs[k] = im
        if im is None:
            if k in need:
                missing.append(k)
        elif float(im.mean()) < thr:
            dark.append(k)
    if missing:
        DAMAGE[tag] = "missing " + ",".join(sorted(missing))
        return None
    if strict and dark:
        DAMAGE[tag] = "black panels: " + ",".join(sorted(dark))
        return None
    DAMAGE.pop(tag, None)

    if size is None:
        some = next(im for im in imgs.values() if im is not None)
        size = (some.shape[1], some.shape[0])
    blank = np.full((size[1], size[0], 3), BG, np.uint8)
    gap_col = np.full((size[1], GAP, 3), BG, np.uint8)

    built = []
    for row in rows:
        cells = []
        for i, k in enumerate(row):
            if i:
                cells.append(gap_col)
            cells.append(blank if k is None or imgs[k] is None else imgs[k])
        built.append(np.hstack(cells))
    stacked = [built[0]]
    for r in built[1:]:
        stacked.append(np.full((GAP, built[0].shape[1], 3), BG, np.uint8))
        stacked.append(r)
    grid = np.vstack(stacked) if len(stacked) > 1 else built[0]
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    cv2.imwrite(out_path, grid)
    return out_path


def from_dir(frames_dir: str, tag: str, out_path: str, *, layout="quad", **kw):
    """`compose` over the frames a shot list wrote: `<frames_dir>/<tag>_<key>.png`."""
    rows = LAYOUTS[layout] if isinstance(layout, str) else layout
    keys = [k for row in rows for k in row if k]
    frames = {k: os.path.join(frames_dir, f"{tag}_{k}.png") for k in keys}
    return compose(frames, out_path, layout=layout, tag=tag, **kw)


def stack(images: list, out_path: str, *, centred: bool = True, gap: int = GAP):
    """Variable-width images stacked vertically on a common grey field.

        E1a's card (an isometric over a row of four elevations) and step 4's preview card
        (an isometric over a front elevation) are both this, and neither is a grid of equal
        frames -- previews are cropped to the built mass, so every one is a different size.
        
    """
    import cv2
    w = max(i.shape[1] for i in images)

    def pad(img):
        out = np.full((img.shape[0], w, 3), BG, np.uint8)
        o = (w - img.shape[1]) // 2 if centred else 0
        out[:, o:o + img.shape[1]] = img
        return out

    parts = [pad(images[0])]
    for i in images[1:]:
        parts.append(np.full((gap, w, 3), BG, np.uint8))
        parts.append(pad(i))
    card = np.vstack(parts)
    if out_path:
        os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
        cv2.imwrite(out_path, card[:, :, ::-1])
    return card


def row(images: list, gap: int = 6, bg: int = BG):
    """A row of variable-height images, top-aligned on a grey field. E1a's elevations."""
    row_w = sum(i.shape[1] for i in images) + gap * (len(images) - 1)
    row_h = max(i.shape[0] for i in images)
    out = np.full((row_h, row_w, 3), bg, np.uint8)
    x = 0
    for i in images:
        out[:i.shape[0], x:x + i.shape[1]] = i
        x += i.shape[1] + gap
    return out
