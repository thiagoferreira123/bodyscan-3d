"""
Height-anchored geometric waist from the two uploaded silhouettes.

The frontend now uploads canonically-framed BINARY silhouettes (person white on
black) and rejects captures where the arms are not clearly away from the torso.
On such a clean silhouette we can MEASURE the waist directly instead of asking a
black-box model to guess absolute cm (which it can't do reliably):

  - body pixel height (head->feet) maps to the known height_cm  => cm per pixel
  - waist width  = narrowest TORSO run on the front silhouette  (arms excluded)
  - waist depth  = torso thickness at that level on the side silhouette
  - circumference ~ ellipse perimeter (Ramanujan) x K (a single shape factor)

This scales physically with body size, so one K generalizes (unlike a per-body
model offset). K is calibrated on real tape data; default tuned to the one known
point. Requires a clean silhouette — hence the frontend capture validation.
"""

from __future__ import annotations

import io
import math

import numpy as np
from PIL import Image


def _is_binary_silhouette(arr: np.ndarray) -> bool:
    """True if the image is (near) pure black/white — i.e. a real silhouette,
    not a raw photo. Guards against measuring geometry on a photo that fell
    back from a failed capture."""
    near = ((arr < 16) | (arr > 239)).mean()
    return near >= 0.95


def _mask(image_bytes: bytes) -> np.ndarray:
    img = Image.open(io.BytesIO(image_bytes)).convert("L")
    arr = np.asarray(img)
    if not _is_binary_silhouette(arr):
        raise ValueError("not a binary silhouette")
    return arr > 127


def _rows(mask: np.ndarray) -> tuple[int, int]:
    ys = np.where(mask.any(axis=1))[0]
    return int(ys.min()), int(ys.max())


def _torso_run(row: np.ndarray) -> int:
    """Longest contiguous True run = torso width (arms are separate thinner runs)."""
    best = cur = 0
    for v in row:
        if v:
            cur += 1
            best = max(best, cur)
        else:
            cur = 0
    return best


def measure_waist_cm(
    front_bytes: bytes,
    side_bytes: bytes,
    height_cm: float,
    k: float = 1.08,
    waist_band: tuple[float, float] = (0.34, 0.53),
    smooth: int = 5,
) -> float | None:
    """Return the geometric waist circumference (cm), or None if unmeasurable."""
    try:
        f = _mask(front_bytes)
        s = _mask(side_bytes)
        if not f.any() or not s.any():
            return None

        ftop, fbot = _rows(f)
        body_px = fbot - ftop
        if body_px < 10:
            return None
        cmpp_f = height_cm / body_px

        y0 = int(ftop + waist_band[0] * body_px)
        y1 = int(ftop + waist_band[1] * body_px)
        widths = np.array([_torso_run(f[r]) for r in range(y0, y1)], dtype=float)
        if widths.size == 0:
            return None
        kern = np.ones(smooth) / smooth
        widths_s = np.convolve(widths, kern, mode="same")
        wi = int(np.argmin(widths_s))
        waist_w_px = float(np.median(widths[max(0, wi - smooth): wi + smooth]))
        width_cm = waist_w_px * cmpp_f
        waist_frac = (y0 + wi - ftop) / body_px

        stop, sbot = _rows(s)
        sbody = sbot - stop
        if sbody < 10:
            return None
        cmpp_s = height_cm / sbody
        srow = int(stop + waist_frac * sbody)
        depth_px = float(np.median(
            [_torso_run(s[r]) for r in range(max(stop, srow - smooth), srow + smooth)]
        ))
        depth_cm = depth_px * cmpp_s

        a, b = width_cm / 2, depth_cm / 2
        circ = math.pi * (3 * (a + b) - math.sqrt((3 * a + b) * (a + 3 * b)))
        result = circ * k
        # sanity clamp: implausible values mean a bad silhouette slipped through
        if not (30 <= result <= 200):
            return None
        return round(result, 1)
    except Exception:
        return None
