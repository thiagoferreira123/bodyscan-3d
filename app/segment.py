"""
Server-side segmentation: turn the ORIGINAL photo the user uploaded into a
binary silhouette, used ONLY to measure (the original photo is never modified
nor replaced — it stays stored/displayed as-is).

photo -> rembg person mask -> binary -> bbox-framed 3:4 silhouette (BodyM-like).
Plus an arms-separation check so contaminated captures fall back to the model.
"""

from __future__ import annotations

import io

import numpy as np
from PIL import Image

_SESSION = None


def _session():
    global _SESSION
    if _SESSION is None:
        from rembg import new_session
        _SESSION = new_session("u2net_human_seg")
    return _SESSION


def photo_to_silhouette(image_bytes: bytes, out_w: int = 768, out_h: int = 1024,
                        frac: float = 0.88) -> bytes | None:
    """Original photo -> canonically-framed binary silhouette PNG. None if no person."""
    try:
        from rembg import remove
        cut = remove(Image.open(io.BytesIO(image_bytes)).convert("RGB"), session=_session())
        binary = np.asarray(cut.split()[-1]) > 127
        ys, xs = np.where(binary)
        if len(ys) < 50:
            return None
        y0, y1, x0, x1 = ys.min(), ys.max(), xs.min(), xs.max()
        body = Image.fromarray((binary[y0:y1 + 1, x0:x1 + 1] * 255).astype("uint8"), "L")
        th = int(out_h * frac)
        sc = th / body.height
        nw = max(1, int(body.width * sc))
        body = body.resize((nw, th), Image.LANCZOS)
        canvas = Image.new("L", (out_w, out_h), 0)
        canvas.paste(body, ((out_w - nw) // 2, (out_h - th) // 2))
        arr = (np.asarray(canvas) > 127).astype("uint8") * 255
        out = io.BytesIO()
        Image.fromarray(arr, "L").save(out, format="PNG")
        return out.getvalue()
    except Exception:
        return None


def arms_separated(silhouette_bytes: bytes) -> bool:
    """True if the FRONT silhouette shows a gap between arms and torso (>=3 white
    runs across the waist band). Arms merged with the body -> inflated waist ->
    we reject and fall back to the model."""
    try:
        a = np.asarray(Image.open(io.BytesIO(silhouette_bytes)).convert("L")) > 127
        ys = np.where(a.any(axis=1))[0]
        if len(ys) == 0:
            return False
        top, bot = ys.min(), ys.max()
        h = bot - top
        for fr in (0.36, 0.43, 0.50):
            row = a[int(top + fr * h)]
            runs, prev = 0, False
            for v in row:
                if v and not prev:
                    runs += 1
                prev = v
            if runs >= 3:
                return True
        return False
    except Exception:
        return False
