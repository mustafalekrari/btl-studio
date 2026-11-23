"""Étape 2 — Placement : composite initial, masque d'inpainting, mise à l'échelle."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from PIL import Image, ImageFilter


@dataclass
class Placement:
    composite: Image.Image   # scène + meuble collé (baseline naïve)
    mask: Image.Image        # zone à régénérer (meuble + marge pour ombres)
    box: tuple[int, int, int, int]


def place_furniture(scene: Image.Image, furniture_rgba: Image.Image,
                    box: tuple[int, int, int, int], margin: int = 24) -> Placement:
    """Colle le meuble dans la boîte (x0, y0, x1, y1) en conservant ses proportions,
    aligné en bas de la boîte (contact avec le sol)."""
    x0, y0, x1, y1 = box
    bw, bh = x1 - x0, y1 - y0
    fw, fh = furniture_rgba.size
    s = min(bw / fw, bh / fh)
    nw, nh = max(1, int(fw * s)), max(1, int(fh * s))
    item = furniture_rgba.resize((nw, nh), Image.LANCZOS)
    px, py = x0 + (bw - nw) // 2, y1 - nh

    composite = scene.convert("RGB").copy()
    composite.paste(item, (px, py), item)

    alpha = Image.new("L", scene.size, 0)
    alpha.paste(item.getchannel("A"), (px, py))
    mask = alpha.point(lambda p: 255 if p > 10 else 0).filter(ImageFilter.MaxFilter(2 * (margin // 2) + 1))
    return Placement(composite=composite, mask=mask, box=(px, py, px + nw, py + nh))


def scale_box_from_depth(depth: np.ndarray, anchor_xy: tuple[int, int], real_width_cm: float,
                         ref_px_per_cm_at_near: float = 2.0) -> int:
    """J5–J7 : largeur en pixels déduite de la profondeur au point d'ancrage.
    Heuristique simple : plus le point est loin (depth faible), plus le meuble est petit."""
    x, y = anchor_xy
    d = float(depth[y, x])
    return int(real_width_cm * ref_px_per_cm_at_near * (0.3 + 0.7 * d))
