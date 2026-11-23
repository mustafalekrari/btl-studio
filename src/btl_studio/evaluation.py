"""Métriques d'évaluation (section 9 du cahier des charges). Implémentation complète J11–J12."""
from __future__ import annotations

import numpy as np
from PIL import Image


def background_preservation_lpips(source: Image.Image, result: Image.Image, mask: Image.Image) -> float:
    """LPIPS hors masque : plus c'est bas, mieux la scène est préservée (cible <= 0,05)."""
    import lpips
    import torch

    m = (np.asarray(mask.convert("L").resize(source.size)) < 128)[..., None]
    a = np.asarray(source.convert("RGB"), dtype=np.float32) * m
    b = np.asarray(result.convert("RGB").resize(source.size), dtype=np.float32) * m
    to_t = lambda x: torch.from_numpy(x / 127.5 - 1).permute(2, 0, 1)[None]
    with torch.no_grad():
        return float(lpips.LPIPS(net="alex", verbose=False)(to_t(a), to_t(b)))


def product_fidelity_dinov2(catalogue: Image.Image, inserted_crop: Image.Image) -> float:
    """Similarité cosinus DINOv2 entre meuble catalogue et meuble inséré (cible >= 0,75). J11."""
    raise NotImplementedError("Prévu J11–J12")
