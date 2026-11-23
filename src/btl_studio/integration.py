"""Étape 3 — Intégration générative (prévue J5–J7).

SDXL Inpainting + ControlNet Depth (géométrie) + IP-Adapter (identité du meuble).
"""
from __future__ import annotations

from PIL import Image


class GenerativeIntegrator:
    def __init__(self, cfg: dict):
        self.cfg = cfg
        # J5 : StableDiffusionXLControlNetInpaintPipeline + ControlNetModel (depth)
        #      + pipe.load_ip_adapter(repo, subfolder, weight_name)

    def __call__(self, composite: Image.Image, mask: Image.Image, depth: Image.Image,
                 furniture: Image.Image, prompt: str, seed: int = 42) -> list[Image.Image]:
        raise NotImplementedError("Prévu J5–J7")
