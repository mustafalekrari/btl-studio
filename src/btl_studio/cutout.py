"""Étape 1 bis — Détourage du meuble (BiRefNet)."""
from __future__ import annotations

import numpy as np
from PIL import Image

from .config import get_device, load_config


class FurnitureCutout:
    def __init__(self, model_id: str | None = None, device: str | None = None):
        import torch
        from torchvision import transforms
        from transformers import AutoModelForImageSegmentation

        cfg = load_config()
        self.device = device or get_device()
        self.model = AutoModelForImageSegmentation.from_pretrained(
            model_id or cfg["models"]["cutout"], trust_remote_code=True
        ).to(self.device).eval()
        self.tf = transforms.Compose([
            transforms.Resize((1024, 1024)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])
        self._torch = torch

    def __call__(self, image: Image.Image) -> tuple[Image.Image, np.ndarray]:
        """Retourne (meuble RGBA détouré, masque binaire HxW)."""
        rgb = image.convert("RGB")
        x = self.tf(rgb).unsqueeze(0).to(self.device)
        with self._torch.no_grad():
            pred = self.model(x)[-1].sigmoid().cpu()[0].squeeze().numpy()
        mask = Image.fromarray((pred * 255).astype(np.uint8)).resize(rgb.size)
        rgba = rgb.copy()
        rgba.putalpha(mask)
        bbox = mask.point(lambda p: 255 if p > 127 else 0).getbbox()
        if bbox:
            rgba = rgba.crop(bbox)
        return rgba, np.asarray(mask) > 127
