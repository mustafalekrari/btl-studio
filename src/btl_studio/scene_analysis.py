"""Étape 1 — Analyse de scène : carte de profondeur (+ segmentation sol/murs en J3–J4)."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from PIL import Image

from .config import get_device, load_config


@dataclass
class SceneAnalysis:
    depth: np.ndarray          # profondeur relative normalisée [0, 1], 1 = proche
    depth_image: Image.Image   # visualisation 8 bits, utilisée comme condition ControlNet
    floor_mask: np.ndarray | None = None  # rempli en J3–J4 (OneFormer ADE20K)


class DepthEstimator:
    def __init__(self, model_id: str | None = None, device: str | None = None):
        from transformers import pipeline

        cfg = load_config()
        self.device = device or get_device()
        self.pipe = pipeline(
            "depth-estimation",
            model=model_id or cfg["models"]["depth"],
            device=0 if self.device == "cuda" else -1,
        )

    def __call__(self, image: Image.Image) -> SceneAnalysis:
        out = self.pipe(image.convert("RGB"))
        depth = out["predicted_depth"].squeeze().cpu().numpy().astype(np.float32)
        depth = (depth - depth.min()) / (depth.max() - depth.min() + 1e-8)
        depth_img = Image.fromarray((depth * 255).astype(np.uint8)).resize(image.size)
        return SceneAnalysis(depth=np.asarray(depth_img, dtype=np.float32) / 255.0,
                             depth_image=depth_img.convert("RGB"))


def segment_floor(image: Image.Image) -> np.ndarray:
    """J3–J4 : masque binaire du sol via OneFormer (classe ADE20K 'floor')."""
    raise NotImplementedError("Prévu J3–J4")
