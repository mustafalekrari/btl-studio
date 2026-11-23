"""Orchestration du pipeline BTL Studio. Les étapes non implémentées lèvent NotImplementedError."""
from __future__ import annotations

from dataclasses import dataclass

from PIL import Image

from .config import load_config
from .cutout import FurnitureCutout
from .placement import Placement, place_furniture
from .scene_analysis import DepthEstimator, SceneAnalysis


@dataclass
class BaselineResult:
    scene: SceneAnalysis
    placement: Placement


def run_baseline(scene_img: Image.Image, furniture_img: Image.Image,
                 box: tuple[int, int, int, int]) -> BaselineResult:
    """Baseline naïve (copier-coller détouré) — sert de point de comparaison à l'évaluation."""
    scene = DepthEstimator()(scene_img)
    furniture_rgba, _ = FurnitureCutout()(furniture_img)
    return BaselineResult(scene=scene, placement=place_furniture(scene_img, furniture_rgba, box))


def run_full(scene_img: Image.Image, furniture_img: Image.Image, box, prompt: str):
    cfg = load_config()
    base = run_baseline(scene_img, furniture_img, box)
    from .integration import GenerativeIntegrator
    from .harmonization import harmonize

    variants = GenerativeIntegrator(cfg)(base.placement.composite, base.placement.mask,
                                         base.scene.depth_image, furniture_img, prompt)
    return [harmonize(v, base.placement.mask) for v in variants]
