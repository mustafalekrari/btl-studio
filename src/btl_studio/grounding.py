"""Placement par instruction texte : « contre le mur de gauche » -> boîte cible (prévu J8–J9)."""
from __future__ import annotations

from PIL import Image


def text_to_box(scene: Image.Image, instruction: str) -> tuple[int, int, int, int]:
    raise NotImplementedError("Prévu J8–J9 (Grounding DINO / Florence-2)")
