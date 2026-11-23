"""Étape 5 — Contenu marketing : VLM (description de la scène) + LLM (textes). Prévu J10."""
from __future__ import annotations

from dataclasses import dataclass

from PIL import Image


@dataclass
class MarketingContent:
    description_produit: str
    legende_reseaux: str
    hashtags: list[str]


def generate_content(image: Image.Image, fiche_produit: dict | None = None,
                     langue: str = "fr") -> MarketingContent:
    raise NotImplementedError("Prévu J10")
