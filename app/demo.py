"""Démo Gradio BTL Studio — étapes disponibles : profondeur, détourage, placement (baseline).

Lancer : python app/demo.py            (local)
         python app/demo.py --share    (Colab : génère un lien public *.gradio.live)
"""
from __future__ import annotations

import argparse
import sys
from functools import lru_cache
from pathlib import Path

import gradio as gr
import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from btl_studio.placement import place_furniture  # noqa: E402


@lru_cache(maxsize=1)
def depth_model():
    from btl_studio.scene_analysis import DepthEstimator
    return DepthEstimator()


@lru_cache(maxsize=1)
def cutout_model():
    from btl_studio.cutout import FurnitureCutout
    return FurnitureCutout()


def white_background_cutout(img: Image.Image) -> Image.Image:
    """Repli si BiRefNet ne charge pas : fond blanc/clair des photos catalogue -> transparent."""
    rgb = np.asarray(img.convert("RGB")).astype(np.int16)
    alpha = np.where(rgb.min(axis=2) > 235, 0, 255).astype(np.uint8)
    out = img.convert("RGB")
    out.putalpha(Image.fromarray(alpha))
    bbox = out.getchannel("A").getbbox()
    return out.crop(bbox) if bbox else out


def cutout(furniture: Image.Image) -> Image.Image:
    if furniture.mode == "RGBA" and np.asarray(furniture.getchannel("A")).min() < 250:
        return furniture  # PNG déjà détouré
    try:
        return cutout_model()(furniture)[0]
    except Exception as e:  # noqa: BLE001
        print(f"[BiRefNet indisponible : {e!r}] -> détourage fond blanc")
        return white_background_cutout(furniture)


def run(scene: Image.Image, furniture: Image.Image, cx: float, base_y: float, width: float):
    if scene is None or furniture is None:
        raise gr.Error("Charge une photo de pièce et une photo de meuble.")
    scene = scene.convert("RGB")
    W, H = scene.size

    analysis = depth_model()(scene)
    rgba = cutout(furniture)

    bw = int(W * width / 100)
    x_center, y_bottom = int(W * cx / 100), int(H * base_y / 100)
    box = (max(0, x_center - bw // 2), 0, min(W, x_center + bw // 2), y_bottom)
    p = place_furniture(scene, rgba, box)

    depth_at_base = float(analysis.depth[min(y_bottom, H - 1), min(x_center, W - 1)])
    info = (f"Boîte placée : {p.box}  ·  profondeur relative au point de contact : {depth_at_base:.2f} "
            f"(1 = proche, 0 = loin)")

    checker = Image.new("RGBA", rgba.size, (235, 235, 235, 255))
    checker.alpha_composite(rgba)
    return analysis.depth_image, checker.convert("RGB"), p.mask, p.composite, info


def build() -> gr.Blocks:
    with gr.Blocks(title="BTL Studio") as demo:
        gr.Markdown("# BTL Studio — démo J1\n"
                    "Profondeur (Depth Anything V2) · détourage (BiRefNet) · placement baseline. "
                    "L'intégration générative (SDXL) arrive en J5–J7.")
        with gr.Row():
            scene = gr.Image(type="pil", label="Photo de la pièce")
            furniture = gr.Image(type="pil", label="Photo du meuble")
        with gr.Row():
            cx = gr.Slider(0, 100, 50, step=1, label="Position horizontale (%)")
            base_y = gr.Slider(0, 100, 85, step=1, label="Contact avec le sol (% hauteur)")
            width = gr.Slider(5, 100, 35, step=1, label="Largeur du meuble (% image)")
        btn = gr.Button("Lancer", variant="primary")
        info = gr.Markdown()
        with gr.Row():
            depth = gr.Image(label="Carte de profondeur")
            cut = gr.Image(label="Meuble détouré")
        with gr.Row():
            mask = gr.Image(label="Masque d'inpainting (zone régénérée en J5)")
            comp = gr.Image(label="Résultat baseline (copier-coller)")
        btn.click(run, [scene, furniture, cx, base_y, width], [depth, cut, mask, comp, info])
    return demo


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--share", action="store_true")
    args = ap.parse_args()
    build().launch(share=args.share)
