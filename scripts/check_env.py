"""Vérifie l'environnement BTL Studio : dépendances, GPU, et test de profondeur.

Usage : python scripts/check_env.py [--image chemin.jpg] [--skip-model]
"""
from __future__ import annotations

import argparse
import importlib
import sys
import time
from pathlib import Path

PACKAGES = ["torch", "torchvision", "diffusers", "transformers", "accelerate", "cv2",
            "PIL", "numpy", "timm", "fastapi", "gradio", "langchain", "lpips", "yaml"]


def check_packages() -> bool:
    ok = True
    for name in PACKAGES:
        try:
            mod = importlib.import_module(name)
            print(f"  [OK]  {name:<13} {getattr(mod, '__version__', '')}")
        except Exception as e:  # noqa: BLE001
            ok = False
            print(f"  [!!]  {name:<13} manquant ({e.__class__.__name__})")
    return ok


def check_gpu() -> None:
    import torch

    if torch.cuda.is_available():
        p = torch.cuda.get_device_properties(0)
        print(f"  GPU : {p.name} — {p.total_memory / 1024**3:.1f} Go VRAM")
        if p.total_memory / 1024**3 < 15:
            print("  Attention : < 16 Go VRAM -> utiliser fp16 + enable_model_cpu_offload()")
    else:
        print("  Aucun GPU CUDA détecté : l'analyse marche sur CPU, la génération SDXL sera très lente.")


def test_depth(image_path: str | None) -> None:
    from PIL import Image

    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
    from btl_studio.scene_analysis import DepthEstimator

    if image_path:
        img = Image.open(image_path)
    else:  # image synthétique : dégradé sol/mur
        import numpy as np
        arr = np.zeros((512, 768, 3), dtype=np.uint8)
        arr[:300] = (225, 220, 210)
        arr[300:] = np.linspace(120, 200, 212, dtype=np.uint8)[:, None, None]
        img = Image.fromarray(arr)

    t = time.time()
    res = DepthEstimator()(img)
    out = Path("outputs")
    out.mkdir(exist_ok=True)
    res.depth_image.save(out / "check_depth.png")
    print(f"  Profondeur calculée en {time.time() - t:.1f} s -> outputs/check_depth.png")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", default=None)
    ap.add_argument("--skip-model", action="store_true")
    args = ap.parse_args()

    print(f"Python {sys.version.split()[0]}\n\n1. Dépendances")
    ok = check_packages()
    print("\n2. Matériel")
    check_gpu()
    if not args.skip_model and ok:
        print("\n3. Test Depth Anything V2")
        test_depth(args.image)
    print("\nEnvironnement prêt." if ok else "\nInstaller les paquets manquants : pip install -r requirements.txt")


if __name__ == "__main__":
    main()
