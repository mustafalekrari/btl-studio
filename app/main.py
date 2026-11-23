"""API BTL Studio (FastAPI). Lancer : uvicorn app.main:app --reload"""
from __future__ import annotations

import io

from fastapi import FastAPI, File, UploadFile
from fastapi.responses import StreamingResponse
from PIL import Image

from btl_studio import __version__

app = FastAPI(title="BTL Studio API", version=__version__)
_depth = None


@app.get("/health")
def health():
    return {"status": "ok", "version": __version__}


@app.post("/depth")
async def depth(scene: UploadFile = File(...)):
    """Carte de profondeur d'une photo de pièce (étape 1 du pipeline)."""
    global _depth
    if _depth is None:
        from btl_studio.scene_analysis import DepthEstimator
        _depth = DepthEstimator()
    img = Image.open(io.BytesIO(await scene.read()))
    buf = io.BytesIO()
    _depth(img).depth_image.save(buf, format="PNG")
    buf.seek(0)
    return StreamingResponse(buf, media_type="image/png")
