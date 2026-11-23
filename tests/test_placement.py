from PIL import Image

from btl_studio.placement import place_furniture


def test_place_furniture_bottom_aligned():
    scene = Image.new("RGB", (800, 600), (200, 200, 200))
    item = Image.new("RGBA", (200, 100), (120, 60, 30, 255))
    p = place_furniture(scene, item, box=(100, 300, 500, 550))
    x0, y0, x1, y1 = p.box
    assert y1 == 550                        # posé sur le bas de la boîte (sol)
    assert abs((x1 - x0) / (y1 - y0) - 2.0) < 0.05   # proportions conservées
    assert p.mask.getbbox() is not None
    assert p.composite.size == scene.size
