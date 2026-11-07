"""Core tests — run with:  python -m pytest -q tests
Uses a temporary SQLite database; never touches data/chronicle.db.
"""
import os
import sys
import tempfile
from pathlib import Path

_TMP = tempfile.mkdtemp()
os.environ["DATABASE_URL"] = f"sqlite:///{Path(_TMP, 'test.db').as_posix()}"
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np  # noqa: E402
from PIL import Image, ImageDraw  # noqa: E402

from config import settings  # noqa: E402
from core import compare, database as db, detection, explain  # noqa: E402
from core.monuments import by_slug, condition_of, load_monuments  # noqa: E402

db.init_db()


def test_fourteen_monuments_with_required_fields():
    mons = load_monuments()
    assert len(mons) == 14
    assert len({m.slug for m in mons}) == 14
    for m in mons:
        assert m.name and m.history and -90 <= m.lat <= 90 and -180 <= m.lon <= 180


def test_class_names_are_canonicalised():
    assert detection.canonical("discoloration") == "Discolouration"
    assert detection.canonical("fabric_loss_of_masonry_units") == "Fabric Loss of Masonry Units"
    assert set(map(detection.canonical, settings.DAMAGE_CLASSES)) == set(settings.DAMAGE_CLASSES)


def test_missing_weights_raise_clear_error(monkeypatch):
    monkeypatch.setattr(settings, "YOLO_WEIGHTS", Path(_TMP) / "nope.pt")
    try:
        detection.get_model()
        assert False, "should raise"
    except detection.DetectorUnavailable as e:
        assert "nope.pt" in str(e)


def test_stats_come_from_database():
    s0 = db.global_stats()
    img = db.add_image(monument_slug="charminar", kind="ground", source="test", path="x.jpg")
    db.save_run(img.id, "charminar", "YOLO26l", "abc", 0.25, "south", "",
                [{"label": "Crack", "confidence": 0.8, "box": [0, 0, 10, 10], "u": .1, "v": .1}])
    s1 = db.global_stats()
    assert s1["detections"] == s0["detections"] + 1
    assert s1["by_class"]["Crack"] >= 1
    assert condition_of(s1["current"]["charminar"])[1] == "alert"
    assert condition_of(None)[1] == "idle"


def _scene(shift=0, add_block=False):
    rng = np.random.default_rng(0)
    a = (rng.random((400, 600, 3)) * 255).astype(np.uint8)
    im = Image.fromarray(a).filter(__import__("PIL.ImageFilter").ImageFilter.GaussianBlur(1))
    if add_block:
        ImageDraw.Draw(im).rectangle([250, 150, 340, 230], fill=(20, 20, 20))
    return im.transform(im.size, Image.AFFINE, (1, 0, shift, 0, 1, 0))


def test_identical_images_show_no_change():
    r = compare.compare(_scene(), _scene())
    assert r["ssim"] > 0.95 and r["changed_pct"] < 1


def test_shifted_image_is_registered_and_change_found():
    r = compare.compare(_scene(), _scene(shift=12, add_block=True))
    assert r["aligned"]
    assert len(r["regions"]) >= 1


def test_explanation_falls_back_to_reference_without_key(monkeypatch):
    monkeypatch.setattr(settings, "GEMINI_API_KEY", "")
    r = explain.explain(by_slug("charminar"), "Moss", 2, 0.7, "ground", "unknown", refresh=True)
    assert r["source"] == "reference" and "Possibly" in r["text"]


def test_every_monument_has_a_valid_glb():
    import json
    import struct
    for m in load_monuments():
        assert m.has_3d, f"missing 3D model for {m.slug}"
        b = m.glb_file.read_bytes()
        assert b[:4] == b"glTF"
        n = struct.unpack("<I", b[12:16])[0]
        js = json.loads(b[20:20 + n])
        assert js.get("meshes"), m.slug
        assert js["asset"].get("extras", {}).get("title"), f"{m.slug} has no provenance"
