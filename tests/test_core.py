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


def test_numeric_class_labels_map_to_damage_names():
    """Regression: a model whose classes are named "0".."6" produced KeyError: '6'."""
    from core.damage_reference import DAMAGE, canonical, color
    for i, expected in enumerate(settings.DAMAGE_CLASSES):
        assert canonical(str(i)) == expected
        assert canonical(i) == expected
    assert canonical("6") == "Surface Dissolution"
    # spelling variants
    assert canonical("discoloration") == "Discolouration"
    assert canonical("CRACKS") == "Crack"
    # out-of-range / unknown labels survive instead of crashing
    assert canonical("99") == "99"
    assert canonical("Rust") == "Rust"
    assert color("99") and color("Rust") and color("6") == DAMAGE["Surface Dissolution"]["color"]


def test_stored_numeric_labels_are_normalised_on_read():
    img = db.add_image(monument_slug="hampi", kind="ground", source="test", path="x.jpg")
    rid = db.save_run(img.id, "hampi", "YOLO26l", "abc", 0.25, "south", "",
                      [{"label": "6", "confidence": 0.5, "box": [0, 0, 1, 1], "u": .5, "v": .5}])
    run = next(r for r in db.runs_for("hampi") if r["id"] == rid)
    assert run["detections"][0]["label"] == "Surface Dissolution"
    assert run["detections"][0]["raw_label"] == "6"          # the row itself is untouched
    assert "6" not in db.global_stats()["by_class"]


def test_hand_annotations_are_valid_and_clearly_marked():
    """The reference annotations stand in for the detector, so they must be well-formed
    and must never be mistakable for model output."""
    from core import annotation
    specs = annotation.load_specs()
    assert specs, "data/annotations.json has no annotations"
    for spec in specs:
        assert by_slug(spec["monument"]), spec["monument"]
        assert (settings.ROOT / spec["image"]).exists(), spec["image"]
        assert spec["boxes"], spec["monument"]
        for b in spec["boxes"]:
            assert b["label"] in settings.DAMAGE_CLASSES, b["label"]
            x1, y1, x2, y2 = b["box"]
            assert 0 <= x1 < x2 <= 1 and 0 <= y1 < y2 <= 1, b
            assert b.get("note"), "every hand annotation says what was marked"
    assert "NOT MODEL OUTPUT" in annotation.BANNER
    assert annotation.is_reference({"is_demo": True})
    assert annotation.is_reference({"model": annotation.REFERENCE_MODEL})
    assert not annotation.is_reference({"is_demo": False, "model": "YOLO26l"})
    assert not annotation.is_reference(None)


def test_real_runs_are_never_flagged_as_reference():
    img = db.add_image(monument_slug="sanchi-stupa", kind="ground", source="test", path="x.jpg")
    rid = db.save_run(img.id, "sanchi-stupa", "YOLO26l", "abc", 0.25, "south", "",
                      [{"label": "Crack", "confidence": 0.9, "box": [0, 0, 5, 5], "u": .2, "v": .3}])
    run = next(r for r in db.runs_for("sanchi-stupa") if r["id"] == rid)
    from core import annotation
    assert run["is_demo"] is False and not annotation.is_reference(run)
    assert db.global_stats()["reference_detections"] == 0
