"""Hand-marked reference annotations — the stand-in while the detector is being trained.

Why this module exists
----------------------
ChronicleHeritage never invents model output. Until `models/weights/yolo26l_heritage.pt`
exists, section 03 has nothing to show, and everything downstream of it (the 3D twin,
Gemini's explanations, the PDF report) has nothing to work on either.

So instead of faking predictions, the team marks damage by eye on real photographs and
stores those boxes as what they are: *hand annotations*. They are written to the same
tables as a detection run, but with `is_demo = True` and `model_name` set to
`REFERENCE_MODEL`, and every surface that displays them says "hand annotation, not model
output". The rendered image carries the same statement burnt into the picture, so a
screenshot cannot be mistaken for a detection result either.

They also give the real detector something to be checked against: once the weights land,
run the model on the same images and compare.

    python scripts/seed_annotations.py            # load / refresh them
    python scripts/seed_annotations.py --clear    # remove them again
"""
from __future__ import annotations

import json
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

from config import settings
from core.damage_reference import color

SPEC_FILE = settings.DATA_DIR / "annotations.json"

REFERENCE_MODEL = "Hand annotation (not a model prediction)"
BANNER = "HAND-ANNOTATED REFERENCE  ·  NOT MODEL OUTPUT  ·  DETECTOR NOT YET TRAINED"
SHORT_NOTICE = ("Marked by eye by the project team on the original photograph. "
                "No detector produced these boxes.")


def is_reference(run: dict | None) -> bool:
    """True when a run dict from the database holds hand annotations, not inference."""
    return bool(run) and (run.get("is_demo") or run.get("model") == REFERENCE_MODEL)


def load_specs() -> list[dict]:
    if not SPEC_FILE.exists():
        return []
    return json.loads(SPEC_FILE.read_text(encoding="utf-8")).get("annotations", [])


def notes_for(slug: str) -> list[str]:
    """The per-box notes for a monument, in the order the boxes were stored.

    The notes live in the spec file rather than in a database column, so loading them
    needs no schema migration on an existing deployment.
    """
    spec = next((s for s in load_specs() if s["monument"] == slug), None)
    return [b.get("note", "") for b in spec["boxes"]] if spec else []


def spec_note(slug: str) -> str:
    spec = next((s for s in load_specs() if s["monument"] == slug), None)
    return spec.get("note", "") if spec else ""


def _font(size: int):
    for name in ("arialbd.ttf", "arial.ttf", "DejaVuSans-Bold.ttf"):
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            continue
    return ImageFont.load_default()


def render(image: Image.Image, dets: list[dict]) -> Image.Image:
    """Draw the annotation boxes in a style that cannot be mistaken for model output.

    Corner brackets rather than the solid boxes `detection.annotate` draws, no confidence
    percentages (there is no confidence in a hand mark), and a banner across the top.
    """
    out = image.convert("RGB").copy()
    dr = ImageDraw.Draw(out)
    W, H = out.size
    w = max(3, W // 300)
    arm = max(18, W // 36)
    font = _font(max(15, W // 68))

    for i, d in enumerate(dets, 1):
        c = color(d["label"])
        x1, y1, x2, y2 = d["box"]
        for cx, sx in ((x1, 1), (x2, -1)):
            for cy, sy in ((y1, 1), (y2, -1)):
                dr.line([(cx, cy), (cx + sx * arm, cy)], fill=c, width=w)
                dr.line([(cx, cy), (cx, cy + sy * arm)], fill=c, width=w)
        tag = f" {i} · {d['label']} "
        tb = dr.textbbox((x1, y1), tag, font=font)
        th = tb[3] - tb[1] + 8
        ty = y1 - th - w if y1 - th - w > 0 else y2 + w
        dr.rectangle([x1, ty, x1 + (tb[2] - tb[0]) + 6, ty + th], fill=c)
        dr.text((x1 + 3, ty + 3), tag, fill="white", font=font)

    bf = _font(max(16, W // 62))
    bh = (dr.textbbox((0, 0), BANNER, font=bf)[3]) + 20
    dr.rectangle([0, 0, W, bh], fill=(176, 58, 46))
    dr.text((14, 10), BANNER, fill="white", font=bf)
    return out


def _dets_from_spec(spec: dict, size: tuple[int, int]) -> list[dict]:
    W, H = size
    out = []
    for b in spec["boxes"]:
        x1, y1, x2, y2 = (b["box"][0] * W, b["box"][1] * H, b["box"][2] * W, b["box"][3] * H)
        out.append({"label": b["label"], "confidence": 0.0, "box": [x1, y1, x2, y2],
                    "u": ((x1 + x2) / 2) / W, "v": ((y1 + y2) / 2) / H, "note": b.get("note", "")})
    return out


def clear() -> int:
    """Remove every stored hand-annotation run. Real detection runs are untouched."""
    from sqlalchemy import select

    from core import database as db
    removed = 0
    with db.session() as s:
        for r in s.scalars(select(db.DetectionRun).where(db.DetectionRun.model_name == REFERENCE_MODEL)).all():
            s.delete(r)
            removed += 1
    return removed


def seed(log=print) -> int:
    """Load data/annotations.json into the database. Idempotent — old rows are replaced."""
    from core import database as db

    specs = load_specs()
    if not specs:
        log(f"No annotations found at {SPEC_FILE}")
        return 0
    log(f"Clearing {clear()} previous hand-annotation run(s)")

    stored = 0
    for spec in specs:
        slug = spec["monument"]
        src = settings.ROOT / spec["image"]
        if not src.exists():
            log(f"  !! {slug}: image not found — {spec['image']}")
            continue

        rec = next((i for i in db.list_images(slug) if Path(i.path).resolve() == src.resolve()), None)
        if rec is None:
            log(f"  !! {slug}: {src.name} is not in the images table (run scripts/fetch_imagery.py first)")
            continue

        img = Image.open(src).convert("RGB")
        dets = _dets_from_spec(spec, img.size)
        out_dir = settings.RESULT_DIR / slug
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"reference_{src.stem}.jpg"
        render(img, dets).save(out_path, quality=90)

        rid = db.save_run(rec.id, slug, REFERENCE_MODEL, "", 0.0, spec.get("facade", ""),
                          str(out_path), dets, is_demo=True)
        stored += 1
        log(f"  {slug}: run #{rid} · {len(dets)} hand annotation(s) · {out_path.name}")
    log(f"Stored {stored} reference run(s). They are labelled as hand annotations everywhere.")
    return stored
