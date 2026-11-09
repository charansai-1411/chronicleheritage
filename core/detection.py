"""YOLO26l heritage-damage detection.

Integration point
-----------------
Put the trained weights at   models/weights/yolo26l_heritage.pt
(or set YOLO_WEIGHTS in .env). The file is the `best.pt` produced by
`yolo detect train model=yolo26l.pt data=heritage.yaml ...` on the 6,317-image,
7-class heritage dataset.

If the file is missing, `DetectorUnavailable` is raised and the UI explains
which file is needed — no placeholder predictions are ever produced.
"""
from __future__ import annotations

import hashlib
from functools import lru_cache
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from config import settings
from core.damage_reference import canonical, color, is_positional  # canonical re-exported for callers


class DetectorUnavailable(RuntimeError):
    pass


def raw_names() -> dict:
    """The class names exactly as the loaded model stores them (no normalisation)."""
    names = get_model().names
    return dict(names) if isinstance(names, dict) else {i: n for i, n in enumerate(names)}


def names_are_positional() -> bool:
    """True when the weights name their classes "0".."6" instead of the damage names.

    Then every label depends on the positional fallback in `canonical()`, which is only
    right if the training class order matches settings.DAMAGE_CLASSES. The UI says so.
    """
    return any(is_positional(n) for n in raw_names().values())


@lru_cache(maxsize=1)
def _load(weights: str, mtime: float):
    from ultralytics import YOLO
    return YOLO(weights)


def weights_sha() -> str:
    h = hashlib.sha256()
    with open(settings.YOLO_WEIGHTS, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()[:16]


def get_model():
    w: Path = settings.YOLO_WEIGHTS
    if not w.exists():
        raise DetectorUnavailable(
            f"Trained YOLO26l weights not found at '{w.relative_to(settings.ROOT) if w.is_relative_to(settings.ROOT) else w}'. "
            "Copy your best.pt there (or set YOLO_WEIGHTS in .env) to enable detection.")
    return _load(str(w), w.stat().st_mtime)


def model_classes() -> list[str]:
    names = raw_names()
    # a positionally-named model carries no name information, so map by index
    return [canonical(i if is_positional(names[i]) else names[i]) for i in sorted(names)]


def detect(image: Image.Image, conf: float = 0.25, iou: float = 0.5) -> list[dict]:
    """Run real inference. Returns [{label, confidence, box[x1,y1,x2,y2], u, v}] in pixel coords."""
    model = get_model()
    image = image.convert("RGB")
    W, H = image.size
    res = model.predict(source=np.array(image)[:, :, ::-1].copy(), imgsz=settings.YOLO_IMGSZ, conf=conf, iou=iou,
                        device=settings.YOLO_DEVICE or None, verbose=False)[0]
    names = res.names if isinstance(res.names, dict) else dict(enumerate(res.names))
    out = []
    if res.boxes is not None:
        for b in res.boxes:
            cid = int(b.cls[0])
            raw = names.get(cid, cid)
            # a numeric "name" carries no information — fall back to the class id itself
            label = canonical(cid if is_positional(raw) else raw)
            x1, y1, x2, y2 = [float(v) for v in b.xyxy[0].tolist()]
            out.append({"label": label, "confidence": float(b.conf[0]),
                        "box": [x1, y1, x2, y2], "u": ((x1 + x2) / 2) / W, "v": ((y1 + y2) / 2) / H})
    return sorted(out, key=lambda d: -d["confidence"])


def annotate(image: Image.Image, dets: list[dict]) -> Image.Image:
    out = image.convert("RGB").copy()
    dr = ImageDraw.Draw(out)
    w = max(2, out.width // 400)
    try:
        font = ImageFont.truetype("arial.ttf", max(14, out.width // 70))
    except OSError:
        font = ImageFont.load_default()
    for i, d in enumerate(dets, 1):
        c = color(d["label"])
        x1, y1, x2, y2 = d["box"]
        dr.rectangle([x1, y1, x2, y2], outline=c, width=w)
        tag = f" {i} · {d['label']} {d['confidence']:.0%} "
        tb = dr.textbbox((x1, y1), tag, font=font)
        th = tb[3] - tb[1] + 6
        ty = y1 - th if y1 - th > 0 else y1
        dr.rectangle([x1, ty, x1 + (tb[2] - tb[0]) + 4, ty + th], fill=c)
        dr.text((x1 + 2, ty + 2), tag, fill="white", font=font)
    return out
