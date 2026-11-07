"""Populate the image archive for all 14 monuments.

    python scripts/fetch_imagery.py                 # covers + satellite series + Commons photos
    python scripts/fetch_imagery.py --only charminar --sat 6 --ground 6

* Cover photo  : Wikipedia lead image (Wikimedia Commons) → data/images/<slug>/cover.jpg,
                 web-sized copy → static/covers/<slug>.jpg (used by the cards)
* Satellite    : visually distinct Esri Wayback releases, each with Esri's capture date
* Ground photos: Wikimedia Commons search results with stated capture dates
Already-stored images are skipped, so the script is safe to re-run.
"""
from __future__ import annotations

import argparse
import json
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from PIL import Image  # noqa: E402

from config import settings  # noqa: E402
from core import acquisition, database as db  # noqa: E402
from core.monuments import load_monuments  # noqa: E402


def one(m, n_sat: int, n_ground: int) -> str:
    msg = [m.name]
    try:
        cover = acquisition.fetch_cover(m)
        if cover:
            out = settings.STATIC_DIR / "covers" / f"{m.slug}.jpg"
            out.parent.mkdir(parents=True, exist_ok=True)
            img = Image.open(cover).convert("RGB")
            img.thumbnail((1600, 1600))
            img.save(out, quality=84)
            msg.append("cover ✓")
    except Exception as e:
        msg.append(f"cover ✗ ({e})")
    try:
        msg.append(f"satellite +{acquisition.fetch_wayback_series(m, n_sat)}")
    except Exception as e:
        msg.append(f"satellite ✗ ({e})")
    try:
        msg.append(f"ground +{acquisition.fetch_commons_gallery(m, n_ground)}")
    except Exception as e:
        msg.append(f"ground ✗ ({e})")
    # Fallback cover: no Wikipedia lead image → use the first stored Commons photo (with its credit)
    out = settings.STATIC_DIR / "covers" / f"{m.slug}.jpg"
    if not out.exists():
        ground = [i for i in db.list_images(m.slug) if i.kind == "ground" and Path(i.path).exists()]
        if ground:
            g = ground[0]
            out.parent.mkdir(parents=True, exist_ok=True)
            img = Image.open(g.path).convert("RGB")
            img.thumbnail((1600, 1600))
            img.save(out, quality=84)
            img.save(settings.IMAGE_DIR / m.slug / "cover.jpg", quality=88)
            (settings.IMAGE_DIR / m.slug / "cover.json").write_text(json.dumps(
                {"title": g.external_id, "url": g.source_url, "author": g.attribution, "license": g.license}),
                encoding="utf-8")
            msg.append("cover ✓ (from Commons gallery)")
    return " · ".join(msg)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", help="monument slug")
    ap.add_argument("--sat", type=int, default=4)
    ap.add_argument("--ground", type=int, default=4)
    a = ap.parse_args()
    db.init_db()
    mons = [m for m in load_monuments() if not a.only or m.slug == a.only]
    with ThreadPoolExecutor(max_workers=4) as ex:
        for line in ex.map(lambda m: one(m, a.sat, a.ground), mons):
            print(line, flush=True)


if __name__ == "__main__":
    main()
