"""Generate the charts used in the project report, straight from the live database.

    python scripts/report_figures.py

Every number in these figures is read from data/chronicle.db at the moment the script
runs, so the report cannot drift away from the system it describes. Output goes to
docs/ChronicleHeritage_report_overleaf/images/.
"""
from __future__ import annotations

import collections
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import matplotlib.dates as mdates  # noqa: E402
from sqlalchemy import select  # noqa: E402

from config import settings  # noqa: E402
from core import database as db  # noqa: E402
from core.damage_reference import DAMAGE, canonical  # noqa: E402
from core.monuments import load_monuments  # noqa: E402

OUT = settings.ROOT / "docs" / "ChronicleHeritage_report_overleaf" / "images"
OUT.mkdir(parents=True, exist_ok=True)

INK, MUTED, LINE, PAPER = "#0B1F3A", "#5B6578", "#E3E6EC", "#FBFAF7"
plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 9,
    "axes.edgecolor": LINE, "axes.labelcolor": INK, "text.color": INK,
    "xtick.color": MUTED, "ytick.color": MUTED, "axes.titlecolor": INK,
    "figure.facecolor": "white", "axes.facecolor": "white", "savefig.facecolor": "white",
})


def _clean(ax, grid="x"):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.grid(axis=grid, color=LINE, lw=.8)
    ax.set_axisbelow(True)


def imagery_by_monument() -> None:
    """How many dated images each monument holds, split by source type."""
    with db.session() as s:
        rows = s.execute(select(db.ImageRecord.monument_slug, db.ImageRecord.kind)).all()
    per = collections.defaultdict(collections.Counter)
    for slug, kind in rows:
        per[slug][kind] += 1
    mons = sorted(load_monuments(), key=lambda m: -sum(per[m.slug].values()))
    names = [m.name for m in mons]
    sat = [per[m.slug]["satellite"] for m in mons]
    gnd = [per[m.slug]["ground"] for m in mons]

    fig, ax = plt.subplots(figsize=(7.2, 4.1))
    y = range(len(names))
    ax.barh(y, sat, color="#2B4A86", label="Satellite (Esri Wayback)")
    ax.barh(y, gnd, left=sat, color="#B07D2B", label="Ground (Wikimedia Commons)")
    for i, (a, b) in enumerate(zip(sat, gnd)):
        ax.text(a + b + .15, i, str(a + b), va="center", fontsize=8, color=MUTED)
    ax.set_yticks(list(y), names)
    ax.invert_yaxis()
    ax.set_xlabel("images on record")
    ax.legend(frameon=False, loc="lower right", fontsize=8)
    _clean(ax)
    fig.tight_layout()
    fig.savefig(OUT / "fig_imagery_by_monument.png", dpi=220)
    plt.close(fig)


def satellite_timeline() -> None:
    """Capture dates of the satellite imagery, as reported by Esri's metadata service."""
    with db.session() as s:
        rows = s.execute(select(db.ImageRecord.monument_slug, db.ImageRecord.acquired_on)
                         .where(db.ImageRecord.kind == "satellite",
                                db.ImageRecord.acquired_on.is_not(None))).all()
    per = collections.defaultdict(list)
    for slug, d in rows:
        per[slug].append(d)
    by_name = {m.slug: m.name for m in load_monuments()}
    order = sorted(per, key=lambda s: min(per[s]))

    fig, ax = plt.subplots(figsize=(7.2, 4.1))
    for i, slug in enumerate(order):
        ds = sorted(per[slug])
        ax.plot(ds, [i] * len(ds), "-", color=LINE, lw=1.2, zorder=1)
        ax.scatter(ds, [i] * len(ds), s=26, color="#2B4A86", zorder=2)
    ax.set_yticks(range(len(order)), [by_name[s] for s in order])
    ax.invert_yaxis()
    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.set_xlabel("capture date reported by the Esri imagery-metadata service")
    _clean(ax, grid="both")
    fig.tight_layout()
    fig.savefig(OUT / "fig_satellite_timeline.png", dpi=220)
    plt.close(fig)


def damage_distribution() -> None:
    """Findings per damage category, and which of the seven have no example yet."""
    stats = db.global_stats()
    labels = list(DAMAGE)
    vals = [stats["by_class"].get(canonical(k), 0) for k in labels]
    fig, ax = plt.subplots(figsize=(7.2, 3.1))
    ax.barh(range(len(labels)), vals, color=[DAMAGE[k]["color"] for k in labels])
    for i, v in enumerate(vals):
        ax.text(v + .08, i, str(v), va="center", fontsize=8, color=MUTED)
    ax.set_yticks(range(len(labels)), labels)
    ax.invert_yaxis()
    ax.set_xlim(0, max(vals + [1]) * 1.18)
    ax.set_xlabel("findings on record (all hand annotations; the detector is not trained yet)")
    _clean(ax)
    fig.tight_layout()
    fig.savefig(OUT / "fig_damage_distribution.png", dpi=220)
    plt.close(fig)


def main() -> int:
    db.init_db()
    imagery_by_monument()
    satellite_timeline()
    damage_distribution()
    print(f"Wrote figures to {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
