"""Monument management: one registry for all 14 monuments.

All monument metadata lives in data/monuments.json so that every page, the RAG
knowledge base and the reports read the same facts. Adding a monument = adding
one JSON object (+ optionally static/models3d/<slug>.glb).
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from functools import lru_cache

from config import settings


@dataclass(frozen=True)
class Monument:
    slug: str
    name: str
    city: str
    state: str
    lat: float
    lon: float
    wikipedia_title: str
    built_by: str = ""
    construction_period: str = ""
    architectural_style: str = ""
    materials: tuple = ()
    unesco: str = ""
    tagline: str = ""
    summary: str = ""
    history: str = ""
    architecture: str = ""
    significance: str = ""
    known_conservation_issues: str = ""
    facades: tuple = field(default_factory=tuple)

    @property
    def location(self) -> str:
        return f"{self.city}, {self.state}" if self.state and self.state not in self.city else self.city

    @property
    def glb_file(self):
        return settings.MODEL3D_DIR / f"{self.slug}.glb"

    @property
    def has_3d(self) -> bool:
        return self.glb_file.exists() and self.glb_file.stat().st_size > 1000

    @property
    def glb_url(self) -> str | None:
        return f"/{settings.STATIC_URL_PREFIX}/models3d/{self.slug}.glb" if self.has_3d else None

    def facade_label(self, fid: str) -> str:
        for f in self.facades:
            if f.get("id") == fid:
                return f.get("label", fid.title())
        return fid.title()


@lru_cache(maxsize=1)
def load_monuments() -> tuple[Monument, ...]:
    raw = json.loads(settings.MONUMENTS_FILE.read_text(encoding="utf-8"))
    out = []
    for m in raw:
        m = dict(m)
        m["materials"] = tuple(m.get("materials", []))
        m["facades"] = tuple(m.get("facades", []))
        known = {k: m[k] for k in Monument.__dataclass_fields__ if k in m}
        out.append(Monument(**known))
    return tuple(out)


def by_slug(slug: str) -> Monument | None:
    return next((m for m in load_monuments() if m.slug == slug), None)


def condition_of(current: dict | None) -> tuple[str, str]:
    """(status label, css tone) derived from the latest stored detection run.

    Status reflects *what the model found*, not structural severity — a box's
    confidence says how sure the detector is, not how bad the damage is.
    """
    if not current:
        return "Awaiting first scan", "idle"
    n = current["count"]
    serious = {"Crack", "Fabric Loss of Masonry Units", "Surface Dissolution"}
    if n == 0:
        return "No damage detected", "ok"
    if serious & set(current["labels"]):
        return "Needs attention", "alert"
    return "Under observation", "watch"
