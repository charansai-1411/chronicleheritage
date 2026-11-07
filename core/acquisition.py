"""Image acquisition from legitimate, attributable sources.

Sources
-------
* Ground-level photographs: Wikimedia Commons API (free licences, with the
  photographer's stated capture date where available).
* Satellite: Esri World Imagery *Wayback* archive — dated historical releases
  of the World Imagery basemap. For each release we also query Esri's
  imagery-metadata service for the real capture date (SRC_DATE) and source
  resolution of the pixels covering the monument.
* Drone / field photos: user upload (the user states date + source).

Nothing here pretends to be real-time: every stored image carries its actual
acquisition date (or an explicit note that the date is unknown) and source.
"""
from __future__ import annotations

import hashlib
import io
import json
import math
import re
from datetime import datetime
from pathlib import Path

import requests
from PIL import Image

from config import settings
from core import database as db
from core.monuments import Monument

from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

S = requests.Session()
S.headers["User-Agent"] = settings.HTTP_USER_AGENT
_retry = HTTPAdapter(max_retries=Retry(total=5, connect=5, read=5, backoff_factor=0.6,
                                       status_forcelist=(429, 500, 502, 503, 504), allowed_methods=("GET",)))
S.mount("https://", _retry)
S.mount("http://", _retry)
COMMONS_API = "https://commons.wikimedia.org/w/api.php"
WIKI_API = "https://en.wikipedia.org/w/api.php"


def _slug_dir(slug: str) -> Path:
    d = settings.IMAGE_DIR / slug
    d.mkdir(parents=True, exist_ok=True)
    return d


def _strip_html(s: str) -> str:
    return re.sub(r"<[^>]+>", "", s or "").strip()


def _parse_date(s: str) -> datetime | None:
    s = _strip_html(s)
    m = re.search(r"(\d{4})[-:/](\d{2})[-:/](\d{2})", s)
    if m:
        try:
            return datetime(int(m[1]), int(m[2]), int(m[3]))
        except ValueError:
            return None
    return None


# =========================================================== Wikimedia Commons
def _commons_info(titles: list[str]) -> list[dict]:
    r = S.get(COMMONS_API, params={
        "action": "query", "format": "json", "titles": "|".join(titles), "prop": "imageinfo",
        "iiprop": "url|extmetadata|size|mime|timestamp", "iiurlwidth": 1600}, timeout=30)
    r.raise_for_status()
    pages = r.json().get("query", {}).get("pages", {})
    out = []
    for p in pages.values():
        ii = (p.get("imageinfo") or [None])[0]
        if ii:
            out.append({"title": p["title"], **ii})
    return out


def _cover_title(m: Monument) -> str | None:
    r = S.get(WIKI_API, params={"action": "query", "format": "json", "titles": m.wikipedia_title,
                                "prop": "pageimages", "piprop": "name", "redirects": 1}, timeout=30)
    r.raise_for_status()
    for p in r.json().get("query", {}).get("pages", {}).values():
        if p.get("pageimage"):
            return "File:" + p["pageimage"]
    return None


def _search_commons(query: str, limit: int) -> list[str]:
    r = S.get(COMMONS_API, params={
        "action": "query", "format": "json", "list": "search", "srnamespace": 6,
        "srsearch": f"{query} filetype:bitmap", "srlimit": limit * 3}, timeout=30)
    r.raise_for_status()
    bad = re.compile(r"map|plan|diagram|logo|stamp|coin|satellite|drawing|sketch|painting|svg", re.I)
    return [h["title"] for h in r.json().get("query", {}).get("search", []) if not bad.search(h["title"])]


def _store_commons(m: Monument, info: dict, filename: str | None = None) -> db.ImageRecord | None:
    if info.get("mime") not in ("image/jpeg", "image/png") or info.get("width", 0) < 800:
        return None
    meta = info.get("extmetadata", {})
    url = info.get("thumburl") or info["url"]
    raw = S.get(url, timeout=60)
    raw.raise_for_status()
    img = Image.open(io.BytesIO(raw.content)).convert("RGB")
    name = filename or ("commons_" + hashlib.sha1(info["title"].encode()).hexdigest()[:12] + ".jpg")
    path = _slug_dir(m.slug) / name
    img.save(path, quality=90)
    captured = _parse_date(meta.get("DateTimeOriginal", {}).get("value", ""))
    upload = info.get("timestamp", "")[:10]
    note = ("Capture date as stated by the photographer on Wikimedia Commons" if captured
            else f"Capture date not stated on Commons (file uploaded {upload})")
    return db.add_image(
        monument_slug=m.slug, kind="ground", source="Wikimedia Commons", acquired_on=captured,
        date_note=note, path=str(path), source_url=info.get("descriptionurl", info["url"]),
        attribution=_strip_html(meta.get("Artist", {}).get("value", ""))[:200] or "Unknown author",
        license=_strip_html(meta.get("LicenseShortName", {}).get("value", ""))[:120],
        external_id=info["title"])


def fetch_cover(m: Monument) -> Path | None:
    """Download the monument's Wikipedia lead photograph as data/images/<slug>/cover.jpg."""
    cover = _slug_dir(m.slug) / "cover.jpg"
    if cover.exists():
        return cover
    title = _cover_title(m)
    if not title:
        return None
    infos = _commons_info([title])
    if not infos:
        return None
    info = infos[0]
    url = info.get("thumburl") or info["url"]
    img = Image.open(io.BytesIO(S.get(url, timeout=60).content)).convert("RGB")
    img.save(cover, quality=88)
    meta = info.get("extmetadata", {})
    (cover.with_suffix(".json")).write_text(json.dumps({
        "title": info["title"], "url": info.get("descriptionurl"),
        "author": _strip_html(meta.get("Artist", {}).get("value", ""))[:200],
        "license": _strip_html(meta.get("LicenseShortName", {}).get("value", ""))}, indent=2), encoding="utf-8")
    _store_commons(m, info, filename="commons_cover.jpg")
    return cover


def fetch_commons_gallery(m: Monument, limit: int = 6) -> int:
    titles = _search_commons(m.name, limit)
    added = 0
    for chunk in [titles[i:i + 10] for i in range(0, len(titles), 10)]:
        for info in _commons_info(chunk):
            if added >= limit:
                return added
            try:
                if _store_commons(m, info):
                    added += 1
            except Exception:
                continue
    return added


def cover_credit(slug: str) -> dict:
    p = settings.IMAGE_DIR / slug / "cover.json"
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else {}


# =========================================================== Esri Wayback (satellite)
def _deg2tile(lat: float, lon: float, z: int) -> tuple[float, float]:
    n = 2 ** z
    x = (lon + 180.0) / 360.0 * n
    y = (1.0 - math.asinh(math.tan(math.radians(lat))) / math.pi) / 2.0 * n
    return x, y


def wayback_releases() -> list[dict]:
    cfg = S.get(settings.WAYBACK_CONFIG_URL, timeout=30).json()
    rel = []
    for k, v in cfg.items():
        m = re.search(r"(\d{4}-\d{2}-\d{2})", v["itemTitle"])
        if m:
            rel.append({"id": int(k), "date": datetime.strptime(m[1], "%Y-%m-%d"),
                        "tile": v["itemURL"], "meta": v.get("metadataLayerUrl", "")})
    return sorted(rel, key=lambda r: r["date"])


def _capture_meta(meta_url: str, lat: float, lon: float) -> dict | None:
    """Ask Esri's metadata service for the capture date of the pixels at (lat, lon)."""
    if not meta_url:
        return None
    for layer in range(0, 8):  # finest resolution layer first
        try:
            r = S.get(f"{meta_url}/{layer}/query", params={
                "geometry": f"{lon},{lat}", "geometryType": "esriGeometryPoint", "inSR": 4326,
                "spatialRel": "esriSpatialRelIntersects", "outFields": "SRC_DATE,SRC_RES,NICE_DESC,NICE_NAME",
                "returnGeometry": "false", "f": "json"}, timeout=20)
            feats = r.json().get("features") or []
        except Exception:
            continue
        if feats:
            a = feats[0]["attributes"]
            d = str(a.get("SRC_DATE") or "")
            date = datetime.strptime(d, "%Y%m%d") if re.fullmatch(r"\d{8}", d) else None
            return {"date": date, "res": a.get("SRC_RES"), "provider": a.get("NICE_DESC") or "",
                    "product": a.get("NICE_NAME") or ""}
    return None


def _stitch(tile_tpl: str, lat: float, lon: float, z: int, n: int = 5) -> Image.Image | None:
    fx, fy = _deg2tile(lat, lon, z)
    x0, y0 = int(fx) - n // 2, int(fy) - n // 2
    canvas = Image.new("RGB", (256 * n, 256 * n))
    for i in range(n):
        for j in range(n):
            url = tile_tpl.replace("{level}", str(z)).replace("{row}", str(y0 + j)).replace("{col}", str(x0 + i))
            r = S.get(url, timeout=30)
            if r.status_code != 200 or len(r.content) < 3000:   # missing tile placeholder
                return None
            canvas.paste(Image.open(io.BytesIO(r.content)).convert("RGB"), (256 * i, 256 * j))
    # crop a 3x3-tile window centred exactly on the monument coordinate
    side = 256 * (n - 2)
    cx, cy = (fx - x0) * 256, (fy - y0) * 256
    return canvas.crop((int(cx - side / 2), int(cy - side / 2), int(cx + side / 2), int(cy + side / 2)))


def fetch_wayback_series(m: Monument, max_versions: int = 5) -> int:
    """Store up to `max_versions` *visually distinct* dated satellite views of the monument."""
    releases = wayback_releases()
    z = settings.WAYBACK_ZOOM
    seen_hash, seen_capture, added = set(), set(), 0
    for rel in reversed(releases):          # newest first
        if added >= max_versions:
            break
        fx, fy = _deg2tile(m.lat, m.lon, z)
        probe = rel["tile"].replace("{level}", str(z)).replace("{row}", str(int(fy))).replace("{col}", str(int(fx)))
        try:
            r = S.get(probe, timeout=20)
        except Exception:
            continue
        if r.status_code != 200 or len(r.content) < 3000:
            continue
        h = hashlib.md5(r.content).hexdigest()
        if h in seen_hash:
            continue
        seen_hash.add(h)
        meta = _capture_meta(rel["meta"], m.lat, m.lon)
        cap_key = (meta or {}).get("date")
        if cap_key and cap_key in seen_capture:
            continue
        img = _stitch(rel["tile"], m.lat, m.lon, z)
        if img is None:
            continue
        if cap_key:
            seen_capture.add(cap_key)
        path = _slug_dir(m.slug) / f"wayback_{rel['id']}.jpg"
        img.save(path, quality=92)
        if meta and meta["date"]:
            note = (f"Capture date from Esri imagery metadata ({meta['provider']} {meta['product']}, "
                    f"~{meta['res']} m/px); shown in Wayback release {rel['date']:%Y-%m-%d}")
            acquired = meta["date"]
        else:
            note = f"Capture date unavailable; Wayback release date {rel['date']:%Y-%m-%d} (capture is earlier)"
            acquired = None
        db.add_image(monument_slug=m.slug, kind="satellite", source="Esri World Imagery Wayback",
                     acquired_on=acquired, date_note=note, path=str(path),
                     source_url=f"https://livingatlas.arcgis.com/wayback/#active={rel['id']}",
                     attribution=f"Esri, {(meta or {}).get('provider') or 'imagery providers'}",
                     license="Esri Master License Agreement (non-commercial academic use)",
                     facade="top", external_id=f"wayback:{rel['id']}")
        added += 1
    return added


# =========================================================== Uploads (drone / field)
def store_upload(m: Monument, file_bytes: bytes, filename: str, kind: str, acquired_on: datetime | None,
                 source: str, facade: str) -> db.ImageRecord:
    img = Image.open(io.BytesIO(file_bytes)).convert("RGB")
    digest = hashlib.sha1(file_bytes).hexdigest()[:16]
    d = settings.UPLOAD_DIR / m.slug
    d.mkdir(parents=True, exist_ok=True)
    path = d / f"{digest}.jpg"
    img.save(path, quality=92)
    exif_date = None
    try:
        raw = Image.open(io.BytesIO(file_bytes)).getexif().get(306)  # DateTime
        exif_date = _parse_date(str(raw)) if raw else None
    except Exception:
        pass
    date = acquired_on or exif_date
    note = ("Date entered by user" if acquired_on else "Date read from EXIF" if exif_date
            else "Acquisition date not provided")
    return db.add_image(monument_slug=m.slug, kind=kind, source=source or "User upload", acquired_on=date,
                        date_note=note, path=str(path), attribution="Uploaded by user", license="",
                        facade=facade, external_id=f"upload:{digest}")
