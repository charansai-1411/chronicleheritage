"""Central configuration. Every path, model name and secret is read from here.

Values come from environment variables (loaded from `.env`), with safe defaults.
Never hardcode API keys anywhere else in the code base.
"""
from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
load_dotenv(ROOT / ".env")


def _path(env: str, default: str) -> Path:
    p = Path(os.getenv(env, default))
    return p if p.is_absolute() else ROOT / p


# ---- Data & storage -------------------------------------------------------
DATA_DIR = _path("CH_DATA_DIR", "data")
IMAGE_DIR = DATA_DIR / "images"            # data/images/<slug>/<file>
UPLOAD_DIR = DATA_DIR / "uploads"          # user uploads, per monument
RESULT_DIR = DATA_DIR / "results"          # annotated detection images
CHROMA_DIR = _path("CH_CHROMA_DIR", "data/chroma")
REPORT_DIR = _path("CH_REPORT_DIR", "reports")
MONUMENTS_FILE = DATA_DIR / "monuments.json"
DAMAGE_REFERENCE_FILE = DATA_DIR / "damage_reference.json"
DATABASE_URL = os.getenv("DATABASE_URL", f"sqlite:///{(DATA_DIR / 'chronicle.db').as_posix()}")

# ---- 3D models --------------------------------------------------------------
# GLB files live in static/models3d/<slug>.glb and are served by Streamlit's
# static file server at /app/static/models3d/<slug>.glb (see .streamlit/config.toml).
STATIC_DIR = ROOT / "static"
MODEL3D_DIR = STATIC_DIR / "models3d"
STATIC_URL_PREFIX = "app/static"

# ---- YOLO26l damage detector -------------------------------------------------
YOLO_WEIGHTS = _path("YOLO_WEIGHTS", "models/weights/yolo26l_heritage.pt")
YOLO_IMGSZ = int(os.getenv("YOLO_IMGSZ", "640"))
YOLO_DEVICE = os.getenv("YOLO_DEVICE", "")  # "" = auto (GPU if available), "cpu", "0"
DAMAGE_CLASSES = [
    "Crack",
    "Discolouration",
    "Efflorescence",
    "Fabric Loss of Masonry Units",
    "Moss",
    "Plant",
    "Surface Dissolution",
]

# ---- Gemini / RAG ------------------------------------------------------------
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "").strip()
# Comma-separated list; the first model that answers is used.
GEMINI_MODELS = [m.strip() for m in os.getenv("GEMINI_MODEL", "gemini-2.5-flash,gemini-2.0-flash").split(",") if m.strip()]
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")

# ---- Imagery -----------------------------------------------------------------
WAYBACK_CONFIG_URL = "https://s3-us-west-2.amazonaws.com/config.maptiles.arcgis.com/waybackconfig.json"
WAYBACK_ZOOM = int(os.getenv("WAYBACK_ZOOM", "18"))
HTTP_USER_AGENT = "ChronicleHeritage/1.0 (academic heritage-monitoring project)"

for d in (DATA_DIR, IMAGE_DIR, UPLOAD_DIR, RESULT_DIR, CHROMA_DIR, REPORT_DIR, MODEL3D_DIR):
    d.mkdir(parents=True, exist_ok=True)


def to_rel(p) -> str:
    """Store file paths relative to the project root so the database is portable (Windows ↔ Linux cloud)."""
    if not p:
        return ""
    P = Path(str(p).replace("\\", "/"))
    try:
        return P.resolve().relative_to(ROOT.resolve()).as_posix()
    except ValueError:
        s = P.as_posix()
        i = s.find("chronicleheritage/")
        return s[i + len("chronicleheritage/"):] if i >= 0 else s


def to_abs(p) -> str:
    if not p:
        return ""
    P = Path(p)
    return str(P if P.is_absolute() else ROOT / P)


def gemini_available() -> bool:
    return bool(GEMINI_API_KEY)


def weights_available() -> bool:
    return YOLO_WEIGHTS.exists()
