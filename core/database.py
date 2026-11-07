"""Database management (SQLAlchemy; SQLite by default, PostgreSQL/MySQL via DATABASE_URL).

Tables
------
images          every image we hold for a monument, with its *real* acquisition date & source
detection_runs  one row per YOLO inference run on an image (model + weights fingerprint)
detections      individual boxes from a run, plus the approximate 3D anchor
explanations    cached Gemini explanations (so we don't re-bill the API)
comparisons     date-wise image comparison results
"""
from __future__ import annotations

from contextlib import contextmanager
from datetime import datetime, timezone

from sqlalchemy import (JSON, Boolean, DateTime, Float, ForeignKey, Integer, String, Text,
                        create_engine, func, select)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship, sessionmaker

from config import settings

engine = create_engine(settings.DATABASE_URL, pool_pre_ping=True,
                       connect_args={"check_same_thread": False} if settings.DATABASE_URL.startswith("sqlite") else {})
SessionLocal = sessionmaker(bind=engine, expire_on_commit=False)


def utcnow() -> datetime:
    return datetime.now(timezone.utc).replace(tzinfo=None)


class Base(DeclarativeBase):
    pass


class ImageRecord(Base):
    __tablename__ = "images"
    id: Mapped[int] = mapped_column(primary_key=True)
    monument_slug: Mapped[str] = mapped_column(String(64), index=True)
    kind: Mapped[str] = mapped_column(String(16))          # satellite | drone | ground
    source: Mapped[str] = mapped_column(String(160))       # e.g. "Wikimedia Commons", "Esri Wayback"
    acquired_on: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    date_note: Mapped[str] = mapped_column(String(240), default="")  # how the date was obtained / its caveat
    path_rel: Mapped[str] = mapped_column("path", Text)          # stored relative to project root
    source_url: Mapped[str] = mapped_column(Text, default="")
    attribution: Mapped[str] = mapped_column(Text, default="")
    license: Mapped[str] = mapped_column(String(120), default="")
    facade: Mapped[str] = mapped_column(String(16), default="")   # north/east/south/west/top
    external_id: Mapped[str] = mapped_column(String(240), default="", index=True)
    added_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow)

    runs: Mapped[list["DetectionRun"]] = relationship(back_populates="image", cascade="all, delete-orphan")

    @property
    def path(self) -> str:
        return settings.to_abs(self.path_rel)

    @path.setter
    def path(self, value) -> None:
        self.path_rel = settings.to_rel(value)


class DetectionRun(Base):
    __tablename__ = "detection_runs"
    id: Mapped[int] = mapped_column(primary_key=True)
    image_id: Mapped[int] = mapped_column(ForeignKey("images.id", ondelete="CASCADE"), index=True)
    monument_slug: Mapped[str] = mapped_column(String(64), index=True)
    model_name: Mapped[str] = mapped_column(String(80))
    weights_sha: Mapped[str] = mapped_column(String(64))
    conf_threshold: Mapped[float] = mapped_column(Float)
    facade: Mapped[str] = mapped_column(String(16), default="")
    annotated_rel: Mapped[str] = mapped_column("annotated_path", Text, default="")
    is_demo: Mapped[bool] = mapped_column(Boolean, default=False)   # never true for real model output
    created_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow, index=True)

    image: Mapped[ImageRecord] = relationship(back_populates="runs")
    detections: Mapped[list["Detection"]] = relationship(back_populates="run", cascade="all, delete-orphan")

    @property
    def annotated_path(self) -> str:
        return settings.to_abs(self.annotated_rel)

    @annotated_path.setter
    def annotated_path(self, value) -> None:
        self.annotated_rel = settings.to_rel(value)


class Detection(Base):
    __tablename__ = "detections"
    id: Mapped[int] = mapped_column(primary_key=True)
    run_id: Mapped[int] = mapped_column(ForeignKey("detection_runs.id", ondelete="CASCADE"), index=True)
    monument_slug: Mapped[str] = mapped_column(String(64), index=True)
    label: Mapped[str] = mapped_column(String(64), index=True)
    confidence: Mapped[float] = mapped_column(Float)
    x1: Mapped[float] = mapped_column(Float)
    y1: Mapped[float] = mapped_column(Float)
    x2: Mapped[float] = mapped_column(Float)
    y2: Mapped[float] = mapped_column(Float)
    u: Mapped[float] = mapped_column(Float)      # box centre, 0..1 across image width
    v: Mapped[float] = mapped_column(Float)      # box centre, 0..1 down image height
    mapping: Mapped[str] = mapped_column(String(32), default="approximate-facade-projection")

    run: Mapped[DetectionRun] = relationship(back_populates="detections")


class Explanation(Base):
    __tablename__ = "explanations"
    id: Mapped[int] = mapped_column(primary_key=True)
    monument_slug: Mapped[str] = mapped_column(String(64), index=True)
    label: Mapped[str] = mapped_column(String(64))
    text: Mapped[str] = mapped_column(Text)
    model: Mapped[str] = mapped_column(String(80))
    created_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow)


class Comparison(Base):
    __tablename__ = "comparisons"
    id: Mapped[int] = mapped_column(primary_key=True)
    monument_slug: Mapped[str] = mapped_column(String(64), index=True)
    before_id: Mapped[int] = mapped_column(Integer)
    after_id: Mapped[int] = mapped_column(Integer)
    aligned: Mapped[bool] = mapped_column(Boolean)
    inliers: Mapped[int] = mapped_column(Integer, default=0)
    changed_pct: Mapped[float] = mapped_column(Float)
    ssim: Mapped[float] = mapped_column(Float)
    regions: Mapped[dict] = mapped_column(JSON, default=dict)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow)


def init_db() -> None:
    Base.metadata.create_all(engine)
    # migrate any absolute paths written by older versions to project-relative ones
    with session() as s:
        for r in s.scalars(select(ImageRecord)).all():
            rel = settings.to_rel(r.path_rel)
            if rel != r.path_rel:
                r.path_rel = rel
        for r in s.scalars(select(DetectionRun)).all():
            rel = settings.to_rel(r.annotated_rel)
            if rel != r.annotated_rel:
                r.annotated_rel = rel


@contextmanager
def session():
    s = SessionLocal()
    try:
        yield s
        s.commit()
    except Exception:
        s.rollback()
        raise
    finally:
        s.close()


# ---------------------------------------------------------------- images
def add_image(**kw) -> ImageRecord:
    with session() as s:
        if kw.get("external_id"):
            existing = s.scalars(select(ImageRecord).where(
                ImageRecord.monument_slug == kw["monument_slug"],
                ImageRecord.external_id == kw["external_id"])).first()
            if existing:
                return existing
        rec = ImageRecord(**kw)
        s.add(rec)
        s.flush()
        s.refresh(rec)
        return rec


def list_images(slug: str) -> list[ImageRecord]:
    with session() as s:
        rows = s.scalars(select(ImageRecord).where(ImageRecord.monument_slug == slug)).all()
    # dated images newest first, undated last
    return sorted(rows, key=lambda r: (r.acquired_on is None, -(r.acquired_on.timestamp() if r.acquired_on else 0)))


def get_image(image_id: int) -> ImageRecord | None:
    with session() as s:
        return s.get(ImageRecord, image_id)


# ---------------------------------------------------------------- detections
def save_run(image_id: int, slug: str, model_name: str, weights_sha: str, conf: float,
             facade: str, annotated_path: str, dets: list[dict]) -> int:
    with session() as s:
        run = DetectionRun(image_id=image_id, monument_slug=slug, model_name=model_name,
                           weights_sha=weights_sha, conf_threshold=conf, facade=facade,
                           annotated_path=annotated_path)
        run.detections = [Detection(monument_slug=slug, label=d["label"], confidence=d["confidence"],
                                    x1=d["box"][0], y1=d["box"][1], x2=d["box"][2], y2=d["box"][3],
                                    u=d["u"], v=d["v"]) for d in dets]
        s.add(run)
        s.flush()
        return run.id


def runs_for(slug: str) -> list[dict]:
    with session() as s:
        runs = s.scalars(select(DetectionRun).where(DetectionRun.monument_slug == slug)
                         .order_by(DetectionRun.created_at.desc())).all()
        return [_run_dict(r) for r in runs]


def latest_run(slug: str) -> dict | None:
    r = runs_for(slug)
    return r[0] if r else None


def _run_dict(r: DetectionRun) -> dict:
    img = r.image
    return {
        "id": r.id, "image_id": r.image_id, "model": r.model_name, "weights_sha": r.weights_sha,
        "conf": r.conf_threshold, "facade": r.facade, "annotated_path": r.annotated_path,
        "created_at": r.created_at, "is_demo": r.is_demo,
        "image": {"path": img.path, "kind": img.kind, "source": img.source, "acquired_on": img.acquired_on,
                  "date_note": img.date_note, "attribution": img.attribution},
        "detections": [{"id": d.id, "label": d.label, "confidence": d.confidence,
                        "box": [d.x1, d.y1, d.x2, d.y2], "u": d.u, "v": d.v, "mapping": d.mapping}
                       for d in r.detections],
    }


# ---------------------------------------------------------------- explanations
def cached_explanation(slug: str, label: str) -> Explanation | None:
    with session() as s:
        return s.scalars(select(Explanation).where(Explanation.monument_slug == slug, Explanation.label == label)
                         .order_by(Explanation.created_at.desc())).first()


def save_explanation(slug: str, label: str, text: str, model: str) -> None:
    with session() as s:
        s.add(Explanation(monument_slug=slug, label=label, text=text, model=model))


# ---------------------------------------------------------------- comparisons
def save_comparison(**kw) -> None:
    with session() as s:
        s.add(Comparison(**kw))


def comparisons_for(slug: str) -> list[Comparison]:
    with session() as s:
        return s.scalars(select(Comparison).where(Comparison.monument_slug == slug)
                         .order_by(Comparison.created_at.desc())).all()


# ---------------------------------------------------------------- dashboard aggregates
def global_stats() -> dict:
    """Everything the dashboard shows, computed from stored rows (never hardcoded)."""
    with session() as s:
        n_images = s.scalar(select(func.count(ImageRecord.id))) or 0
        n_runs = s.scalar(select(func.count(DetectionRun.id))) or 0
        n_dets = s.scalar(select(func.count(Detection.id))) or 0
        by_class = dict(s.execute(select(Detection.label, func.count(Detection.id)).group_by(Detection.label)).all())
        imgs_by_m = dict(s.execute(select(ImageRecord.monument_slug, func.count(ImageRecord.id))
                                   .group_by(ImageRecord.monument_slug)).all())
        # latest run per monument -> its detections are the "current condition"
        latest_ids = s.execute(select(DetectionRun.monument_slug, func.max(DetectionRun.id))
                               .group_by(DetectionRun.monument_slug)).all()
        current: dict[str, dict] = {}
        for slug, rid in latest_ids:
            run = s.get(DetectionRun, rid)
            labels = [d.label for d in run.detections]
            current[slug] = {"run_id": rid, "count": len(labels), "labels": labels,
                             "max_conf": max([d.confidence for d in run.detections], default=0.0),
                             "at": run.created_at}
        recent = s.scalars(select(DetectionRun).order_by(DetectionRun.created_at.desc()).limit(8)).all()
        recent_list = [{"slug": r.monument_slug, "at": r.created_at, "count": len(r.detections),
                        "source": r.image.source, "kind": r.image.kind} for r in recent]
        recent_imgs = s.scalars(select(ImageRecord).order_by(ImageRecord.added_at.desc()).limit(8)).all()
        recent_img_list = [{"slug": i.monument_slug, "at": i.added_at, "source": i.source, "kind": i.kind}
                           for i in recent_imgs]
    return {"images": n_images, "runs": n_runs, "detections": n_dets, "by_class": by_class,
            "images_by_monument": imgs_by_m, "current": current, "recent_runs": recent_list,
            "recent_images": recent_img_list}
