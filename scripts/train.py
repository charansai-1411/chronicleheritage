"""Fine-tune YOLO26 on the heritage damage dataset, then install the weights into the app.

    python scripts/train.py --data D:/heritage_dataset            # full run, yolo26l
    python scripts/train.py --data D:/heritage_dataset --model yolo26s.pt --epochs 20   # quick trial
    python scripts/train.py --data D:/heritage_dataset --check-only                     # inspect only

Why fine-tuning and not "from scratch": training randomly-initialised weights needs
hundreds of thousands of images. Starting from the COCO-pretrained checkpoint
(`yolo26l.pt`, downloaded automatically) and continuing on ~6k heritage images is the
standard approach and is what "training YOLO26l on our dataset" means in the literature.

Expected dataset layout (YOLO format), any root directory:

    <root>/images/train/*.jpg   <root>/labels/train/*.txt
    <root>/images/val/*.jpg     <root>/labels/val/*.txt
    <root>/images/test/*.jpg    <root>/labels/test/*.txt

(`train/images` + `train/labels` — the Roboflow layout — is detected too.)

This script writes `<root>/data.yaml` itself, with the class names in exactly the order of
settings.DAMAGE_CLASSES. That keeps the trained model's `names` identical to what the app
expects, so labels never have to be guessed by position. If your existing labels use a
*different* class order, pass --keep-yaml and fix settings.DAMAGE_CLASSES instead.
"""
from __future__ import annotations

import argparse
import collections
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import settings  # noqa: E402

IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def _split_dirs(root: Path, split: str) -> tuple[Path | None, Path | None]:
    """Support both <root>/images/<split> and <root>/<split>/images."""
    for img, lbl in ((root / "images" / split, root / "labels" / split),
                     (root / split / "images", root / split / "labels")):
        if img.is_dir():
            return img, (lbl if lbl.is_dir() else None)
    return None, None


def inspect(root: Path) -> dict:
    """Count images/labels per split and the class ids actually used."""
    report = {}
    for split in ("train", "val", "test"):
        img_dir, lbl_dir = _split_dirs(root, split)
        if img_dir is None:
            report[split] = None
            continue
        imgs = [p for p in img_dir.iterdir() if p.suffix.lower() in IMG_EXT]
        ids, empty, bad = collections.Counter(), 0, []
        if lbl_dir:
            for p in lbl_dir.glob("*.txt"):
                lines = [ln for ln in p.read_text(errors="ignore").splitlines() if ln.strip()]
                if not lines:
                    empty += 1
                for ln in lines:
                    parts = ln.split()
                    try:
                        ids[int(parts[0])] += 1
                        if len(parts) < 5 or not all(0 <= float(v) <= 1 for v in parts[1:5]):
                            bad.append(p.name)
                    except (ValueError, IndexError):
                        bad.append(p.name)
        report[split] = {"images": len(imgs), "labels": len(list(lbl_dir.glob("*.txt"))) if lbl_dir else 0,
                         "boxes": sum(ids.values()), "ids": dict(sorted(ids.items())),
                         "empty_label_files": empty, "malformed": sorted(set(bad))[:5],
                         "img_dir": img_dir, "lbl_dir": lbl_dir}
    return report


def print_report(root: Path, report: dict) -> int:
    names = settings.DAMAGE_CLASSES
    print(f"\nDataset: {root}")
    total = 0
    for split, r in report.items():
        if r is None:
            print(f"  {split:<6} MISSING")
            continue
        total += r["images"]
        print(f"  {split:<6} {r['images']:>5} images  {r['labels']:>5} label files  {r['boxes']:>6} boxes"
              + (f"  ({r['empty_label_files']} empty)" if r["empty_label_files"] else ""))
        if r["malformed"]:
            print(f"         malformed labels e.g. {r['malformed']}")
    print(f"  TOTAL  {total} images (your report says 6,317: 5,097 train / 972 val / 248 test)")

    used = sorted({i for r in report.values() if r for i in r["ids"]})
    print(f"\nClass ids used in labels: {used}")
    print(f"{'id':<4}{'name that will be trained':<34}boxes")
    print("-" * 56)
    counts = collections.Counter()
    for r in report.values():
        if r:
            counts.update(r["ids"])
    for i in used:
        nm = names[i] if 0 <= i < len(names) else f"!! id {i} has no name !!"
        print(f"{i:<4}{nm:<34}{counts[i]}")
    problems = 0
    if used and max(used) >= len(names):
        print(f"\nERROR: labels use id {max(used)} but DAMAGE_CLASSES has only {len(names)} names.")
        problems += 1
    missing = [i for i in range(len(names)) if i not in counts]
    if missing:
        print(f"\nWARNING: no boxes for class ids {missing} ({[names[i] for i in missing]}). "
              "The model cannot learn a class it never sees.")
    rare = [(i, c) for i, c in counts.items() if c < 50]
    if rare:
        print(f"WARNING: very few boxes for {[(names[i] if i < len(names) else i, c) for i, c in rare]} — "
              "expect poor accuracy on those classes; say so in your report.")
    return problems


def write_yaml(root: Path) -> Path:
    p = root / "data.yaml"
    lines = [f"# generated by scripts/train.py — class order matches config/settings.py DAMAGE_CLASSES",
             f"path: {root.as_posix()}"]
    for split in ("train", "val", "test"):
        img_dir, _ = _split_dirs(root, split)
        if img_dir:
            lines.append(f"{split}: {img_dir.relative_to(root).as_posix()}")
    lines += [f"nc: {len(settings.DAMAGE_CLASSES)}", "names:"]
    lines += [f"  {i}: {n}" for i, n in enumerate(settings.DAMAGE_CLASSES)]
    p.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return p


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="dataset root folder")
    ap.add_argument("--model", default="yolo26l.pt", help="starting checkpoint (yolo26n/s/m/l.pt)")
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--imgsz", type=int, default=settings.YOLO_IMGSZ)
    ap.add_argument("--batch", default="-1", help="-1 = auto-fit to your GPU memory")
    ap.add_argument("--device", default=settings.YOLO_DEVICE or None, help='"0" for GPU, "cpu"')
    ap.add_argument("--name", default="heritage")
    ap.add_argument("--patience", type=int, default=25, help="early-stop after N epochs without improvement")
    ap.add_argument("--keep-yaml", action="store_true", help="use the existing data.yaml, do not overwrite")
    ap.add_argument("--check-only", action="store_true", help="inspect the dataset and exit")
    a = ap.parse_args()

    root = Path(a.data).expanduser().resolve()
    if not root.is_dir():
        print(f"Dataset folder not found: {root}")
        return 1
    report = inspect(root)
    if all(r is None for r in report.values()):
        print(f"No images/<split> or <split>/images folders under {root}.")
        return 1
    if print_report(root, report):
        return 1

    yaml_path = root / "data.yaml"
    if a.keep_yaml and yaml_path.exists():
        print(f"\nUsing existing {yaml_path}")
    else:
        yaml_path = write_yaml(root)
        print(f"\nWrote {yaml_path} with class names in DAMAGE_CLASSES order")
    if a.check_only:
        return 0

    from ultralytics import YOLO
    batch = int(a.batch) if str(a.batch).lstrip("-").isdigit() else a.batch
    print(f"\nFine-tuning {a.model} · {a.epochs} epochs · imgsz {a.imgsz} · batch {batch}\n")
    model = YOLO(a.model)
    model.train(data=str(yaml_path), epochs=a.epochs, imgsz=a.imgsz, batch=batch, device=a.device,
                name=a.name, patience=a.patience, pretrained=True, plots=True, seed=0)

    metrics = model.val(data=str(yaml_path), split="test" if report.get("test") else "val")
    best = Path(model.trainer.best)
    print("\n" + "=" * 60)
    print(f"mAP50-95 {metrics.box.map:.3f}   mAP50 {metrics.box.map50:.3f}")
    print(f"{'class':<34}{'mAP50':>8}{'mAP50-95':>10}")
    for i, c in enumerate(metrics.box.ap_class_index):
        nm = settings.DAMAGE_CLASSES[c] if c < len(settings.DAMAGE_CLASSES) else str(c)
        print(f"{nm:<34}{metrics.box.ap50[i]:>8.3f}{metrics.box.ap[i]:>10.3f}")
    print("=" * 60)

    settings.YOLO_WEIGHTS.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(best, settings.YOLO_WEIGHTS)
    print(f"\nInstalled {best} -> {settings.YOLO_WEIGHTS}")
    print("Detection is now live in the app. Verify with: python scripts/check_model.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
