"""Train the ChronicleHeritage damage detector on Kaggle (GPU + Internet must be ON).

Run it as the only line of a Kaggle notebook cell:

    !pip install -q ultralytics polars && curl -sL https://raw.githubusercontent.com/charansai-1411/chronicleheritage/main/scripts/kaggle_train.py -o t.py && python t.py

Dataset
-------
"Cracks in the Foundation" (CiF, IBM Research) — the largest openly licensed
(CDLA-Permissive-2.0) civil-infrastructure defect dataset with COCO boxes.
We stream the 1024x1024 *tiled* split, so nothing like the 65 GB full-resolution
split is ever downloaded; only the shards we actually consume are fetched.

CiF category -> ChronicleHeritage category
    1 Algae                     -> Moss                           (biological growth)
    2 Crack                     -> Crack
    3 Crack (net-crack)         -> Crack
    4 Crack with precipitation  -> Efflorescence                  (mineral deposit)
    5 Rust                      -> Discolouration                 (metal staining)
    6 Spalling                  -> Fabric Loss of Masonry Units   (detached material)

Plant and Surface Dissolution have no CiF equivalent. They stay in the class list
(so the model's class order matches the app exactly and labels never have to be
guessed), but the model cannot predict them — the app should say so.

CiF is concrete/steel civil infrastructure, not heritage masonry. The defect
*appearance* transfers reasonably, the material does not. Say so in the report,
and retrain on MSD-Det if you obtain it.
"""
from __future__ import annotations

import collections
import json
import os
import random
import shutil
import sys
import time
from pathlib import Path

REPO = "ibm-research/cif-dataset"
CLASSES = ["Crack", "Discolouration", "Efflorescence", "Fabric Loss of Masonry Units",
           "Moss", "Plant", "Surface Dissolution"]
CIF_TO_IDX = {1: 4, 2: 0, 3: 0, 4: 2, 5: 1, 6: 3}

ON_KAGGLE = Path("/kaggle").exists()
ROOT = Path("/kaggle/temp/cif") if ON_KAGGLE else Path("./cif_data")
OUT = Path("/kaggle/working") if ON_KAGGLE else Path("./runs_out")

MAX_TRAIN = int(os.getenv("MAX_TRAIN", "14000"))   # tiles with defects
MAX_VAL = int(os.getenv("MAX_VAL", "2500"))
BG_FRACTION = 0.08                                  # empty tiles help suppress false positives
EPOCHS = int(os.getenv("EPOCHS", "60"))
IMGSZ = int(os.getenv("IMGSZ", "640"))
MODEL = os.getenv("MODEL", "yolo26l.pt")
BATCH = os.getenv("BATCH", "-1")                    # -1 = auto-fit GPU memory
MIN_BOX_PX = 4


def log(*a):
    print(f"[{time.strftime('%H:%M:%S')}]", *a, flush=True)


# --------------------------------------------------------------- build dataset
def build_split(hf_split: str, out_split: str, cap: int) -> dict:
    from datasets import load_dataset

    img_dir = ROOT / "images" / out_split
    lbl_dir = ROOT / "labels" / out_split
    img_dir.mkdir(parents=True, exist_ok=True)
    lbl_dir.mkdir(parents=True, exist_ok=True)

    log(f"streaming {hf_split} -> {out_split} (target {cap} annotated tiles)")
    ds = load_dataset(REPO, hf_split, split="train", streaming=True)
    counts = collections.Counter()
    kept = bg = seen = 0
    bg_budget = int(cap * BG_FRACTION)
    rng = random.Random(0)

    for s in ds:
        seen += 1
        obj = s.get("objects") or {}
        cats = obj.get("category_id") or []
        boxes = obj.get("bbox") or []
        W, H = s["width"], s["height"]
        lines = []
        for cid, (x, y, w, h) in zip(cats, boxes):
            idx = CIF_TO_IDX.get(int(cid))
            if idx is None or w < MIN_BOX_PX or h < MIN_BOX_PX:
                continue
            cx, cy = (x + w / 2) / W, (y + h / 2) / H
            nw, nh = w / W, h / H
            if not (0 < nw <= 1 and 0 < nh <= 1 and 0 <= cx <= 1 and 0 <= cy <= 1):
                continue
            lines.append(f"{idx} {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}")
            counts[idx] += 1

        if lines:
            if kept >= cap:
                if kept + bg >= cap + bg_budget:
                    break
                continue
            kept += 1
        else:
            if bg >= bg_budget or rng.random() > 0.3:
                continue
            bg += 1

        stem = f"{out_split}_{s['image_id']}_{s.get('tile_row') or 0}_{s.get('tile_col') or 0}"
        s["image"].convert("RGB").save(img_dir / f"{stem}.jpg", quality=90)
        (lbl_dir / f"{stem}.txt").write_text("\n".join(lines))

        if (kept + bg) % 1000 == 0:
            log(f"  {out_split}: {kept} annotated + {bg} background (scanned {seen})")

    log(f"  {out_split} done: {kept} annotated + {bg} background tiles")
    return {"kept": kept, "bg": bg, "counts": dict(counts)}


def write_yaml() -> Path:
    p = ROOT / "data.yaml"
    lines = [f"path: {ROOT.as_posix()}", "train: images/train", "val: images/val",
             f"nc: {len(CLASSES)}", "names:"]
    lines += [f"  {i}: {c}" for i, c in enumerate(CLASSES)]
    p.write_text("\n".join(lines) + "\n")
    return p


# --------------------------------------------------------------- main
def main() -> int:
    log(f"Kaggle={ON_KAGGLE}  model={MODEL}  imgsz={IMGSZ}  epochs={EPOCHS}")
    try:
        import torch
        log("torch", torch.__version__, "| CUDA", torch.cuda.is_available(),
            "|", torch.cuda.device_count(), "GPU(s)",
            torch.cuda.get_device_name(0) if torch.cuda.is_available() else "")
        if not torch.cuda.is_available():
            log("!! No GPU. In Kaggle: Settings -> Accelerator -> GPU. Aborting.")
            return 1
    except ImportError:
        log("torch missing"); return 1

    stats = {"train": build_split("train_tiled", "train", MAX_TRAIN),
             "val": build_split("val_tiled", "val", MAX_VAL)}

    total = collections.Counter()
    for s in stats.values():
        total.update({int(k): v for k, v in s["counts"].items()})
    log("\nboxes per class")
    for i, c in enumerate(CLASSES):
        log(f"  {i} {c:<32}{total.get(i, 0)}")
    absent = [CLASSES[i] for i in range(len(CLASSES)) if not total.get(i)]
    if absent:
        log(f"  (no data, model cannot predict these: {absent})")
    if sum(total.values()) == 0:
        log("No boxes produced — aborting."); return 1

    yaml_path = write_yaml()
    log(f"wrote {yaml_path}")

    from ultralytics import YOLO
    import torch
    devices = ",".join(str(i) for i in range(torch.cuda.device_count()))
    batch = int(BATCH) if str(BATCH).lstrip("-").isdigit() else BATCH

    log(f"\ntraining {MODEL} on device {devices}")
    model = YOLO(MODEL)
    model.train(data=str(yaml_path), epochs=EPOCHS, imgsz=IMGSZ, batch=batch, device=devices,
                project=str(OUT / "runs"), name="heritage", exist_ok=True,
                patience=12, pretrained=True, plots=True, seed=0, workers=4,
                cache=False, val=True, optimizer="auto", cos_lr=True, close_mosaic=10)

    metrics = model.val(data=str(yaml_path), split="val", device=devices.split(",")[0])
    log("\n" + "=" * 62)
    log(f"mAP50-95 {metrics.box.map:.4f}   mAP50 {metrics.box.map50:.4f}")
    per_class = {}
    for i, c in enumerate(metrics.box.ap_class_index):
        nm = CLASSES[c] if c < len(CLASSES) else str(c)
        per_class[nm] = {"mAP50": round(float(metrics.box.ap50[i]), 4),
                         "mAP50-95": round(float(metrics.box.ap[i]), 4)}
        log(f"  {nm:<34}mAP50 {metrics.box.ap50[i]:.4f}   mAP50-95 {metrics.box.ap[i]:.4f}")
    log("=" * 62)

    best = Path(model.trainer.best)
    final = OUT / "yolo26l_heritage.pt"
    shutil.copy2(best, final)
    (OUT / "training_summary.json").write_text(json.dumps({
        "dataset": REPO, "license": "CDLA-Permissive-2.0", "mapping": CIF_TO_IDX,
        "classes": CLASSES, "classes_without_data": absent,
        "boxes_per_class": {CLASSES[i]: total.get(i, 0) for i in range(len(CLASSES))},
        "tiles": {k: {"annotated": v["kept"], "background": v["bg"]} for k, v in stats.items()},
        "model": MODEL, "imgsz": IMGSZ, "epochs": EPOCHS,
        "mAP50": round(float(metrics.box.map50), 4), "mAP50-95": round(float(metrics.box.map), 4),
        "per_class": per_class}, indent=2))
    log(f"\nSaved {final}")
    log("Download it from the notebook's Output tab, then put it in the app at "
        "models/weights/yolo26l_heritage.pt")
    return 0


if __name__ == "__main__":
    sys.exit(main())
