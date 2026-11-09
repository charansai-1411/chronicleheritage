"""Verify the trained YOLO26l weights and how their classes map to the damage names.

    python scripts/check_model.py

Prints the class names stored inside the weights next to the name ChronicleHeritage will
show, so you can confirm the mapping before trusting any detection result.

Two cases:

* The weights carry real names ("Crack", "Moss", ...). The mapping is read from the model
  and nothing is assumed — spelling differences are normalised.
* The weights name their classes "0".."6" (some dataset exports do this). Then the name
  carries no information and the app maps each id **positionally** into
  config/settings.py DAMAGE_CLASSES. Check that list against the `names:` block of the
  data.yaml you trained with. If the order differs, fix DAMAGE_CLASSES — every label,
  colour, 3D marker and Gemini explanation depends on it.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import settings  # noqa: E402
from core import detection  # noqa: E402


def main() -> int:
    print(f"\nWeights: {settings.YOLO_WEIGHTS}")
    if not settings.weights_available():
        print("  NOT FOUND — copy your trained best.pt there (see models/weights/README.md).")
        return 1
    try:
        names = detection.raw_names()
    except Exception as e:
        print(f"  Could not load the weights: {e}")
        return 1

    print(f"  SHA (first 16): {detection.weights_sha()}")
    print(f"  Classes in model: {len(names)} (expected 7)\n")
    positional = detection.names_are_positional()
    print(f"{'id':<4}{'name in weights':<34}{'shown in app':<34}source")
    print("-" * 94)
    for i in sorted(names):
        raw = names[i]
        shown = detection.canonical(i if detection.is_positional(raw) else raw)
        src = "POSITIONAL (assumed)" if detection.is_positional(raw) else "from model"
        print(f"{i:<4}{str(raw):<34}{shown:<34}{src}")

    unknown = [n for n in (detection.canonical(i if detection.is_positional(names[i]) else names[i])
                           for i in sorted(names)) if n not in settings.DAMAGE_CLASSES]
    print()
    if len(names) != 7:
        print(f"WARNING: the model has {len(names)} classes, not 7. These weights may not be the "
              "heritage-damage model.")
    if positional:
        print("WARNING: the weights name their classes numerically, so the app maps them by position:")
        for i, c in enumerate(settings.DAMAGE_CLASSES):
            print(f"         {i} -> {c}")
        print("         Confirm this against the `names:` block of your training data.yaml.")
        print("         If it differs, edit DAMAGE_CLASSES in config/settings.py to match that order.")
    if unknown:
        print(f"WARNING: these labels match none of the seven damage categories: {unknown}")
        print("         They will be shown as-is, in a neutral colour, with no reference text.")
    if not positional and not unknown and len(names) == 7:
        print("OK: the model's own class names map cleanly onto the seven damage categories.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
