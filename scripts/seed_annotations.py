"""Load the hand-marked reference annotations in data/annotations.json into the database.

    python scripts/seed_annotations.py            # load / refresh
    python scripts/seed_annotations.py --clear    # remove them again

They are stored as runs with is_demo=True and the model name
"Hand annotation (not a model prediction)", and the app says so wherever they appear.
See core/annotation.py for why they exist.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core import annotation, database as db  # noqa: E402


def main() -> int:
    db.init_db()
    if "--clear" in sys.argv:
        print(f"Removed {annotation.clear()} hand-annotation run(s).")
        return 0
    annotation.seed()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
