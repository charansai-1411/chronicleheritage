# Audit of the original repository (Jyothi-ch-tech/ChronicleHeritage)

The original code was reviewed file by file. Decision: **rebuild with a clean modular
architecture**, reusing the sound ideas (SQLAlchemy store, Esri imagery, Three.js viewer)
and the four usable GLB models. Issues found:

## Security
1. **Leaked secret** — `.env.example` contains a real Sentinel Hub `SH_CLIENT_ID` and
   `SH_CLIENT_SECRET`. These are public on GitHub. **Revoke them in the Sentinel Hub dashboard.**
2. `.env` itself is committed (even though `.gitignore` lists it).
3. `utils/__pycache__/*.pyc` files are committed.

## Wrong or inconsistent configuration
4. The report says the model is **YOLO26l**, but the code loads `models/yolo12s/best.pt`
   (`utils/damage_detection.py`, captions say "YOLO12s"), and `pipeline/defects.py` loads yet
   another path, `models/defects_best.pt`. Neither file is in the repo.
5. **Three different monument lists**: `config/monuments.py` has 24 monuments, `utils/digital_twin.py`
   has 12 (including India Gate, Lotus Temple, Iron Pillar, Gateway of India — not in the 14),
   `utils/recognition.py` has 23 classes. Qutb Shahi Tombs is missing everywhere.
6. Charminar's GLB path is `assets/models/charminar.glb`, but the file is at `models/3d/charminar.glb`,
   so the 3D view **always showed a placeholder box**.
7. Gemini model names differ (`gemini-3.5-flash` vs `gemini-3.8-flash`); the latter does not exist.
8. `utils/heritage_ai.py` raises `ValueError` **at import time** if the key is missing, so the app crashes.
9. `README.md` is UTF‑16 encoded with only a title; there is no `requirements.txt`.

## Scientific / correctness problems
10. **Domain mismatch**: the damage model (trained on close-range photos of cracks/moss) was run on
    satellite tiles. At 0.3–10 m/pixel (Sentinel‑2 is 10 m — the whole Charminar is ~3 pixels),
    cracks and moss are physically invisible, so any boxes are noise.
11. **Severity derived from confidence** (`score ≥ 0.85 → Critical`). Confidence is the detector's
    certainty, not damage severity — this mislabels findings.
12. **False 3D precision**: a box's (u, v) on a top-down satellite image was mapped to the 3D model's
    bounding box and every marker placed on the *roof* (`y = box.max.y`). It was not labelled approximate.
13. `rotation_deg` was 0 for all models, so north on the satellite image did not match the model.
14. "Bypass 24-hour cache (fresh imagery)" implies new imagery, but Esri World Imagery is an undated
    mosaic — it returns the same picture. The image date was never shown.
15. `detect_damage` maps class IDs through a hardcoded list instead of `model.names`, so any
    change in class order silently mislabels detections.
16. RAG exists only for Charminar; the knowledge files (`data/heritage_knowledge/...`) are not in the
    repo while the built vector DB is, so it cannot be rebuilt. The chatbot does not see stored damage results.
17. The monument-recognition EfficientNet (Keras) is not wired into the app and its classes don't match the 14.

## Engineering
18. Whole GLB files (up to 55 MB) are base64‑embedded into the page on every rerun — very slow.
19. ~200 MB of GLBs for monuments not in scope are committed without Git LFS.
20. Test files (`test_3d.py`, `test_damage.py`, …) are separate Streamlit apps, not tests.
21. Mixed deprecated (`use_container_width`) and new (`width="stretch"`) Streamlit APIs.
22. No dashboard, no historical comparison, no explanations, no per-monument chatbot, no reports.

## How the rebuild fixes them
| Issue | Fix |
|---|---|
| 1–3 | `.env.example` has no values; `.env`, caches, data and weights are git-ignored |
| 4 | One weights path (`models/weights/yolo26l_heritage.pt`, configurable), class names read from the model and normalised |
| 5 | Single registry `data/monuments.json` with exactly the 14 monuments |
| 6, 18, 19 | GLBs in `static/models3d/<slug>.glb`, served by Streamlit static serving (not base64) |
| 7, 8 | Configurable Gemini model list; missing key → graceful fallback, never a crash |
| 10 | Detection is meant for ground/drone photos; satellite images get a clear domain warning |
| 11 | Status = which classes were found; confidence is shown only as confidence |
| 12, 13 | Facade-projection + mesh ray-casting, per-model calibration file, every marker labelled **approximate** |
| 14 | Esri **Wayback** releases + Esri metadata service → real capture dates for every satellite image |
| 16 | One Chroma collection per monument + live database context in every answer |
