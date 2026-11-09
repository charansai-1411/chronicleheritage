---
title: ChronicleHeritage
emoji: 🏛️
colorFrom: blue
colorTo: yellow
sdk: docker
app_port: 7860
pinned: false
short_description: AI heritage monitoring with 3D damage visualisation
---

# ChronicleHeritage

**An AI-driven digital twin and threat-assessment platform for endangered monuments.**
CBIT · Dept. of AI & DS · Batch 103 · Guide: Ms. P. Kiranmaie

ChronicleHeritage monitors 14 Indian monuments. It collects dated satellite and ground
images, detects 7 kinds of masonry damage with a trained **YOLO26l** model, places the
findings on interactive **3D models**, and uses **Gemini** to suggest possible causes. It
also compares images from different dates and gives each monument its own **RAG chatbot**.
Monitoring reports can be exported as **PDF**.

## Quick start

```bash
pip install -r requirements.txt
cp .env.example .env            # add GEMINI_API_KEY (optional, enables AI text)
python scripts/fetch_imagery.py # downloads dated imagery for all 14 monuments (safe to re-run)
streamlit run app.py
```

Run `streamlit` **from this folder**. The app needs `.streamlit/config.toml`, which turns on
static file serving so the browser can load the 3D models.

### Required files you must supply
| File | Purpose | Without it |
|---|---|---|
| `models/weights/yolo26l_heritage.pt` | your trained YOLO26l `best.pt` | detection is disabled; the app shows **hand-marked reference annotations** instead, labelled as such everywhere (see below). No placeholder predictions are ever shown |

### While the detector is untrained: hand annotations

Without weights there is nothing for sections 04, 05 and 08 to work on, so the project team
marks damage **by eye** on real photographs and stores those marks in `data/annotations.json`:

```bash
python scripts/seed_annotations.py          # load / refresh them
python scripts/seed_annotations.py --clear  # remove them
```

They are stored as runs with `is_demo = True` and the model name
`Hand annotation (not a model prediction)`. Everywhere they appear — section 03, the 3D twin,
Gemini's input, the monitoring history, the PDF report and the dashboard — the app says they are
hand annotations and not model output, and the rendered image carries the same statement burnt
into the picture so a screenshot cannot be mistaken for a detection result either. No confidence
score is shown, because a hand mark has none.

They are also the detector's first test set: once the weights land, run the model on the same
images and compare its boxes against these.

Currently annotated: Golconda Fort (7), Ellora Caves (5), Qutb Shahi Tombs (4), Charminar (3).

After adding the weights, run `python scripts/check_model.py` once. It prints the class names stored
in the weights next to the names the app will show. If your export named the classes `0`–`6` instead
of the damage names, the app maps them **by position** into `DAMAGE_CLASSES` in `config/settings.py` —
check that order against the `names:` block of your training `data.yaml` and fix it there if it differs.

| `.env` → `GEMINI_API_KEY` | AI explanations + conversational chatbot | general reference text is shown instead, and the chatbot only returns retrieved passages |
| `static/models3d/<slug>.glb` | 3D model per monument | placeholder massing geometry, labelled as a placeholder |

### 3D models (all 14 monuments)
| Monument | Model | Source |
|---|---|---|
| Charminar, Taj Mahal, Qutub Minar | community models | Sketchfab, CC-BY-4.0 (credit shown in the viewer) |
| Golconda Fort, Qutb Shahi Tombs, Red Fort, Hampi, Ajanta, Ellora, Konark, Sanchi, Brihadeeswara, Fatehpur Sikri, Meenakshi | parametric massing models | `python scripts/build_models.py` |

The original repo's "Fatehpur Sikri" GLB was actually a 360° photo sphere of a room interior ("Fatehpur Mosque with Boy"), so it was replaced. Parametric models are built from documented dimensions and orientations (e.g. Sanchi dome 36.6 m, Brihadisvara 13-tier vimana, Konark's 24 wheels and 7 horses facing east, Lahori Gate facing west). They show the correct form and layout, but they are not scans, and the viewer labels them **parametric approximation**. To replace one, drop a photogrammetry or COLMAP GLB at `static/models3d/<slug>.glb`.

## Training the damage model

There is no public pretrained model for these seven classes (Hugging Face has only
binary crack detectors), so the detector has to be fine-tuned on the project dataset.
Fine-tuning means starting from the COCO-pretrained `yolo26l.pt` checkpoint, which
Ultralytics downloads automatically — not training randomly-initialised weights, which
would need hundreds of thousands of images.

```bash
python scripts/train.py --data <dataset-root> --check-only   # inspect counts and class ids
python scripts/train.py --data <dataset-root>                # fine-tune yolo26l, 100 epochs
```

`--check-only` reports images, boxes and class balance per split and flags classes with
no or very few boxes. The script writes `data.yaml` itself with the class names in
`DAMAGE_CLASSES` order, so the trained model carries real names and the app never has to
guess a label from its class id. When training finishes it prints per-class mAP and copies
`best.pt` to `models/weights/yolo26l_heritage.pt`, which switches detection on in the app.

On a 6 GB GPU, start with `--model yolo26s.pt --epochs 20` to confirm the pipeline, then
run `yolo26l` for the numbers you report.

The seven categories match **MSD-Det** (*Journal of Cultural Heritage*, 2025), the published
masonry heritage damage dataset.

## Workflow (single scrolling page, card navigation, no sidebar, nav bar or tabs)
Dashboard → monument card → **§01** dossier → **§02** imagery (acquire/upload) → **§03** YOLO26l
detection → **§04** 3D damage view → **§05** Gemini possible causes → **§06** historical
comparison → **§07** monument chatbot → **§08** PDF report.

## Architecture
```
app.py                    router (?m=<slug>)
config/settings.py        every path, model name and secret (read from .env)
core/
  monuments.py            Monument Management – registry of 14 (data/monuments.json)
  acquisition.py          Image Acquisition – Esri Wayback (dated satellite), Wikimedia Commons, uploads
  detection.py            YOLO26l Detection – inference, class normalisation, annotation
  damage_reference.py     7 damage classes: colours, markers, literature causes/prevention
  annotation.py           Hand-marked reference annotations used while the detector is untrained
  viewer3d.py             3D Visualization – Three.js, facade projection + ray-casting
  explain.py              AI Damage Explanation – Gemini, cached, hypothesis-framed
  compare.py              Historical Comparison – ORB+RANSAC registration, CLAHE, SSIM
  rag.py                  RAG Chatbot – LangChain + Chroma + MiniLM + Gemini, per-monument
  reports.py              Monitoring Reports – reportlab PDF
  database.py             Database Management – SQLAlchemy (SQLite / PostgreSQL)
  llm.py                  Gemini client with model fallbacks
ui/  theme.py · components.py · dashboard.py · monument_page.py
scripts/fetch_imagery.py  populate the image archive
tests/test_core.py        pytest suite
```

## Honesty rules built into the code
* Every image keeps its **actual acquisition date and source**. Satellite capture dates come
  from Esri's imagery-metadata service. If a date is unknown, the app says so.
* Dashboard numbers are **computed from the database**; nothing is hardcoded.
* Detections come only from real model inference. `is_demo` is set by exactly one thing —
  `scripts/seed_annotations.py`, loading damage the team marked by hand — and every surface that
  reads it says so. Inference always writes it `False`, so the two can never be confused.
* 3D marker positions are **approximate**. A box is projected from the facade the photo
  shows onto the mesh. This is a manual spatial correspondence, not photogrammetry. To
  place markers precisely, add camera poses from COLMAP to the calibration file
  `static/models3d/<slug>.json`.
* Gemini text is labelled **AI-generated hypothesis** and always says "possible causes".
* Image differences are labelled "visible changes to review", never "confirmed damage".

## Tests
```bash
python -m pytest -q tests
```

See `AUDIT.md` for the issues found in the original repository, and `WEEKLY_PROGRESS.md` for
week-by-week progress.
