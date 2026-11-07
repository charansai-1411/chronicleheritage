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
| `models/weights/yolo26l_heritage.pt` | your trained YOLO26l `best.pt` | detection is disabled with a clear message; no placeholder predictions are shown |
| `.env` → `GEMINI_API_KEY` | AI explanations + conversational chatbot | general reference text is shown instead, and the chatbot only returns retrieved passages |
| `static/models3d/<slug>.glb` | 3D model per monument | placeholder massing geometry, labelled as a placeholder |

### 3D models (all 14 monuments)
| Monument | Model | Source |
|---|---|---|
| Charminar, Taj Mahal, Qutub Minar | community models | Sketchfab, CC-BY-4.0 (credit shown in the viewer) |
| Golconda Fort, Qutb Shahi Tombs, Red Fort, Hampi, Ajanta, Ellora, Konark, Sanchi, Brihadeeswara, Fatehpur Sikri, Meenakshi | parametric massing models | `python scripts/build_models.py` |

The original repo's "Fatehpur Sikri" GLB was actually a 360° photo sphere of a room interior ("Fatehpur Mosque with Boy"), so it was replaced. Parametric models are built from documented dimensions and orientations (e.g. Sanchi dome 36.6 m, Brihadisvara 13-tier vimana, Konark's 24 wheels and 7 horses facing east, Lahori Gate facing west). They show the correct form and layout, but they are not scans, and the viewer labels them **parametric approximation**. To replace one, drop a photogrammetry or COLMAP GLB at `static/models3d/<slug>.glb`.

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
* Detections come only from real model inference. `is_demo` is never set by the app.
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
