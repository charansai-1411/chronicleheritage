# Weekly progress — ChronicleHeritage (Batch 103)

Weeks 1–6 are taken from the submitted weekly status reports. Weeks 7–10 describe the work in
this rebuild. You can copy them into the "Work carried out for the week" box.

| Week | Title | Status |
|---|---|---|
| 1 | Project Selection and Planning | done (reported) |
| 2 | Problem Identification and Literature Survey | done (reported) |
| 3 | Methodology and System Architecture | done (reported) |
| 4 | Dataset Collection and Preprocessing (6,317 images, 7 classes) | done (reported) |
| 5 | Model Training and Module Implementation | done (reported) |
| 6 | System Development and Documentation | done (reported) |

## Week 7 – Code Review and Re-architecture
1. Reviewed the existing implementation and found 22 issues (see AUDIT.md), including a leaked
   Sentinel Hub secret, inconsistent model paths, and running the detector on satellite imagery.
2. Redesigned the application as 11 separate modules with central configuration (.env) and a
   single registry of the 14 monuments.
3. Built a structured knowledge base (history, architecture, significance, conservation issues)
   for all 14 monuments.
4. Designed the database schema for images, detection runs, detections, explanations and comparisons.

## Week 8 – Dashboard, Imagery Pipeline and YOLO26l Integration
1. Developed a premium single-page dashboard: statistics computed from the database, a
   damage-distribution chart, a condition matrix, recent activity, and 14 monument cards.
2. Integrated Esri World Imagery Wayback to collect dated multi-temporal satellite images, with
   real capture dates from Esri's metadata service, plus ground photographs from Wikimedia Commons.
3. Integrated YOLO26l inference with class-name normalisation, annotated outputs and stored
   monitoring history.
4. Developed the Charminar pilot monitoring page.

## Week 9 – 3D Damage Visualization and AI Modules
1. Implemented an interactive Three.js 3D viewer for the GLB models: rotate, zoom and pan,
   per-class markers, clickable details, a legend, and a switch between normal and damage view.
2. Implemented image-to-3D mapping by facade projection and mesh ray-casting. Every marker is
   labelled as an approximate location.
3. Integrated Gemini damage explanations (possible causes, prevention, uncertainty), cached in the database.
4. Built 3D models for all 14 monuments. Verified the 3 usable community GLBs and replaced the incorrect Fatehpur Sikri file (a 360° interior photo). Generated parametric massing models from documented dimensions for the other 11.
5. Implemented historical image comparison with OpenCV: ORB + RANSAC registration, CLAHE
   normalisation, SSIM change maps and a before/after slider.

## Week 10 – RAG Chatbot, Reports, Testing and Rollout
1. Built a separate chatbot for each monument using LangChain, ChromaDB, Sentence-Transformers and Gemini.
   Each chatbot is grounded in that monument's knowledge base and its live monitoring records.
2. Implemented monument-wise PDF monitoring reports.
3. Wrote automated tests (pytest) for the registry, the database statistics, the comparison
   pipeline and the explanation fallback.
4. Extended the tested Charminar page to all 14 monuments and wrote the README and setup instructions.

## Pending (next weeks)
* Copy the final trained YOLO26l `best.pt` to `models/weights/yolo26l_heritage.pt` and report mAP on the 248-image test set.
* Replace the parametric models with COLMAP/photogrammetry reconstructions where field photos allow (pilot: Charminar).
* Optionally, register photos to the 3D models with COLMAP camera poses, to place markers precisely.
