# YOLO26l weights — integration point

Place your trained heritage-damage model here as:

    models/weights/yolo26l_heritage.pt

(the `best.pt` from `runs/detect/<name>/weights/` after training `yolo26l.pt` on the
7-class heritage dataset), or point `YOLO_WEIGHTS` in `.env` at it.

Expected class names (any order; spelling variants are normalised):
Crack, Discolouration, Efflorescence, Fabric Loss of Masonry Units, Moss, Plant, Surface Dissolution.

Without this file the app disables detection and says so — it never fabricates predictions.
