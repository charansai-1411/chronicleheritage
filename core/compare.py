"""Historical image comparison (OpenCV).

Pipeline: resize → illumination normalisation (CLAHE on L channel) → ORB
features + RANSAC homography registration → SSIM map + absolute difference →
morphological clean-up → changed regions.

If registration fails (too few inliers / degenerate homography) we still show
the images side by side, but report that pixel differences are NOT meaningful.
Even after alignment, differences can come from lighting, season, shadows,
vegetation, people or camera angle — the UI labels them "visible changes to
review", never "confirmed damage".
"""
from __future__ import annotations

import cv2
import numpy as np
from PIL import Image


def _prep(img: Image.Image, width: int = 900) -> np.ndarray:
    a = cv2.cvtColor(np.array(img.convert("RGB")), cv2.COLOR_RGB2BGR)
    h = int(a.shape[0] * width / a.shape[1])
    return cv2.resize(a, (width, h), interpolation=cv2.INTER_AREA)


def _normalise(bgr: np.ndarray) -> np.ndarray:
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
    lab[:, :, 0] = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(lab[:, :, 0])
    return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)


def _ssim_map(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    a, b = a.astype(np.float64), b.astype(np.float64)
    C1, C2 = (0.01 * 255) ** 2, (0.03 * 255) ** 2
    blur = lambda x: cv2.GaussianBlur(x, (11, 11), 1.5)
    mu_a, mu_b = blur(a), blur(b)
    sa, sb, sab = blur(a * a) - mu_a ** 2, blur(b * b) - mu_b ** 2, blur(a * b) - mu_a * mu_b
    return ((2 * mu_a * mu_b + C1) * (2 * sab + C2)) / ((mu_a ** 2 + mu_b ** 2 + C1) * (sa + sb + C2))


def register(before: np.ndarray, after: np.ndarray) -> tuple[np.ndarray | None, int]:
    orb = cv2.ORB_create(5000)
    g1, g2 = cv2.cvtColor(before, cv2.COLOR_BGR2GRAY), cv2.cvtColor(after, cv2.COLOR_BGR2GRAY)
    k1, d1 = orb.detectAndCompute(g1, None)
    k2, d2 = orb.detectAndCompute(g2, None)
    if d1 is None or d2 is None or len(k1) < 20 or len(k2) < 20:
        return None, 0
    matches = cv2.BFMatcher(cv2.NORM_HAMMING).knnMatch(d2, d1, k=2)
    good = [m for m, n in (p for p in matches if len(p) == 2) if m.distance < 0.75 * n.distance]
    if len(good) < 25:
        return None, len(good)
    src = np.float32([k2[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
    dst = np.float32([k1[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
    H, mask = cv2.findHomography(src, dst, cv2.RANSAC, 4.0)
    inliers = int(mask.sum()) if mask is not None else 0
    if H is None or inliers < 30 or abs(np.linalg.det(H[:2, :2])) < 0.2 or abs(np.linalg.det(H[:2, :2])) > 5:
        return None, inliers
    return H, inliers


def compare(before_img: Image.Image, after_img: Image.Image, sensitivity: float = 0.5) -> dict:
    b, a0 = _prep(before_img), _prep(after_img)
    H, inliers = register(b, a0)
    aligned = H is not None
    if aligned:
        a = cv2.warpPerspective(a0, H, (b.shape[1], b.shape[0]))
        valid = cv2.warpPerspective(np.full(a0.shape[:2], 255, np.uint8), H, (b.shape[1], b.shape[0])) > 0
    else:
        a = cv2.resize(a0, (b.shape[1], b.shape[0]))
        valid = np.ones(b.shape[:2], bool)
    bn, an = _normalise(b), _normalise(a)
    gb, ga = cv2.cvtColor(bn, cv2.COLOR_BGR2GRAY), cv2.cvtColor(an, cv2.COLOR_BGR2GRAY)
    smap = _ssim_map(gb, ga)
    ssim = float(smap[valid].mean()) if valid.any() else 0.0
    thr = 0.05 + 0.4 * sensitivity                  # SSIM below thr = changed; higher sensitivity → more flagged
    mask = ((smap < thr) & valid).astype(np.uint8) * 255
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8))
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    min_area = 0.0008 * mask.size
    regions = [cv2.boundingRect(c) for c in cnts if cv2.contourArea(c) > min_area]
    overlay = an.copy()
    heat = cv2.applyColorMap(((1 - np.clip(smap, 0, 1)) * 255).astype(np.uint8), cv2.COLORMAP_INFERNO)
    overlay = cv2.addWeighted(overlay, 0.65, heat, 0.35, 0)
    for x, y, w, h in regions:
        cv2.rectangle(overlay, (x, y), (x + w, y + h), (58, 31, 11), 2)
    changed = float((mask > 0)[valid].mean() * 100) if valid.any() else 0.0
    to_pil = lambda x: Image.fromarray(cv2.cvtColor(x, cv2.COLOR_BGR2RGB))
    return {"aligned": aligned, "inliers": inliers, "ssim": ssim, "changed_pct": changed,
            "regions": [{"x": int(x), "y": int(y), "w": int(w), "h": int(h)} for x, y, w, h in regions],
            "before": to_pil(b), "after": to_pil(a), "overlay": to_pil(overlay),
            "diff": to_pil(cv2.applyColorMap(cv2.absdiff(gb, ga), cv2.COLORMAP_BONE))}
