from __future__ import annotations

import cv2
import numpy as np


def read_rgb(path: str, image_size: int = 150) -> np.ndarray:
    image = cv2.imread(path, cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (image_size, image_size), interpolation=cv2.INTER_AREA)
    return image


def normalize_image(image: np.ndarray) -> np.ndarray:
    return image.astype(np.float32) / 255.0


def estimate_lesion_mask(image_rgb: np.ndarray) -> np.ndarray:
    """Otsu + morphological opening/closing + largest connected component."""
    gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    _, mask1 = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    _, mask2 = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    # Pick the polarity whose foreground is not almost the whole image and is closest to lesion-like coverage.
    candidates = []
    for m in (mask1, mask2):
        cov = float((m > 0).mean())
        score = abs(cov - 0.35) + (10.0 if cov < 0.02 or cov > 0.95 else 0.0)
        candidates.append((score, m))
    mask = min(candidates, key=lambda x: x[0])[1]
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    num, labels, stats, _ = cv2.connectedComponentsWithStats((mask > 0).astype(np.uint8), connectivity=8)
    if num <= 1:
        return (mask > 0).astype(np.uint8)
    largest = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
    return (labels == largest).astype(np.uint8)


def lesion_coverage(image_rgb: np.ndarray) -> float:
    mask = estimate_lesion_mask(image_rgb)
    return float(mask.mean())


def patch_size_from_coverage(coverage: float, thresholds=(0.25, 0.50)) -> int:
    if coverage < thresholds[0]:
        return 8
    if coverage < thresholds[1]:
        return 16
    return 32


def preprocess_path(path: str, image_size: int = 150, thresholds=(0.25, 0.50)):
    rgb = read_rgb(path, image_size)
    coverage = lesion_coverage(rgb)
    patch_size = patch_size_from_coverage(coverage, thresholds)
    return normalize_image(rgb), coverage, patch_size
