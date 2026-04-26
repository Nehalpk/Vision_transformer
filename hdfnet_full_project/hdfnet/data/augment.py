from __future__ import annotations

import cv2
import numpy as np


def random_flip_rotate(image: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    if rng.random() < 0.5:
        image = np.flip(image, axis=1)
    if rng.random() < 0.5:
        image = np.flip(image, axis=0)
    k = int(rng.integers(0, 4))
    image = np.rot90(image, k)
    return np.ascontiguousarray(image)


def color_jitter(image: np.ndarray, rng: np.random.Generator, brightness=0.12, contrast=0.18, saturation=0.12) -> np.ndarray:
    x = image.astype(np.float32)
    x = x * (1.0 + rng.uniform(-contrast, contrast)) + rng.uniform(-brightness, brightness)
    hsv = cv2.cvtColor(np.clip(x, 0, 1), cv2.COLOR_RGB2HSV)
    hsv[..., 1] = np.clip(hsv[..., 1] * (1.0 + rng.uniform(-saturation, saturation)), 0, 1)
    x = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
    return np.clip(x, 0, 1).astype(np.float32)


def sharpen_texture_regions(image: np.ndarray, amount: float = 0.25) -> np.ndarray:
    blur = cv2.GaussianBlur(image, (0, 0), 1.0)
    sharp = np.clip(image + amount * (image - blur), 0, 1)
    return sharp.astype(np.float32)


def augment_image(image: np.ndarray, rng: np.random.Generator, strong: bool = False) -> np.ndarray:
    image = random_flip_rotate(image, rng)
    image = color_jitter(image, rng, brightness=0.16 if strong else 0.08, contrast=0.22 if strong else 0.12)
    if strong and rng.random() < 0.8:
        image = sharpen_texture_regions(image, amount=float(rng.uniform(0.15, 0.35)))
    return np.clip(image, 0, 1).astype(np.float32)


def class_aware_oversample(df, majority_ratio_threshold: float = 0.25, seed: int = 42):
    """Duplicate minority-class rows so augmentation can be applied only in training."""
    rng = np.random.default_rng(seed)
    counts = df["label"].value_counts()
    majority = int(counts.max())
    minority_labels = counts[counts < majority_ratio_threshold * majority].index.tolist()
    rows = [df]
    for label in minority_labels:
        part = df[df["label"] == label]
        need = majority - len(part)
        if need > 0 and len(part) > 0:
            extra = part.sample(n=need, replace=True, random_state=int(rng.integers(0, 1_000_000))).copy()
            extra["augmented"] = True
            rows.append(extra)
    out = df.copy()
    out["augmented"] = False
    if len(rows) > 1:
        out = __import__("pandas").concat([out] + rows[1:], ignore_index=True)
    return out.sample(frac=1.0, random_state=seed).reset_index(drop=True)
