from __future__ import annotations

from pathlib import Path
from typing import Iterable
import pandas as pd

TARGET_CLASSES = ["BKL", "MEL", "NV", "BCC"]

ISIC2019_MAP = {"MEL": "MEL", "NV": "NV", "BKL": "BKL", "BCC": "BCC"}
PAD_UFES_MAP = {"MEL": "MEL", "NEV": "NV", "SEK": "BKL", "BCC": "BCC"}
ISIC2020_MAP = {"melanoma": "MEL", "nevus": "NV", "benign": "BKL", "seborrheic keratosis": "BKL", "BKL": "BKL", "MEL": "MEL"}


def _first_existing(paths: Iterable[Path]) -> Path | None:
    for p in paths:
        if p.exists():
            return p
    return None


def manifest_from_folder(root: str | Path, classes: list[str] | None = None) -> pd.DataFrame:
    """Build a manifest from root/class_name/image files."""
    root = Path(root)
    classes = classes or TARGET_CLASSES
    rows = []
    for label in classes:
        class_dir = root / label
        for ext in ("*.jpg", "*.jpeg", "*.png", "*.JPG", "*.JPEG", "*.PNG"):
            for p in class_dir.glob(ext):
                rows.append({"image_path": str(p), "label": label, "source": "folder"})
    if not rows:
        raise FileNotFoundError(f"No images found under {root}/<class>/*.jpg|png")
    return pd.DataFrame(rows)


def build_isic2019_manifest(metadata_csv: str | Path, image_dir: str | Path) -> pd.DataFrame:
    """Build manifest for ISIC 2019 ground-truth CSV with one-hot diagnosis columns."""
    meta = pd.read_csv(metadata_csv)
    image_dir = Path(image_dir)
    rows = []
    for _, r in meta.iterrows():
        image_id = str(r.get("image", r.get("image_name", "")))
        label = None
        for src, dst in ISIC2019_MAP.items():
            if src in r and int(r[src]) == 1:
                label = dst
                break
        if label is None:
            continue
        path = _first_existing([image_dir / f"{image_id}.jpg", image_dir / f"{image_id}.png", image_dir / image_id])
        if path:
            rows.append({"image_path": str(path), "label": label, "source": "ISIC2019"})
    return pd.DataFrame(rows)


def build_isic2020_manifest(metadata_csv: str | Path, image_dir: str | Path) -> pd.DataFrame:
    """Build manifest for ISIC 2020 train CSV. Keeps MEL and maps benign examples to BKL if no detailed label exists."""
    meta = pd.read_csv(metadata_csv)
    image_dir = Path(image_dir)
    rows = []
    for _, r in meta.iterrows():
        image_id = str(r.get("image_name", r.get("image", "")))
        diagnosis = str(r.get("diagnosis", "")).strip()
        target = r.get("target", None)
        if target is not None and int(target) == 1:
            label = "MEL"
        else:
            label = ISIC2020_MAP.get(diagnosis, "BKL")
        if label not in TARGET_CLASSES:
            continue
        path = _first_existing([image_dir / f"{image_id}.jpg", image_dir / f"{image_id}.png", image_dir / image_id])
        if path:
            rows.append({"image_path": str(path), "label": label, "source": "ISIC2020"})
    return pd.DataFrame(rows)


def build_pad_ufes_manifest(metadata_csv: str | Path, image_dir: str | Path) -> pd.DataFrame:
    meta = pd.read_csv(metadata_csv)
    image_dir = Path(image_dir)
    rows = []
    for _, r in meta.iterrows():
        image_name = str(r.get("img_id", r.get("image", r.get("image_name", ""))))
        src_label = str(r.get("diagnostic", r.get("label", ""))).upper().strip()
        label = PAD_UFES_MAP.get(src_label)
        if label is None:
            continue
        path = _first_existing([image_dir / image_name, image_dir / f"{image_name}.jpg", image_dir / f"{image_name}.png"])
        if path:
            rows.append({"image_path": str(path), "label": label, "source": "PAD_UFES"})
    return pd.DataFrame(rows)


def combine_manifests(config: dict) -> pd.DataFrame:
    data_cfg = config["data"]
    manifest_csv = data_cfg.get("manifest_csv")
    if manifest_csv and Path(manifest_csv).exists():
        df = pd.read_csv(manifest_csv)
    else:
        frames = []
        if data_cfg.get("isic2019_metadata_csv") and data_cfg.get("isic2019_image_dir"):
            frames.append(build_isic2019_manifest(data_cfg["isic2019_metadata_csv"], data_cfg["isic2019_image_dir"]))
        if data_cfg.get("isic2020_metadata_csv") and data_cfg.get("isic2020_image_dir"):
            frames.append(build_isic2020_manifest(data_cfg["isic2020_metadata_csv"], data_cfg["isic2020_image_dir"]))
        if data_cfg.get("pad_ufes_metadata_csv") and data_cfg.get("pad_ufes_image_dir"):
            frames.append(build_pad_ufes_manifest(data_cfg["pad_ufes_metadata_csv"], data_cfg["pad_ufes_image_dir"]))
        if not frames:
            df = manifest_from_folder(data_cfg.get("image_root", "data/images"), config.get("classes", TARGET_CLASSES))
        else:
            df = pd.concat(frames, ignore_index=True)
    df = df[df["label"].isin(config.get("classes", TARGET_CLASSES))].copy()
    df = df.drop_duplicates("image_path").reset_index(drop=True)
    if df.empty:
        raise ValueError("Manifest is empty after harmonization.")
    return df
