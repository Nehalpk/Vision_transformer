from __future__ import annotations

from pathlib import Path
import json
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score, classification_report, confusion_matrix

from hdfnet.data.manifest import combine_manifests
from hdfnet.data.augment import class_aware_oversample
from hdfnet.data.dataset import make_tf_dataset, compute_weights
from hdfnet.training.train import build_model_by_name, compile_model, callbacks_for
from hdfnet.utils.io import ensure_dir, save_json
from hdfnet.utils.seed import set_global_seed


def _stratification_key(df: pd.DataFrame, by_source: bool = True) -> pd.Series:
    """Return a robust stratification key for K-fold splitting.
    For rare source/class pairs we fall back to label-only stratification so
    StratifiedKFold does not fail.
    """
    if by_source and "source" in df.columns:
        key = df["label"].astype(str) + "__" + df["source"].astype(str)
        counts = key.value_counts()
        rare = key.map(counts) < 2
        key.loc[rare] = df.loc[rare, "label"].astype(str)
        return key
    return df["label"].astype(str)


def _fold_metrics(y_true: np.ndarray, probs: np.ndarray, classes: list[str]) -> dict:
    y_pred = probs.argmax(axis=1)
    y_onehot = np.eye(len(classes))[y_true]
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="macro", zero_division=0
    )
    out = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_precision": float(precision),
        "macro_recall": float(recall),
        "macro_f1": float(f1),
    }
    try:
        out["macro_auc_ovr"] = float(roc_auc_score(y_onehot, probs, multi_class="ovr", average="macro"))
        for i, cls in enumerate(classes):
            # Per-class AUC can fail if a fold has no positive examples of a class.
            out[f"auc_{cls}"] = float(roc_auc_score(y_onehot[:, i], probs[:, i]))
    except ValueError:
        out["macro_auc_ovr"] = float("nan")
    return out


def _predict_model(model: tf.keras.Model, df: pd.DataFrame, config: dict, batch_size: int) -> tuple[np.ndarray, np.ndarray]:
    ds = make_tf_dataset(df, config, training=False, batch_size=batch_size)
    steps = int(np.ceil(len(df) / batch_size))
    probs = model.predict(ds, steps=steps, verbose=1)[: len(df)]
    y_true = np.array([config["classes"].index(x) for x in df["label"].tolist()], dtype=np.int64)
    return y_true, probs


def _write_fold_outputs(
    fold_dir: Path,
    val_df: pd.DataFrame,
    y_true: np.ndarray,
    probs: np.ndarray,
    classes: list[str],
) -> dict:
    y_pred = probs.argmax(axis=1)
    metrics = _fold_metrics(y_true, probs, classes)
    save_json(metrics, fold_dir / "metrics.json")

    report = classification_report(y_true, y_pred, target_names=classes, output_dict=True, zero_division=0)
    pd.DataFrame(report).T.to_csv(fold_dir / "classification_report.csv")

    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(classes))))
    np.savetxt(fold_dir / "confusion_matrix.csv", cm, delimiter=",", fmt="%d")

    pred_df = val_df.copy()
    for i, cls in enumerate(classes):
        pred_df[f"prob_{cls}"] = probs[:, i]
    pred_df["pred_label"] = [classes[i] for i in y_pred]
    pred_df.to_csv(fold_dir / "predictions.csv", index=False)
    return metrics


def _summarize_folds(fold_rows: list[dict], out_dir: Path) -> pd.DataFrame:
    df = pd.DataFrame(fold_rows)
    df.to_csv(out_dir / "kfold_metrics_by_fold.csv", index=False)

    numeric_cols = [c for c in df.columns if c not in {"fold"} and pd.api.types.is_numeric_dtype(df[c])]
    rows = []
    for col in numeric_cols:
        values = df[col].astype(float).to_numpy()
        rows.append({
            "metric": col,
            "mean": float(np.nanmean(values)),
            "std": float(np.nanstd(values, ddof=1)) if np.sum(~np.isnan(values)) > 1 else 0.0,
            "min": float(np.nanmin(values)),
            "max": float(np.nanmax(values)),
        })
    summary = pd.DataFrame(rows)
    summary.to_csv(out_dir / "kfold_metrics_summary.csv", index=False)
    return summary


def train_kfold(config: dict, model_name: str = "hdfnet", n_splits: int | None = None):
    """Train model using stratified K-fold CV.

    Mapping:
    - DP-ViT: use --n-folds 4 to match the configuration tables.
    - HDFNet final generalization: use --n-folds 10 to match the results text.
    """
    seed = int(config.get("seed", 42))
    set_global_seed(seed)

    kcfg = config.get("kfold", {})
    n_splits = int(n_splits or kcfg.get("n_splits", 4))
    if n_splits < 2:
        raise ValueError("n_splits must be at least 2")

    out_dir = ensure_dir(Path(config.get("output_dir", "runs/hdfnet")) / f"{model_name}_kfold_{n_splits}")
    df = combine_manifests(config).reset_index(drop=True)
    classes = config["classes"]

    key = _stratification_key(df, by_source=bool(kcfg.get("stratify_by_source", True)))
    counts = key.value_counts()
    if counts.min() < n_splits:
        # Fallback when a source-class stratum is smaller than K.
        key = df["label"].astype(str)
        label_counts = key.value_counts()
        if label_counts.min() < n_splits:
            raise ValueError(
                f"Cannot run {n_splits}-fold CV: smallest class has {int(label_counts.min())} samples. "
                "Use fewer folds or add more samples."
            )

    df.to_csv(out_dir / "full_manifest_used.csv", index=False)
    save_json({
        "model": model_name,
        "n_splits": n_splits,
        "seed": seed,
        "classes": classes,
        "total_images": int(len(df)),
        "label_counts": df["label"].value_counts().to_dict(),
        "source_counts": df["source"].value_counts().to_dict() if "source" in df.columns else {},
    }, out_dir / "kfold_setup.json")

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    fold_rows = []

    for fold, (train_idx, val_idx) in enumerate(skf.split(df, key), start=1):
        fold_dir = ensure_dir(out_dir / f"fold_{fold:02d}")
        train_df = df.iloc[train_idx].reset_index(drop=True)
        val_df = df.iloc[val_idx].reset_index(drop=True)
        train_balanced = class_aware_oversample(
            train_df,
            majority_ratio_threshold=float(config.get("augmentation", {}).get("minority_threshold", 0.25)),
            seed=seed + fold,
        )

        train_df.to_csv(fold_dir / "train_original.csv", index=False)
        train_balanced.to_csv(fold_dir / "train_balanced.csv", index=False)
        val_df.to_csv(fold_dir / "val.csv", index=False)

        model, batch_size, epochs = build_model_by_name(model_name, config)
        model = compile_model(model, lr=float(config["training"].get("lr", 1e-4)))
        model.summary(print_fn=lambda s, p=fold_dir / "model_summary.txt": open(p, "a", encoding="utf-8").write(s + "\n"))

        train_ds = make_tf_dataset(train_balanced, config, training=True, batch_size=batch_size)
        val_ds = make_tf_dataset(val_df, config, training=False, batch_size=batch_size)
        steps_per_epoch = max(1, len(train_balanced) // batch_size)
        val_steps = max(1, int(np.ceil(len(val_df) / batch_size)))

        history = model.fit(
            train_ds,
            epochs=epochs,
            steps_per_epoch=steps_per_epoch,
            validation_data=val_ds,
            validation_steps=val_steps,
            callbacks=callbacks_for(fold_dir, f"{model_name}_fold_{fold:02d}", int(config["training"].get("early_stopping_patience", 12))),
            class_weight=compute_weights(train_df["label"].tolist(), classes),
            verbose=1,
        )
        pd.DataFrame(history.history).to_csv(fold_dir / "history.csv", index=False)
        model.save(fold_dir / f"{model_name}_fold_{fold:02d}_final.keras")

        y_true, probs = _predict_model(model, val_df, config, batch_size=batch_size)
        metrics = _write_fold_outputs(fold_dir, val_df, y_true, probs, classes)
        metrics["fold"] = fold
        fold_rows.append(metrics)

        # Clear graph between folds to reduce GPU memory fragmentation.
        del model
        tf.keras.backend.clear_session()

    summary = _summarize_folds(fold_rows, out_dir)
    with open(out_dir / "README_KFOLD_RESULTS.txt", "w", encoding="utf-8") as f:
        f.write("K-fold training completed.\n")
        f.write(f"Model: {model_name}\n")
        f.write(f"Folds: {n_splits}\n\n")
        f.write(summary.to_string(index=False))
        f.write("\n")
    return summary
