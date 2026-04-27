from __future__ import annotations

import pandas as pd
from sklearn.model_selection import train_test_split


def stratified_train_val_test_split(df: pd.DataFrame, config: dict) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    seed = int(config.get("seed", 42))
    split = config.get("split", {})
    test_size = float(split.get("test", 0.10))
    val_size = float(split.get("val", 0.10))
    strat_cols = ["label"]
    if split.get("stratify_by_source", True) and "source" in df.columns:
        df = df.copy()
        df["_strat"] = df["label"].astype(str) + "__" + df["source"].astype(str)
        counts = df["_strat"].value_counts()
        df.loc[df["_strat"].map(counts) < 3, "_strat"] = df.loc[df["_strat"].map(counts) < 3, "label"]
        stratify = df["_strat"]
    else:
        stratify = df["label"]
    train_val, test = train_test_split(df, test_size=test_size, random_state=seed, stratify=stratify)
    relative_val = val_size / (1.0 - test_size)
    stratify_tv = train_val["_strat"] if "_strat" in train_val else train_val["label"]
    train, val = train_test_split(train_val, test_size=relative_val, random_state=seed, stratify=stratify_tv)
    return train.drop(columns=["_strat"], errors="ignore"), val.drop(columns=["_strat"], errors="ignore"), test.drop(columns=["_strat"], errors="ignore")
