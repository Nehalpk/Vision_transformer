from __future__ import annotations

import numpy as np
import tensorflow as tf
from sklearn.utils.class_weight import compute_class_weight
from .preprocess import preprocess_path
from .augment import augment_image


def class_to_index(classes: list[str]) -> dict[str, int]:
    return {c: i for i, c in enumerate(classes)}


def compute_weights(labels, classes):
    idx = np.array([class_to_index(classes)[x] for x in labels])
    weights = compute_class_weight(class_weight="balanced", classes=np.arange(len(classes)), y=idx)
    return {int(i): float(w) for i, w in enumerate(weights)}


def make_generator(df, config: dict, training: bool = False):
    classes = config["classes"]
    c2i = class_to_index(classes)
    image_size = int(config.get("image_size", 150))
    thresholds = tuple(config["vit"].get("coverage_thresholds", [0.25, 0.50]))
    seed = int(config.get("seed", 42))
    paths = df["image_path"].tolist()
    labels = df["label"].tolist()
    augmented = df.get("augmented", False)
    if hasattr(augmented, "tolist"):
        augmented = augmented.tolist()
    else:
        augmented = [False] * len(paths)

    def gen():
        rng = np.random.default_rng(seed)
        order = np.arange(len(paths))
        while True:
            if training:
                rng.shuffle(order)
            for i in order:
                image, coverage, patch_size = preprocess_path(paths[i], image_size=image_size, thresholds=thresholds)
                if training and (augmented[i] or rng.random() < 0.35):
                    image = augment_image(image, rng, strong=bool(augmented[i]))
                y = np.zeros(len(classes), dtype=np.float32)
                y[c2i[labels[i]]] = 1.0
                yield {
                    "image": image.astype(np.float32),
                    "patch_selector": np.array([0 if patch_size == 8 else 1 if patch_size == 16 else 2], dtype=np.int32),
                    "coverage": np.array([coverage], dtype=np.float32),
                }, y
            if not training:
                break
    return gen


def make_tf_dataset(df, config: dict, training: bool, batch_size: int):
    image_size = int(config.get("image_size", 150))
    output_signature = (
        {
            "image": tf.TensorSpec(shape=(image_size, image_size, 3), dtype=tf.float32),
            "patch_selector": tf.TensorSpec(shape=(1,), dtype=tf.int32),
            "coverage": tf.TensorSpec(shape=(1,), dtype=tf.float32),
        },
        tf.TensorSpec(shape=(len(config["classes"]),), dtype=tf.float32),
    )
    ds = tf.data.Dataset.from_generator(make_generator(df, config, training), output_signature=output_signature)
    ds = ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return ds
