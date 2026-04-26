from __future__ import annotations

from pathlib import Path
import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint, CSVLogger, EarlyStopping, ReduceLROnPlateau

from hdfnet.data.manifest import combine_manifests
from hdfnet.data.splits import stratified_train_val_test_split
from hdfnet.data.augment import class_aware_oversample
from hdfnet.data.dataset import make_tf_dataset, compute_weights
from hdfnet.models.vgg import build_alc_vgg16_classifier
from hdfnet.models.resnet import build_dfe_resnet50_classifier
from hdfnet.models.vit import build_dpvit_classifier
from hdfnet.models.hdfnet import build_end_to_end_hdfnet
from hdfnet.utils.io import ensure_dir, save_json
from hdfnet.utils.seed import set_global_seed


def compile_model(model, lr: float):
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=lr, beta_1=0.9, beta_2=0.999, epsilon=1e-7),
        loss="categorical_crossentropy",
        metrics=["accuracy", tf.keras.metrics.AUC(name="auc", multi_label=True)],
    )
    return model


def callbacks_for(out_dir: Path, name: str, patience: int):
    ensure_dir(out_dir)
    return [
        ModelCheckpoint(str(out_dir / f"{name}_best.keras"), monitor="val_loss", save_best_only=True, verbose=1),
        CSVLogger(str(out_dir / f"{name}_log.csv")),
        ReduceLROnPlateau(monitor="val_loss", factor=0.1, patience=max(3, patience // 2), min_lr=1e-8, verbose=1),
        EarlyStopping(monitor="val_loss", patience=patience, restore_best_weights=True, verbose=1),
    ]


def build_model_by_name(name: str, config: dict):
    name = name.lower()
    if name in {"vgg", "alc-vgg16", "alc_vgg16"}:
        return build_alc_vgg16_classifier(config), int(config["training"].get("batch_size_cnn", 16)), int(config["training"].get("epochs_cnn", 50))
    if name in {"resnet", "dfe-resnet50", "dfe_resnet50"}:
        return build_dfe_resnet50_classifier(config), int(config["training"].get("batch_size_cnn", 16)), int(config["training"].get("epochs_cnn", 50))
    if name in {"vit", "dp-vit", "dp_vit"}:
        return build_dpvit_classifier(config), int(config["training"].get("batch_size_vit", 32)), int(config["training"].get("epochs_vit", 3))
    if name in {"hdfnet", "fusion", "end_to_end"}:
        return build_end_to_end_hdfnet(config), int(config["training"].get("batch_size_fusion", 16)), int(config["training"].get("epochs_fusion", 50))
    raise ValueError(f"Unknown model: {name}")


def train_model(config: dict, model_name: str = "hdfnet"):
    set_global_seed(int(config.get("seed", 42)))
    out_dir = ensure_dir(Path(config.get("output_dir", "runs/hdfnet")) / model_name)
    df = combine_manifests(config)
    train_df, val_df, test_df = stratified_train_val_test_split(df, config)
    train_balanced = class_aware_oversample(train_df, seed=int(config.get("seed", 42)))

    train_df.to_csv(out_dir / "train_original.csv", index=False)
    train_balanced.to_csv(out_dir / "train_balanced.csv", index=False)
    val_df.to_csv(out_dir / "val.csv", index=False)
    test_df.to_csv(out_dir / "test.csv", index=False)
    save_json({"classes": config["classes"], "counts": df["label"].value_counts().to_dict()}, out_dir / "dataset_summary.json")

    model, batch_size, epochs = build_model_by_name(model_name, config)
    model = compile_model(model, lr=float(config["training"].get("lr", 1e-4)))
    model.summary(print_fn=lambda s: open(out_dir / "model_summary.txt", "a", encoding="utf-8").write(s + "\n"))

    train_ds = make_tf_dataset(train_balanced, config, training=True, batch_size=batch_size)
    val_ds = make_tf_dataset(val_df, config, training=False, batch_size=batch_size)
    steps_per_epoch = max(1, len(train_balanced) // batch_size)
    val_steps = max(1, len(val_df) // batch_size)

    history = model.fit(
        train_ds,
        epochs=epochs,
        steps_per_epoch=steps_per_epoch,
        validation_data=val_ds,
        validation_steps=val_steps,
        callbacks=callbacks_for(out_dir, model_name, int(config["training"].get("early_stopping_patience", 12))),
        class_weight=compute_weights(train_df["label"].tolist(), config["classes"]),
    )
    model.save(out_dir / f"{model_name}_final.keras")
    return history
