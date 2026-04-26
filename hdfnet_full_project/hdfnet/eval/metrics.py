from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve, ConfusionMatrixDisplay
import tensorflow as tf

from hdfnet.data.dataset import make_tf_dataset


def predict_dataframe(model, df: pd.DataFrame, config: dict, batch_size: int = 16):
    ds = make_tf_dataset(df, config, training=False, batch_size=batch_size)
    steps = int(np.ceil(len(df) / batch_size))
    probs = model.predict(ds, steps=steps, verbose=1)[: len(df)]
    y_true = np.array([config["classes"].index(x) for x in df["label"].tolist()])
    y_pred = probs.argmax(axis=1)
    return y_true, y_pred, probs


def save_classification_outputs(model_path: str | Path, split_csv: str | Path, config: dict, out_dir: str | Path, batch_size: int = 16):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    model = tf.keras.models.load_model(model_path, compile=False)
    df = pd.read_csv(split_csv)
    y_true, y_pred, probs = predict_dataframe(model, df, config, batch_size=batch_size)
    labels = list(range(len(config["classes"])))
    report = classification_report(y_true, y_pred, target_names=config["classes"], output_dict=True, zero_division=0)
    pd.DataFrame(report).T.to_csv(out_dir / "classification_report.csv")

    cm = confusion_matrix(y_true, y_pred, labels=labels)
    np.savetxt(out_dir / "confusion_matrix.csv", cm, delimiter=",", fmt="%d")
    disp = ConfusionMatrixDisplay(cm, display_labels=config["classes"])
    disp.plot(cmap="Blues", values_format="d")
    plt.title("Confusion Matrix")
    plt.tight_layout()
    plt.savefig(out_dir / "confusion_matrix.png", dpi=200)
    plt.close()

    y_onehot = np.eye(len(config["classes"]))[y_true]
    auc_macro = roc_auc_score(y_onehot, probs, multi_class="ovr", average="macro")
    with open(out_dir / "auc.txt", "w", encoding="utf-8") as f:
        f.write(f"macro_auc_ovr={auc_macro:.6f}\n")
        for i, cls in enumerate(config["classes"]):
            auc_i = roc_auc_score(y_onehot[:, i], probs[:, i])
            f.write(f"{cls}_auc={auc_i:.6f}\n")
            fpr, tpr, _ = roc_curve(y_onehot[:, i], probs[:, i])
            plt.plot(fpr, tpr, label=f"{cls} AUC={auc_i:.3f}")
    plt.plot([0, 1], [0, 1], linestyle="--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("AUC-ROC Curve")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "roc_curve.png", dpi=200)
    plt.close()

    pred_df = df.copy()
    for i, cls in enumerate(config["classes"]):
        pred_df[f"prob_{cls}"] = probs[:, i]
    pred_df["pred_label"] = [config["classes"][i] for i in y_pred]
    pred_df.to_csv(out_dir / "predictions.csv", index=False)
