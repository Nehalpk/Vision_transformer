from __future__ import annotations

from pathlib import Path
import cv2
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

from hdfnet.data.preprocess import read_rgb, normalize_image, lesion_coverage, patch_size_from_coverage


def find_last_conv_layer(model: tf.keras.Model) -> str:
    for layer in reversed(model.layers):
        if isinstance(layer, tf.keras.layers.Conv2D):
            return layer.name
        if hasattr(layer, "layers"):
            try:
                return find_last_conv_layer(layer)
            except ValueError:
                pass
    raise ValueError("No Conv2D layer found for Grad-CAM.")


def gradcam_heatmap(model, image_array, patch_selector, class_index=None, layer_name=None):
    if layer_name is None:
        layer_name = find_last_conv_layer(model)
    conv_layer = model.get_layer(layer_name)
    grad_model = tf.keras.Model(model.inputs, [conv_layer.output, model.output])
    with tf.GradientTape() as tape:
        conv_out, preds = grad_model({"image": image_array, "patch_selector": patch_selector}, training=False)
        if class_index is None:
            class_index = tf.argmax(preds[0])
        loss = preds[:, class_index]
    grads = tape.gradient(loss, conv_out)
    weights = tf.reduce_mean(grads, axis=(0, 1, 2))
    cam = tf.reduce_sum(tf.multiply(weights, conv_out[0]), axis=-1)
    cam = np.maximum(cam.numpy(), 0)
    if cam.max() > 0:
        cam = cam / cam.max()
    return cam, preds.numpy()[0]


def save_gradcam(model_path: str, image_path: str, config: dict, out_path: str, class_index=None, layer_name=None):
    model = tf.keras.models.load_model(model_path, compile=False)
    rgb = read_rgb(image_path, int(config.get("image_size", 150)))
    image = normalize_image(rgb)
    cov = lesion_coverage(rgb)
    ps = patch_size_from_coverage(cov, tuple(config["vit"].get("coverage_thresholds", [0.25, 0.50])))
    selector = 0 if ps == 8 else 1 if ps == 16 else 2
    cam, probs = gradcam_heatmap(
        model,
        image[None, ...].astype(np.float32),
        np.array([[selector]], dtype=np.int32),
        class_index=class_index,
        layer_name=layer_name,
    )
    cam_resized = cv2.resize(cam, (rgb.shape[1], rgb.shape[0]))
    heat = cv2.applyColorMap(np.uint8(255 * cam_resized), cv2.COLORMAP_JET)
    heat = cv2.cvtColor(heat, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    overlay = np.clip(0.55 * image + 0.45 * heat, 0, 1)

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 3, figsize=(10, 3))
    axes[0].imshow(image); axes[0].set_title("Original"); axes[0].axis("off")
    axes[1].imshow(cam_resized, cmap="jet"); axes[1].set_title("Grad-CAM"); axes[1].axis("off")
    axes[2].imshow(overlay); axes[2].set_title(f"Overlay: {config['classes'][int(np.argmax(probs))]}"); axes[2].axis("off")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()
    return probs
