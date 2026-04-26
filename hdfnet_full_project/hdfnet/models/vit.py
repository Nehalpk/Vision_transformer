from __future__ import annotations

import tensorflow as tf
from tensorflow.keras import layers, Model
from .layers import ClassToken, Patches, PatchEncoder, transformer_encoder


def vit_feature_path(image_input, patch_size: int, config: dict, name: str):
    pad_size = int(config.get("pad_size", 160))
    hidden_dim = int(config["vit"].get("hidden_dim", 512))
    mlp_dim = int(config["vit"].get("mlp_dim", 3072))
    num_layers = int(config["vit"].get("num_layers", 12))
    num_heads = int(config["vit"].get("num_heads", 4))
    dropout = float(config["vit"].get("dropout", 0.1))
    num_patches = (pad_size // patch_size) ** 2

    x = Patches(patch_size, pad_size, name=f"{name}_patches")(image_input)
    x = PatchEncoder(num_patches, hidden_dim, name=f"{name}_patch_encoder")(x)
    cls = ClassToken(name=f"{name}_class_token")(x)
    x = layers.Concatenate(axis=1, name=f"{name}_with_cls")([cls, x])
    for i in range(num_layers):
        x = transformer_encoder(x, hidden_dim, mlp_dim, num_heads, dropout, name_prefix=f"{name}_block{i+1}")
    x = layers.LayerNormalization(epsilon=1e-6, name=f"{name}_final_ln")(x)
    return layers.Lambda(lambda t: t[:, 0, :], name=f"{name}_cls_feature")(x)


def build_dpvit_feature_model(config: dict, name: str = "DP_ViT_Feature") -> Model:
    image_size = int(config.get("image_size", 150))
    image = layers.Input(shape=(image_size, image_size, 3), name="image")
    selector = layers.Input(shape=(1,), dtype="int32", name="patch_selector")
    paths = []
    for ps in config["vit"].get("patch_sizes", [8, 16, 32]):
        paths.append(vit_feature_path(image, int(ps), config, name=f"vit_p{ps}"))
    stacked = layers.Lambda(lambda xs: tf.stack(xs, axis=1), name="stack_patch_features")(paths)  # B,3,D
    one_hot = layers.Lambda(lambda s: tf.one_hot(tf.squeeze(s, axis=-1), depth=len(paths)), name="patch_selector_onehot")(selector)
    one_hot = layers.Reshape((len(paths), 1), name="patch_selector_reshape")(one_hot)
    selected = layers.Multiply(name="select_dynamic_patch_feature")([stacked, one_hot])
    feature = layers.Lambda(lambda t: tf.reduce_sum(t, axis=1), name="dpvit_512_feature")(selected)
    return Model(inputs={"image": image, "patch_selector": selector}, outputs=feature, name=name)


def build_dpvit_classifier(config: dict) -> Model:
    feature_model = build_dpvit_feature_model(config)
    x = layers.Dropout(float(config["training"].get("dropout", 0.1)), name="dpvit_head_dropout")(feature_model.output)
    out = layers.Dense(len(config["classes"]), activation="softmax", name="classification")(x)
    return Model(feature_model.inputs, out, name="DP_ViT_Classifier")
