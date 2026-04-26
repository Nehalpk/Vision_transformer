from __future__ import annotations

import tensorflow as tf
from tensorflow.keras import layers, Model


def dermatological_attention_module(x, reduction: int = 16, name: str = "dfe"):
    channels = int(x.shape[-1])
    avg = layers.GlobalAveragePooling2D(name=f"{name}_gap")(x)
    dense1 = layers.Dense(max(channels // reduction, 8), activation="relu", name=f"{name}_ca_fc1")(avg)
    dense2 = layers.Dense(channels, activation="sigmoid", name=f"{name}_ca_fc2")(dense1)
    channel_mask = layers.Reshape((1, 1, channels), name=f"{name}_channel_mask")(dense2)
    x_c = layers.Multiply(name=f"{name}_channel_refined")([x, channel_mask])

    avg_spatial = layers.Lambda(lambda t: tf.reduce_mean(t, axis=-1, keepdims=True), name=f"{name}_spatial_avg")(x_c)
    max_spatial = layers.Lambda(lambda t: tf.reduce_max(t, axis=-1, keepdims=True), name=f"{name}_spatial_max")(x_c)
    spatial = layers.Concatenate(axis=-1, name=f"{name}_spatial_concat")([avg_spatial, max_spatial])
    spatial_mask = layers.Conv2D(1, 7, padding="same", activation="sigmoid", name=f"{name}_spatial_mask")(spatial)
    x_s = layers.Multiply(name=f"{name}_spatial_refined")([x_c, spatial_mask])
    return x_s


def build_dfe_resnet50_feature_model(config: dict, name: str = "DFE_ResNet50_Feature") -> Model:
    image_size = int(config.get("image_size", 150))
    pretrained = bool(config.get("resnet", {}).get("pretrained", True))
    weights = "imagenet" if pretrained else None
    inputs = layers.Input(shape=(image_size, image_size, 3), name="image")
    # ResNet50 includes the 49 convolutional layers. We expose major stage endpoints.
    base = tf.keras.applications.ResNet50(include_top=False, weights=weights, input_tensor=inputs)
    c2 = base.get_layer("conv2_block3_out").output
    c3 = base.get_layer("conv3_block4_out").output
    c4 = base.get_layer("conv4_block6_out").output
    c5 = base.get_layer("conv5_block3_out").output
    a2 = dermatological_attention_module(c2, name="dfe_stage2")
    a3 = dermatological_attention_module(c3, name="dfe_stage3")
    a4 = dermatological_attention_module(c4, name="dfe_stage4")
    a5 = dermatological_attention_module(c5, name="dfe_stage5")
    pooled = []
    for i, t in enumerate([a2, a3, a4, a5], start=2):
        pooled.append(layers.GlobalAveragePooling2D(name=f"stage{i}_gap_out")(t))
    x = layers.Concatenate(name="dfe_multistage_concat")(pooled)
    x = layers.Dropout(float(config.get("resnet", {}).get("dropout", 0.1)), name="dfe_dropout")(x)
    feature = layers.Dense(512, activation="relu", name="dfe_resnet50_512_feature")(x)
    return Model(inputs, feature, name=name)


def build_dfe_resnet50_classifier(config: dict) -> Model:
    feature_model = build_dfe_resnet50_feature_model(config)
    out = layers.Dense(len(config["classes"]), activation="softmax", name="classification")(feature_model.output)
    return Model(feature_model.input, out, name="DFE_ResNet50_Classifier")
