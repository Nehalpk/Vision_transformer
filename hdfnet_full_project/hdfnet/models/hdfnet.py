from __future__ import annotations

from tensorflow.keras import layers, Model
from .vgg import build_alc_vgg16_feature_model
from .resnet import build_dfe_resnet50_feature_model
from .vit import build_dpvit_feature_model


def build_hdfnet_from_features(config: dict) -> Model:
    inputs = layers.Input(shape=(1536,), name="fused_feature_vector")
    x = layers.Dense(int(config["hdfnet"].get("hidden_units", 1024)), activation="relu", name="hdfnet_dense_1024")(inputs)
    x = layers.BatchNormalization(name="hdfnet_bn")(x)
    x = layers.Dropout(float(config["hdfnet"].get("dropout", 0.1)), name="hdfnet_dropout")(x)
    outputs = layers.Dense(len(config["classes"]), activation="softmax", name="classification")(x)
    return Model(inputs, outputs, name="HDFNet_Feature_Classifier")


def build_end_to_end_hdfnet(config: dict) -> Model:
    vgg = build_alc_vgg16_feature_model(config)
    resnet = build_dfe_resnet50_feature_model(config)
    vit = build_dpvit_feature_model(config)

    image = layers.Input(shape=(int(config.get("image_size", 150)), int(config.get("image_size", 150)), 3), name="image")
    selector = layers.Input(shape=(1,), dtype="int32", name="patch_selector")

    f_vgg = vgg(image)
    f_res = resnet(image)
    f_vit = vit({"image": image, "patch_selector": selector})
    fused = layers.Concatenate(name="fused_1536_feature")([f_vgg, f_res, f_vit])
    x = layers.Dense(int(config["hdfnet"].get("hidden_units", 1024)), activation="relu", name="hdfnet_dense_1024")(fused)
    x = layers.BatchNormalization(name="hdfnet_bn")(x)
    x = layers.Dropout(float(config["hdfnet"].get("dropout", 0.1)), name="hdfnet_dropout")(x)
    outputs = layers.Dense(len(config["classes"]), activation="softmax", name="classification")(x)
    return Model(inputs={"image": image, "patch_selector": selector}, outputs=outputs, name="HDFNet_EndToEnd")
