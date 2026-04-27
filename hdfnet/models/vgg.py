from __future__ import annotations

from tensorflow.keras import layers, Model


def conv_block(x, filters: int, n_conv: int, name: str):
    for i in range(n_conv):
        x = layers.Conv2D(filters, 3, padding="same", name=f"{name}_conv{i+1}")(x)
        x = layers.LeakyReLU(alpha=0.01, name=f"{name}_lrelu{i+1}")(x)
    x = layers.AveragePooling2D(pool_size=2, strides=2, name=f"{name}_avgpool")(x)
    return x


def build_alc_vgg16_feature_model(config: dict, name: str = "ALC_VGG16_Feature") -> Model:
    image_size = int(config.get("image_size", 150))
    dropout = float(config.get("vgg", {}).get("dropout", 0.1))
    inputs = layers.Input(shape=(image_size, image_size, 3), name="image")
    x = conv_block(inputs, 64, 2, "block1")
    x = conv_block(x, 128, 2, "block2")
    x = conv_block(x, 256, 3, "block3")
    x = conv_block(x, 512, 3, "block4")
    x = conv_block(x, 512, 3, "block5")
    x = layers.Flatten(name="flatten")(x)
    x = layers.Dense(4096, name="fc1")(x)
    x = layers.LeakyReLU(alpha=0.01, name="fc1_lrelu")(x)
    x = layers.Dropout(dropout, name="fc1_dropout")(x)
    x = layers.Dense(4096, name="fc2")(x)
    x = layers.LeakyReLU(alpha=0.01, name="fc2_lrelu")(x)
    x = layers.Dropout(dropout, name="fc2_dropout")(x)
    feature = layers.Dense(512, name="alc_vgg16_512_feature")(x)
    return Model(inputs, feature, name=name)


def build_alc_vgg16_classifier(config: dict) -> Model:
    feature_model = build_alc_vgg16_feature_model(config)
    x = layers.LeakyReLU(alpha=0.01, name="feature_lrelu")(feature_model.output)
    out = layers.Dense(len(config["classes"]), activation="softmax", name="classification")(x)
    return Model(feature_model.input, out, name="ALC_VGG16_Classifier")
