from __future__ import annotations

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


class ClassToken(layers.Layer):
    def build(self, input_shape):
        self.cls = self.add_weight(
            name="class_token",
            shape=(1, 1, int(input_shape[-1])),
            initializer="random_normal",
            trainable=True,
        )

    def call(self, inputs):
        batch = tf.shape(inputs)[0]
        return tf.broadcast_to(self.cls, [batch, 1, tf.shape(self.cls)[-1]])


class Patches(layers.Layer):
    def __init__(self, patch_size: int, pad_size: int = 160, **kwargs):
        super().__init__(**kwargs)
        self.patch_size = int(patch_size)
        self.pad_size = int(pad_size)

    def call(self, images):
        images = tf.image.resize_with_pad(images, self.pad_size, self.pad_size)
        patches = tf.image.extract_patches(
            images=images,
            sizes=[1, self.patch_size, self.patch_size, 1],
            strides=[1, self.patch_size, self.patch_size, 1],
            rates=[1, 1, 1, 1],
            padding="VALID",
        )
        batch = tf.shape(images)[0]
        patch_dim = self.patch_size * self.patch_size * tf.shape(images)[-1]
        return tf.reshape(patches, [batch, -1, patch_dim])

    def get_config(self):
        cfg = super().get_config()
        cfg.update({"patch_size": self.patch_size, "pad_size": self.pad_size})
        return cfg


class PatchEncoder(layers.Layer):
    def __init__(self, num_patches: int, hidden_dim: int, **kwargs):
        super().__init__(**kwargs)
        self.num_patches = int(num_patches)
        self.hidden_dim = int(hidden_dim)
        self.projection = layers.Dense(hidden_dim)
        self.position_embedding = layers.Embedding(input_dim=num_patches, output_dim=hidden_dim)

    def call(self, patches):
        positions = tf.range(start=0, limit=self.num_patches, delta=1)
        return self.projection(patches) + self.position_embedding(positions)

    def get_config(self):
        cfg = super().get_config()
        cfg.update({"num_patches": self.num_patches, "hidden_dim": self.hidden_dim})
        return cfg


def mlp_block(x, hidden_units: int, out_units: int, dropout: float):
    x = layers.Dense(hidden_units, activation="gelu")(x)
    x = layers.Dropout(dropout)(x)
    x = layers.Dense(out_units)(x)
    x = layers.Dropout(dropout)(x)
    return x


def transformer_encoder(x, hidden_dim: int, mlp_dim: int, num_heads: int, dropout: float, name_prefix: str):
    skip = x
    x = layers.LayerNormalization(epsilon=1e-6, name=f"{name_prefix}_ln1")(x)
    x = layers.MultiHeadAttention(num_heads=num_heads, key_dim=hidden_dim, dropout=dropout, name=f"{name_prefix}_mha")(x, x)
    x = layers.Add(name=f"{name_prefix}_add1")([x, skip])
    skip = x
    x = layers.LayerNormalization(epsilon=1e-6, name=f"{name_prefix}_ln2")(x)
    x = mlp_block(x, mlp_dim, hidden_dim, dropout)
    x = layers.Add(name=f"{name_prefix}_add2")([x, skip])
    return x
