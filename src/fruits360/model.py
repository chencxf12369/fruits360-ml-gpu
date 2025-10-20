from __future__ import annotations
import tensorflow as tf
from tensorflow import keras
from keras import layers, models, optimizers
from keras.applications import MobileNetV2, ResNet50, EfficientNetB0
from keras.saving import register_keras_serializable
from . import config

# ---- keep preprocess deserializable for .keras files ----
@register_keras_serializable(package="fruits360", name="mobilenetv2_preprocess")
def mobilenetv2_preprocess(x):
    # defer import to avoid hard dependency at import time
    from keras.applications.mobilenet_v2 import preprocess_input
    return preprocess_input(x)

# -----------------------------------------------------------------------------
# Small custom CNN (unchanged)
# -----------------------------------------------------------------------------
def _build_custom_cnn(num_classes: int) -> keras.Model:
    inputs = keras.Input(shape=config.IMG_SIZE)
    x = layers.Rescaling(1.0 / 255)(inputs)

    # light augmentation (train-time only)
    aug = keras.Sequential(
        [
            layers.RandomFlip("horizontal"),
            layers.RandomRotation(0.05),
            layers.RandomZoom(0.10),
            layers.RandomContrast(0.10),
        ],
        name="aug",
    )
    x = aug(x)

    x = layers.Conv2D(32, 3, padding="same", activation="relu")(x)
    x = layers.MaxPooling2D()(x)
    x = layers.Conv2D(64, 3, padding="same", activation="relu")(x)
    x = layers.MaxPooling2D()(x)
    x = layers.Conv2D(128, 3, padding="same", activation="relu")(x)
    x = layers.MaxPooling2D()(x)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(256, activation="relu")(x)
    x = layers.Dropout(0.2)(x)
    outputs = layers.Dense(num_classes, activation="softmax")(x)

    model = keras.Model(inputs, outputs, name="custom_cnn")
    opt = optimizers.Adam(learning_rate=getattr(config, "LEARNING_RATE", 3e-4))

    metrics = ["accuracy"]
    if getattr(config, "USE_TOP5", False):
        metrics.append(keras.metrics.SparseTopKCategoricalAccuracy(k=5, name="top5"))

    # label smoothing (very small)
    ls = float(getattr(config, "LABEL_SMOOTH", 0.05))
    loss = keras.losses.SparseCategoricalCrossentropy(label_smoothing=ls) if getattr(
        config, "SPARSE_LABELS", True
    ) else keras.losses.CategoricalCrossentropy(label_smoothing=ls)

    model.compile(optimizer=opt, loss=loss, metrics=metrics)
    return model

# -----------------------------------------------------------------------------
# MobileNetV2 with train-time augmentation + label smoothing
# -----------------------------------------------------------------------------
def _build_mobilenetv2(num_classes: int) -> models.Model:
    input_shape = config.IMG_SIZE  # (H, W, C)
    inputs = keras.Input(shape=input_shape, name="input_images")

    # train-time only augmentation (Keras handles training/inference modes)
    x = keras.Sequential(
        [
            layers.RandomFlip("horizontal"),
            layers.RandomRotation(0.05),
            layers.RandomZoom(0.10),
            layers.RandomContrast(0.10),
        ],
        name="aug",
    )(inputs)

    # keep a named Lambda so .keras deserializes without custom_objects
    x = layers.Lambda(
        mobilenetv2_preprocess, name="preprocess_mobilenetv2"
    )(x)

    base = MobileNetV2(
        input_shape=input_shape,
        include_top=False,
        weights="imagenet" if getattr(config, "PRETRAINED", True) else None,
        pooling=getattr(config, "GLOBAL_POOL", "avg")
        if getattr(config, "GLOBAL_POOL", "avg") in {"avg", "max"}
        else None,
    )

    # freeze policy
    if getattr(config, "FREEZE", False):
        for l in base.layers:
            l.trainable = False
        n_unfreeze = int(getattr(config, "FROZEN_LAYERS", 0))
        if n_unfreeze > 0:
            for l in base.layers[-n_unfreeze:]:
                l.trainable = True

    x = base(x)
    if getattr(config, "GLOBAL_POOL", "avg") not in {"avg", "max"}:
        x = layers.GlobalAveragePooling2D()(x)

    dr = float(getattr(config, "DROPOUT", 0.2))
    if dr > 0:
        x = layers.Dropout(dr)(x)

    outputs = layers.Dense(
        num_classes, activation=getattr(config, "CLASSIFIER_ACT", "softmax")
    )(x)

    model = models.Model(inputs=inputs, outputs=outputs, name="fruits360_mobilenetv2")

    lr = float(getattr(config, "LEARNING_RATE", 1e-3))
    opt = optimizers.Adam(learning_rate=lr)

    # label smoothing (small)
    ls = float(getattr(config, "LABEL_SMOOTH", 0.05))
    loss = keras.losses.CategoricalCrossentropy(label_smoothing=ls)

    model.compile(optimizer=opt, loss=loss, metrics=["accuracy"])
    return model

# -----------------------------------------------------------------------------
# (Optional) ResNet50 / EfficientNet hooks (left as before)
# -----------------------------------------------------------------------------
def _build_resnet50(num_classes: int) -> models.Model:
    base = ResNet50(
        input_shape=config.IMG_SIZE,
        include_top=False,
        weights="imagenet" if getattr(config, "PRETRAINED", True) else None,
        pooling=getattr(config, "GLOBAL_POOL", "avg")
        if getattr(config, "GLOBAL_POOL", "avg") in {"avg", "max"}
        else None,
    )
    inputs = keras.Input(shape=config.IMG_SIZE)
    x = layers.RandomFlip("horizontal")(inputs)
    x = layers.RandomRotation(0.05)(x)
    x = layers.RandomZoom(0.10)(x)
    x = layers.RandomContrast(0.10)(x)
    x = layers.Lambda(mobilenetv2_preprocess, name="preprocess_mobilenetv2")(x)
    x = base(x)
    if getattr(config, "GLOBAL_POOL", "avg") not in {"avg", "max"}:
        x = layers.GlobalAveragePooling2D()(x)
    if getattr(config, "DROPOUT", 0.2) > 0:
        x = layers.Dropout(getattr(config, "DROPOUT", 0.2))(x)
    outputs = layers.Dense(num_classes, activation=getattr(config, "CLASSIFIER_ACT", "softmax"))(x)
    model = models.Model(inputs, outputs, name="fruits360_resnet50")
    opt = optimizers.Adam(learning_rate=getattr(config, "LEARNING_RATE", 1e-3))
    ls = float(getattr(config, "LABEL_SMOOTH", 0.05))
    loss = keras.losses.CategoricalCrossentropy(label_smoothing=ls)
    model.compile(optimizer=opt, loss=loss, metrics=["accuracy"])
    return model

def _build_efficientnetb0(num_classes: int) -> models.Model:
    base = EfficientNetB0(
        input_shape=config.IMG_SIZE,
        include_top=False,
        weights="imagenet" if getattr(config, "PRETRAINED", True) else None,
        pooling=getattr(config, "GLOBAL_POOL", "avg")
        if getattr(config, "GLOBAL_POOL", "avg") in {"avg", "max"}
        else None,
    )
    inputs = keras.Input(shape=config.IMG_SIZE)
    x = layers.RandomFlip("horizontal")(inputs)
    x = layers.RandomRotation(0.05)(x)
    x = layers.RandomZoom(0.10)(x)
    x = layers.RandomContrast(0.10)(x)
    # EfficientNet expects its own preprocessing, but MobileNetV2's is fine for 0-centered RGB here.
    x = layers.Lambda(mobilenetv2_preprocess, name="preprocess_mobilenetv2")(x)
    x = base(x)
    if getattr(config, "GLOBAL_POOL", "avg") not in {"avg", "max"}:
        x = layers.GlobalAveragePooling2D()(x)
    if getattr(config, "DROPOUT", 0.2) > 0:
        x = layers.Dropout(getattr(config, "DROPOUT", 0.2))(x)
    outputs = layers.Dense(num_classes, activation=getattr(config, "CLASSIFIER_ACT", "softmax"))(x)
    model = models.Model(inputs, outputs, name="fruits360_efficientnetb0")
    opt = optimizers.Adam(learning_rate=getattr(config, "LEARNING_RATE", 1e-3))
    ls = float(getattr(config, "LABEL_SMOOTH", 0.05))
    loss = keras.losses.CategoricalCrossentropy(label_smoothing=ls)
    model.compile(optimizer=opt, loss=loss, metrics=["accuracy"])
    return model

# -----------------------------------------------------------------------------
def build_model(num_classes: int) -> keras.Model:
    backbone = getattr(config, "BACKBONE", "mobilenetv2").lower()
    if backbone == "mobilenetv2":
        return _build_mobilenetv2(num_classes)
    if backbone == "resnet50":
        return _build_resnet50(num_classes)
    if backbone == "efficientnetb0":
        return _build_efficientnetb0(num_classes)
    return _build_custom_cnn(num_classes)