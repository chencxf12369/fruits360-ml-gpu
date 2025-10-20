# src/fruits360/data.py
from __future__ import annotations
import tensorflow as tf
from . import config

# Keep your aliases
AUTO = tf.data.AUTOTUNE
dAUTO = tf.data.AUTOTUNE

# ---------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------
def _hw_from_config() -> tuple[int, int]:
    """Return a valid (H, W) no matter how config is set up."""
    sz = getattr(config, "IMAGE_SIZE", None)
    if isinstance(sz, (tuple, list)) and len(sz) >= 2:
        try:
            return int(sz[0]), int(sz[1])
        except Exception:
            pass
    return int(getattr(config, "IMG_HEIGHT", 100)), int(getattr(config, "IMG_WIDTH", 100))


def _interpolation():
    """Pick resize interpolation based on config.INTERPOLATION (default bilinear)."""
    interp_name = str(getattr(config, "INTERPOLATION", "bilinear")).lower()
    table = {
        "bilinear": tf.image.ResizeMethod.BILINEAR,
        "bicubic": tf.image.ResizeMethod.BICUBIC,
        "nearest": tf.image.ResizeMethod.NEAREST_NEIGHBOR,
        "lanczos3": tf.image.ResizeMethod.LANCZOS3,
        "lanczos5": tf.image.ResizeMethod.LANCZOS5,
        "area": tf.image.ResizeMethod.AREA,
        "mitchellcubic": tf.image.ResizeMethod.MITCHELLCUBIC,
    }
    return table.get(interp_name, tf.image.ResizeMethod.BILINEAR)


def _augment_pipeline():
    """Light augmentation pipeline – gated by config.USE_AUGMENTATION."""
    use_aug = getattr(config, "USE_AUGMENTATION", True)
    if not use_aug:
        return None
    layers = [
        tf.keras.layers.RandomFlip("horizontal"),
        tf.keras.layers.RandomRotation(0.05),
        tf.keras.layers.RandomZoom(0.1),
    ]
    return tf.keras.Sequential(layers, name="aug")


def _finalize_pipeline(ds, training: bool):
    """
    Final touches:
      - ensure dtype float32 in [0,1]
      - (optionally) apply light augmentation on train
      - cache & prefetch
    NOTE: image_dataset_from_directory already resized to (H, W).
    """
    aug = _augment_pipeline() if training else None

    def _to_float01(x, y):
        x = tf.image.convert_image_dtype(x, tf.float32)  # [0,1]
        return x, y

    ds = ds.map(_to_float01, num_parallel_calls=AUTO)

    if training and aug is not None:
        ds = ds.map(lambda x, y: (aug(x, training=True), y), num_parallel_calls=AUTO)

    ds = ds.cache()
    ds = ds.prefetch(AUTO)
    return ds


# ---------------------------------------------------------------------
# public loaders
# ---------------------------------------------------------------------
def load_train_val():
    """Create train/val datasets with consistent preprocessing."""
    H, W = _hw_from_config()
    batch = int(getattr(config, "BATCH", getattr(config, "BATCH_SIZE", 32)))
    seed = int(getattr(config, "SEED", 42))
    val_split = float(getattr(config, "VALIDATION_SPLIT", 0.15))

    train_raw = tf.keras.utils.image_dataset_from_directory(
        config.TRAIN_DIR,
        labels="inferred",
        label_mode="categorical",
        validation_split=val_split,
        subset="training",
        seed=seed,
        image_size=(H, W),  # Keras resizes here
        batch_size=batch,
        shuffle=True,
    )

    val_raw = tf.keras.utils.image_dataset_from_directory(
        config.TRAIN_DIR,
        labels="inferred",
        label_mode="categorical",
        validation_split=val_split,
        subset="validation",
        seed=seed,
        image_size=(H, W),
        batch_size=batch,
        shuffle=False,
    )

    class_names = train_raw.class_names
    if hasattr(config, "save_class_names"):
        try:
            config.save_class_names(class_names)
        except Exception:
            pass

    train_ds = _finalize_pipeline(train_raw, training=True)
    val_ds = _finalize_pipeline(val_raw, training=False)
    return train_ds, val_ds, class_names


def load_test():
    """Create test dataset with identical preprocessing."""
    if not config.TEST_DIR.exists():
        raise FileNotFoundError(f"Test dir not found: {config.TEST_DIR}")

    H, W = _hw_from_config()
    batch = int(getattr(config, "BATCH", getattr(config, "BATCH_SIZE", 32)))

    test_raw = tf.keras.utils.image_dataset_from_directory(
        config.TEST_DIR,
        labels="inferred",
        label_mode="categorical",
        image_size=(H, W),
        batch_size=batch,
        shuffle=False,
    )
    test_ds = _finalize_pipeline(test_raw, training=False)
    return test_ds