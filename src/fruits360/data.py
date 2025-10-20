# src/fruits360/data.py
from __future__ import annotations
import tensorflow as tf
from . import config

# ---------------------------------------------------------------------
# Aliases and constants
# ---------------------------------------------------------------------
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


# ---------------------------------------------------------------------
# square padding and resize
# ---------------------------------------------------------------------
def _pad_to_square_and_resize(img_uint8: tf.Tensor,
                              target_hw: tuple[int, int],
                              method=tf.image.ResizeMethod.BILINEAR) -> tf.Tensor:
    """
    Pads an unbatched [H,W,3] uint8 image to square shape and resizes to target_hw.
    Leaves values in [0,255]; later normalization brings it to [0,1].
    """
    shape = tf.shape(img_uint8)
    h, w = shape[0], shape[1]
    dim = tf.maximum(h, w)
    dh, dw = dim - h, dim - w
    pad_top, pad_bottom = dh // 2, dh - dh // 2
    pad_left, pad_right = dw // 2, dw - dw // 2
    img_pad = tf.pad(img_uint8,
                     [[pad_top, pad_bottom], [pad_left, pad_right], [0, 0]],
                     mode="CONSTANT", constant_values=0)
    img_resized = tf.image.resize(img_pad, target_hw, method=method, antialias=True)
    return tf.cast(img_resized, tf.float32)


# ---------------------------------------------------------------------
# dataset post-processing
# ---------------------------------------------------------------------
def _finalize_pipeline(ds, training: bool):
    """
    Final touches:
      - Pad & resize each image safely to square.
      - Convert to float32 [0,1].
      - Optionally apply light augmentation on training.
      - Cache & prefetch.
    """
    H, W = _hw_from_config()
    interp = _interpolation()
    aug = _augment_pipeline() if training else None

    def _prep_one(x, y):
        # Ensure shape rank-3 (unbatched image)
        x = _pad_to_square_and_resize(tf.cast(x, tf.float32), (H, W), method=interp)
        x = tf.image.convert_image_dtype(x, tf.float32)  # [0,1]
        return (aug(x, training=True), y) if (training and aug) else (x, y)

    ds = ds.map(_prep_one, num_parallel_calls=AUTO)
    ds = ds.cache()
    ds = ds.prefetch(AUTO)
    return ds


# ---------------------------------------------------------------------
# public loaders
# ---------------------------------------------------------------------
def load_train_val():
    """Create train/val datasets with consistent padding, resize, and augmentation."""
    H, W = _hw_from_config()
    batch = int(getattr(config, "BATCH", getattr(config, "BATCH_SIZE", 32)))
    seed = int(getattr(config, "SEED", 42))
    val_split = float(getattr(config, "VALIDATION_SPLIT", 0.15))

    # Use unbatched mode to allow per-image pad/resize
    train_raw = tf.keras.utils.image_dataset_from_directory(
        config.TRAIN_DIR,
        labels="inferred",
        label_mode="categorical",
        validation_split=val_split,
        subset="training",
        seed=seed,
        image_size=None,       # keep raw sizes
        batch_size=None,       # unbatched
        shuffle=True,
    )

    val_raw = tf.keras.utils.image_dataset_from_directory(
        config.TRAIN_DIR,
        labels="inferred",
        label_mode="categorical",
        validation_split=val_split,
        subset="validation",
        seed=seed,
        image_size=None,
        batch_size=None,
        shuffle=False,
    )

    class_names = train_raw.class_names
    if hasattr(config, "save_class_names"):
        try:
            config.save_class_names(class_names)
        except Exception:
            pass

    # Apply final preprocessing and batching
    train_ds = _finalize_pipeline(train_raw, training=True).batch(batch, drop_remainder=False)
    val_ds = _finalize_pipeline(val_raw, training=False).batch(batch, drop_remainder=False)
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
        image_size=None,
        batch_size=None,
        shuffle=False,
    )

    test_ds = _finalize_pipeline(test_raw, training=False).batch(batch, drop_remainder=False)
    return test_ds
