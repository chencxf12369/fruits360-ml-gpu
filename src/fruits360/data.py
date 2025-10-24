from __future__ import annotations
import tensorflow as tf
from . import config

# ---------------------------------------------------------------------
# Aliases (kept)
# ---------------------------------------------------------------------
AUTO = tf.data.AUTOTUNE
dAUTO = tf.data.AUTOTUNE

# ---------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------
def _hw_from_config() -> tuple[int, int]:
    """
    Return a guaranteed valid (H, W) tuple from config.
    Prefers config.IMAGE_SIZE (2-tuple), falls back to IMG_HEIGHT/IMG_WIDTH,
    and finally to (224,224) if anything goes wrong.
    """
    try:
        sz = getattr(config, "IMAGE_SIZE", None)
        if isinstance(sz, (tuple, list)) and len(sz) >= 2:
            h, w = int(sz[0]), int(sz[1])
            return h, w
        # Fallback to discrete values
        h = int(getattr(config, "IMG_HEIGHT", 224))
        w = int(getattr(config, "IMG_WIDTH", 224))
        return h, w
    except Exception:
        # Final safe default
        return (224, 224)


def _interpolation():
    """Pick resize interpolation based on config.INTERPOLATION (default bilinear)."""
    name = str(getattr(config, "INTERPOLATION",
                       getattr(config, "RESIZE_INTERP", "bilinear"))).lower()
    table = {
        "bilinear": tf.image.ResizeMethod.BILINEAR,
        "bicubic": tf.image.ResizeMethod.BICUBIC,
        "nearest": tf.image.ResizeMethod.NEAREST_NEIGHBOR,
        "lanczos3": tf.image.ResizeMethod.LANCZOS3,
        "lanczos5": tf.image.ResizeMethod.LANCZOS5,
        "area": tf.image.ResizeMethod.AREA,
        "mitchellcubic": tf.image.ResizeMethod.MITCHELLCUBIC,
    }
    return table.get(name, tf.image.ResizeMethod.BILINEAR)


def _resize_with_pad(img: tf.Tensor, tgt_h: int, tgt_w: int) -> tf.Tensor:
    """
    Aspect-preserving resize with padding to (tgt_h, tgt_w).
    Works for both single images (H, W, 3) and batches (B, H, W, 3).
    """
    img = tf.image.resize_with_pad(
        image=img,
        target_height=tgt_h,
        target_width=tgt_w,
        method=_interpolation(),
        antialias=True,
    )
    return img

def _augment_pipeline():
    """Light augmentation pipeline – only if AUG_IN_DATA=True."""
    if not getattr(config, "AUG_IN_DATA", False):
        return None
    layers = [
        tf.keras.layers.RandomFlip("horizontal"),
        tf.keras.layers.RandomRotation(0.05),
        tf.keras.layers.RandomZoom(0.1),
    ]
    return tf.keras.Sequential(layers, name="aug")


def _finalize_pipeline(ds: tf.data.Dataset, training: bool) -> tf.data.Dataset:
    H, W = _hw_from_config()
    use_pad = bool(getattr(config, "PAD_TO_SQUARE", False))
    #define aug in local scope
    aug = _augment_pipeline() if (training and getattr(config, "AUG_IN_DATA", False)) else None
    
    def _maybe_pad(x, y):
        # If enabled, letterbox to (H, W) without distortion
        if use_pad:
            x = _resize_with_pad(x, H, W)  # returns float32
        return x, y

    def _to_float01(x, y):
        # Convert to [0,1] regardless of whether we padded (which returns float32)
        x = tf.image.convert_image_dtype(x, tf.float32)
        return x, y

    # (Optional) aspect-ratio preserving resize + pad
    ds = ds.map(_maybe_pad, num_parallel_calls=AUTO)

    # Normalize
    ds = ds.map(_to_float01, num_parallel_calls=AUTO)
    
    ds = ds.cache()
    if training:
        ds = ds.shuffle(1000, reshuffle_each_iteration=True)

    if aug is not None:
        ds = ds.map(lambda x, y: (aug(x, training=True), y), num_parallel_calls=AUTO)

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
    mode = "keep-aspect+pad" if getattr(config, "PAD_TO_SQUARE", False) else "direct-resize"

    print(f"[data] Using image_size={(H, W)} ({mode})")

    train_raw = tf.keras.utils.image_dataset_from_directory(
        config.TRAIN_DIR,
        labels="inferred",
        label_mode="categorical",
        validation_split=val_split,
        subset="training",
        seed=seed,
        image_size=(H, W),  # Keras resizes here; we may re-letterbox later if PAD_TO_SQUARE=True
        batch_size=batch,
        shuffle=True,
    )

    val_raw = tf.keras.utils.image_dataset_from_directory(
        config.TRAIN_DIR,
        labels="inferred",
        label_mode="categorical" if False else "categorical",
        validation_split=val_split,
        subset="validation",
        seed=seed,
        image_size=(H, W),
        batch_size=batch,
        shuffle=False,
    )

    class_names = train_raw.class_names
    # Optional: persist class names if config provides a helper
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
    mode = "keep-aspect+pad" if getattr(config, "PAD_TO_SQUARE", False) else "direct-resize"

    print(f"[data] Using image_size={(H, W)} for TEST ({mode})")

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
