# src/fruits360/data.py
from __future__ import annotations
import pathlib
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
        h = int(getattr(config, "IMG_HEIGHT", 224))
        w = int(getattr(config, "IMG_WIDTH", 224))
        return h, w
    except Exception:
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

def _list_classes(root: pathlib.Path) -> list[str]:
    return sorted([d.name for d in root.iterdir() if d.is_dir()])

def _enumerate_files(root: pathlib.Path, class_names: list[str]) -> tuple[list[str], list[int]]:
    paths, labels = [], []
    name_to_id = {n: i for i, n in enumerate(class_names)}
    for cls in class_names:
        cdir = root / cls
        for ext in ("*.jpg", "*.jpeg", "*.png"):
            for p in cdir.rglob(ext):
                paths.append(str(p))
                labels.append(name_to_id[cls])
    return paths, labels

@tf.function
def _pad_to_square_and_resize(img_uint8: tf.Tensor, target_hw: tuple[int, int]) -> tf.Tensor:
    """
    img_uint8: [H,W,3] uint8 → pad to square → resize to target_hw.
    Returns float32 in 0..255 (let the model's internal preprocess handle scaling).
    """
    h = tf.shape(img_uint8)[0]
    w = tf.shape(img_uint8)[1]
    dim = tf.maximum(h, w)
    dh  = dim - h
    dw  = dim - w
    pad_top    = dh // 2
    pad_bottom = dh - pad_top
    pad_left   = dw // 2
    pad_right  = dw - pad_left
    img_pad = tf.pad(
        img_uint8,
        paddings=[[pad_top, pad_bottom], [pad_left, pad_right], [0, 0]],
        mode="CONSTANT",
        constant_values=0
    )
    method = _interpolation()
    img_resized = tf.image.resize(img_pad, target_hw, method=method, antialias=True)
    return tf.cast(img_resized, tf.float32)  # 0..255

def _decode_and_preprocess(path: tf.Tensor, label: tf.Tensor,
                           target_hw: tuple[int, int]) -> tuple[tf.Tensor, tf.Tensor]:
    raw = tf.io.read_file(path)
    img = tf.image.decode_image(raw, channels=3, expand_animations=False)  # uint8
    img = _pad_to_square_and_resize(img, target_hw)                        # float32 0..255
    return img, label

def _augment_pipeline():
    """Light augmentation pipeline – only if AUG_IN_DATA=True."""
    if not getattr(config, "AUG_IN_DATA", False):
        return None
    return tf.keras.Sequential(
        [
            tf.keras.layers.RandomFlip("horizontal"),
            tf.keras.layers.RandomRotation(0.05),
            tf.keras.layers.RandomZoom(0.1),
        ],
        name="aug"
    )

def _finalize_pipeline(ds: tf.data.Dataset, training: bool) -> tf.data.Dataset:
    batch = int(getattr(config, "BATCH", getattr(config, "BATCH_SIZE", 32)))
    if training:
        aug = _augment_pipeline()
        if aug is not None:
            ds = ds.map(lambda x, y: (aug(x, training=True), y), num_parallel_calls=AUTO)
        ds = ds.shuffle(1000, reshuffle_each_iteration=True)
        ds = ds.batch(batch, drop_remainder=False).prefetch(AUTO)
    else:
        ds = ds.batch(batch, drop_remainder=False).prefetch(AUTO)
    return ds

# ---------------------------------------------------------------------
# public loaders — pad→resize for BOTH train & eval
# ---------------------------------------------------------------------
def load_train_val():
    """Create train/val datasets with PAD→RESIZE preprocessing (matching eval)."""
    H, W = _hw_from_config()
    seed = int(getattr(config, "SEED", 42))
    val_split = float(getattr(config, "VALIDATION_SPLIT", 0.15))

    # Informational print
    print(f"[data] Using pad→resize to {(H, W)} for TRAIN/VAL")

    train_root = pathlib.Path(config.TRAIN_DIR)
    if not train_root.exists():
        raise FileNotFoundError(f"Train dir not found: {train_root}")

    class_names = _list_classes(train_root)
    paths, labels = _enumerate_files(train_root, class_names)
    C = len(class_names)

    # Split deterministically
    # Deterministic permutation using stateless shuffle (seed must be 2-int)
    n = len(paths)
    idx = tf.range(n, dtype=tf.int32)

    try:
        # TensorFlow ≥ 2.17 supports stateless shuffle
        perm = tf.random.stateless_shuffle(idx, seed=(int(seed), 1337)).numpy()
    except AttributeError:
        # Fallback for TF ≤ 2.16 (e.g., macOS Metal)
        tf.random.set_seed(int(seed))
        perm = tf.random.shuffle(idx).numpy()

    cut = int(n * (1.0 - val_split))
    train_idx = perm[:cut].tolist()
    val_idx   = perm[cut:].tolist()
    paths_tf  = tf.convert_to_tensor(paths)
    labels_tf = tf.convert_to_tensor(labels)
    x_train = tf.gather(paths_tf, train_idx)
    y_train = tf.gather(labels_tf, train_idx)
    x_val   = tf.gather(paths_tf, val_idx)
    y_val   = tf.gather(labels_tf, val_idx)

    one_hot = lambda y: tf.one_hot(y, C, dtype=tf.float32)

    train_ds = tf.data.Dataset.from_tensor_slices((x_train, y_train))
    train_ds = train_ds.map(lambda p, y: _decode_and_preprocess(p, one_hot(y), (H, W)),
                            num_parallel_calls=AUTO)
    val_ds   = tf.data.Dataset.from_tensor_slices((x_val, y_val))
    val_ds   = val_ds.map(lambda p, y: _decode_and_preprocess(p, one_hot(y), (H, W)),
                          num_parallel_calls=AUTO)

    # Determinism option (applies BEFORE return)
    if not getattr(config, "DETERMINISTIC_DATA", True):
        opts = tf.data.Options()
        opts.experimental_deterministic = False
        train_ds = train_ds.with_options(opts)
        val_ds   = val_ds.with_options(opts)

    # Persist class names (optional helper in your config)
    if hasattr(config, "save_class_names"):
        try:
            config.save_class_names(class_names)
        except Exception:
            pass

    train_ds = _finalize_pipeline(train_ds, training=True)
    val_ds   = _finalize_pipeline(val_ds,   training=False)
    return train_ds, val_ds, class_names

def load_test():
    """Create test dataset with the SAME PAD→RESIZE preprocessing."""
    if not config.TEST_DIR.exists():
        raise FileNotFoundError(f"Test dir not found: {config.TEST_DIR}")

    H, W = _hw_from_config()
    print(f"[data] Using pad→resize to {(H, W)} for TEST")

    class_names = _list_classes(pathlib.Path(config.TEST_DIR))
    paths, labels = _enumerate_files(pathlib.Path(config.TEST_DIR), class_names)
    C = len(class_names)
    one_hot = lambda y: tf.one_hot(y, C, dtype=tf.float32)

    ds = tf.data.Dataset.from_tensor_slices((paths, labels))
    ds = ds.map(lambda p, y: _decode_and_preprocess(tf.convert_to_tensor(p),
                                                    one_hot(tf.convert_to_tensor(y)),
                                                    (H, W)),
                num_parallel_calls=AUTO)

    # Determinism option
    if not getattr(config, "DETERMINISTIC_DATA", True):
        opts = tf.data.Options()
        opts.experimental_deterministic = False
        ds = ds.with_options(opts)

    ds = _finalize_pipeline(ds, training=False)
    return ds
