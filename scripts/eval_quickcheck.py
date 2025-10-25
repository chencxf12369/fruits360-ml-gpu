"""
One-shot evaluation (auto-detect Test folder):
- Robust model load (handles Lambda: fruits360>mobilenetv2_preprocess)
- Auto-detects the Test/ directory if the provided path doesn't exist
- Outputs: overall acc, top-5 acc, per-class CSV, worst-20 PNG, confusion-matrix PNG, top-confused-pairs JSON, summary JSON
"""

import os, json, re
from pathlib import Path
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import pandas as pd
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"  # quiet TF

# =========================
# CONFIG (env overridable)
# =========================
MODEL_PATH = os.getenv("MODEL_PATH", "/Users/jxufe/Documents/ml-gpu/artifacts/fruits360_best.keras")
TEST_DIR   = os.getenv("TEST_DIR",  "/Users/jxufe/data/Fruit-Images-Dataset/Test")
ARTIFACTS  = Path(os.getenv("ARTIFACTS", "/Users/jxufe/Documents/ml-gpu/artifacts"))
IMAGE_SIZE = (224, 224)
BATCH_SIZE = 64
TOP_K      = int(os.getenv("TOP_K", "5"))
# =========================

ARTIFACTS.mkdir(parents=True, exist_ok=True)

# ---------- robust model load (register custom Lambda) ----------
from keras.saving import register_keras_serializable
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input as _mbv2_pp

@register_keras_serializable(package="fruits360", name="mobilenetv2_preprocess")
def mobilenetv2_preprocess(x):
    x = tf.cast(x, tf.float32)
    return _mbv2_pp(x)

CUSTOM_OBJECTS = {
    "fruits360>mobilenetv2_preprocess": mobilenetv2_preprocess,
    "mobilenetv2_preprocess": mobilenetv2_preprocess,
}

print(f"[load] model: {MODEL_PATH}")
model = tf.keras.models.load_model(MODEL_PATH, custom_objects=CUSTOM_OBJECTS, compile=False)
print("[load] model loaded OK")

# ---------- Test dir auto-detection ----------
def _try_pkg_config_dir():
    try:
        import fruits360.config as cfg
        p = Path(str(cfg.TEST_DIR))
        return p if p.exists() else None
    except Exception:
        return None

def _looks_like_test_dir(p: Path) -> bool:
    # Must contain subdirectories (class folders) and at least, say, 1000 images total
    if not p.is_dir():
        return False
    subdirs = [d for d in p.iterdir() if d.is_dir()]
    if len(subdirs) < 10:
        return False
    img_ext = re.compile(r"\.(jpe?g|png|bmp|gif|tiff)$", re.I)
    count = 0
    for d in subdirs:
        # count only a few files per class to keep it fast
        k = 0
        for f in d.iterdir():
            if f.is_file() and img_ext.search(f.suffix):
                count += 1
                k += 1
                if k >= 100:  # cap per-class counting
                    break
        if count >= 1000:
            return True
    return count >= 1000

def _auto_find_test_dir(search_roots):
    candidates = []
    for root in search_roots:
        root = Path(root).expanduser()
        if not root.exists():
            continue
        # find folders literally named 'Test' or case variants
        for p in root.rglob("*"):
            if p.name.lower() == "test" and _looks_like_test_dir(p):
                # collect with a rough score = total subdirs
                score = len([d for d in p.iterdir() if d.is_dir()])
                candidates.append((score, p))
    if candidates:
        candidates.sort(reverse=True)
        return candidates[0][1]
    return None

test_dir_path = Path(TEST_DIR)
if not test_dir_path.exists():
    print(f"[warn] TEST_DIR not found: {test_dir_path}")
    pkg_dir = _try_pkg_config_dir()
    if pkg_dir:
        test_dir_path = pkg_dir
        print(f"[auto] Using fruits360.config.TEST_DIR: {test_dir_path}")
    else:
        search_roots = [
            Path.cwd(),
            Path.cwd().parent,
            Path("/Users/jxufe/Documents/ml-gpu"),
            Path("/Users/jxufe/Documents"),
            Path.home(),
        ]
        guess = _auto_find_test_dir(search_roots)
        if guess:
            test_dir_path = guess
            print(f"[auto] Discovered Test dir: {test_dir_path}")
        else:
            raise FileNotFoundError(
                f"TEST_DIR not found. Set env TEST_DIR to your Test folder. Tried default and auto-search roots: {', '.join(map(str, search_roots))}"
            )

print(f"[data] Using test dataset: {test_dir_path}")

# ---------- Build dataset ----------
test_ds = tf.keras.utils.image_dataset_from_directory(
    test_dir_path,
    image_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    shuffle=False,
)
test_ds = test_ds.prefetch(tf.data.AUTOTUNE)
class_names = getattr(test_ds, "class_names", [str(i) for i in range(model.output_shape[-1])])

# ---------- Predict ----------
y_true = []
for _, y in test_ds:
    y_true.append(y.numpy())
y_true = np.concatenate(y_true, axis=0).astype(int)

print("[predict] running model.predict ...")
probs = model.predict(test_ds, verbose=1)
y_pred = np.argmax(probs, axis=1)

# ---------- Metrics ----------
overall_acc = float((y_pred == y_true).mean())
topk_idx = np.argpartition(-probs, kth=TOP_K-1, axis=1)[:, :TOP_K]
topk_hits = np.any(topk_idx == y_true[:, None], axis=1).astype(int)
topk_acc = float(topk_hits.mean())

print(f"\nOverall accuracy: {overall_acc:.4f}")
print(f"Top-{TOP_K} accuracy: {topk_acc:.4f}")

# Per-class
totals = np.bincount(y_true, minlength=len(class_names))
rights  = np.bincount(y_true[(y_pred == y_true)], minlength=len(class_names))
acc_per_class = np.divide(rights, totals, out=np.zeros_like(rights, dtype=float), where=totals>0)

df_acc = pd.DataFrame({
    "class_id": np.arange(len(class_names)),
    "class_name": class_names,
    "n_samples": totals,
    "n_correct": rights,
    "accuracy": acc_per_class
}).sort_values("accuracy")

csv_path = ARTIFACTS / "TEST_per_class_accuracy.csv"
df_acc.to_csv(csv_path, index=False)
print(f"[saved] {csv_path}")

# Worst-20 plot
k = min(20, len(class_names))
subset = df_acc.head(k)
plt.figure(figsize=(10, 7))
plt.barh(subset["class_name"], subset["accuracy"])
plt.gca().invert_yaxis()
plt.xlabel("Accuracy")
plt.title(f"Worst {k} Classes by Accuracy (Test Set)")
plt.tight_layout()
worst_png = ARTIFACTS / "TEST_per_class_accuracy_worst20.png"
plt.savefig(worst_png, dpi=220)
plt.close()
print(f"[plot]  {worst_png}")

# Confusion matrix
num_classes = len(class_names)
cm = tf.math.confusion_matrix(labels=y_true, predictions=y_pred, num_classes=num_classes).numpy()
row_sums = cm.sum(axis=1, keepdims=True)
cm_norm = np.divide(cm, row_sums, out=np.zeros_like(cm, dtype=float), where=row_sums>0)

plt.figure(figsize=(10, 8))
plt.imshow(cm_norm, interpolation="nearest", aspect="auto")
plt.title("Confusion Matrix (Row-Normalized)")
plt.xlabel("Predicted")
plt.ylabel("True")
plt.colorbar(fraction=0.046, pad=0.04)
tick_idx = np.linspace(0, num_classes-1, num=min(20, num_classes), dtype=int)
plt.xticks(tick_idx, [class_names[i] for i in tick_idx], rotation=90)
plt.yticks(tick_idx, [class_names[i] for i in tick_idx])
plt.tight_layout()
cm_png = ARTIFACTS / "TEST_confusion_matrix.png"
plt.savefig(cm_png, dpi=220)
plt.close()
print(f"[plot]  {cm_png}")

# Most confused pairs
cm_off = cm.copy()
np.fill_diagonal(cm_off, 0)
flat_sorted = np.argsort(cm_off.ravel())[::-1]
pairs = np.dstack(np.unravel_index(flat_sorted, cm_off.shape))[0]

top_pairs = []
for r, c in pairs:
    if cm_off[r, c] == 0: break
    top_pairs.append({
        "true_class_id": int(r),
        "true_class": class_names[r],
        "pred_class_id": int(c),
        "pred_class": class_names[c],
        "count": int(cm_off[r, c])
    })
    if len(top_pairs) >= 10: break

confused_json = ARTIFACTS / "TEST_top_confused_pairs.json"
with open(confused_json, "w") as f:
    json.dump(top_pairs, f, indent=2)
print(f"[saved] {confused_json}")

# Summary JSON
summary = {
    "overall_accuracy": overall_acc,
    "top_k": TOP_K,
    "top_k_accuracy": topk_acc,
    "num_classes": int(len(class_names)),
    "num_test_samples": int(len(y_true)),
    "test_dir": str(test_dir_path),
    "outputs": {
        "TEST_per_class_csv": str(csv_path),
        "TEST_worst20_png": str(worst_png),
        "TEST_confusion_matrix_png": str(cm_png),
        "TEST_top_confused_pairs_json": str(confused_json)
    }
}
with open(ARTIFACTS / "TEST_eval_summary.json", "w") as f:
    json.dump(summary, f, indent=2)
print(f"[saved] {ARTIFACTS / 'TEST_eval_summary.json'}\nDone.")
