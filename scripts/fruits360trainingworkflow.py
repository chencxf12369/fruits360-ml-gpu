import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path

fig, ax = plt.subplots(figsize=(12, 6))

# Define boxes
boxes = [
    (0, 0, 2.5, 1, 'Dataset Preparation\n(Fruit-Images-Dataset)'),
    (3, 0, 2.5, 1, 'Data Loading\n(tf.keras.utils.image_dataset_from_directory)'),
    (6, 0, 2.5, 1, 'Preprocessing &\nAugmentation\n(resize, flip, rotation)'),
    (9, 0, 2.5, 1, 'Model Construction\n(MobileNetV2 Transfer Learning)'),
    (12, 0, 2.5, 1, 'Training Loop\n(Forward + Backpropagation)'),
    (15, 0, 2.5, 1, 'Evaluation &\nMetrics (Accuracy, Loss)'),
    (18, 0, 2.5, 1, 'Inference &\nVisualization\n(Confusion Matrix, Plots)')
]

# Draw boxes
for (x, y, w, h, text) in boxes:
    ax.add_patch(mpatches.FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.1", fc="#d0e1f9", ec="#004c91", lw=1.8))
    ax.text(x + w/2, y + h/2, text, ha='center', va='center', fontsize=10, color='#002b5c')

# Draw arrows
for i in range(len(boxes) - 1):
    x_start = boxes[i][0] + boxes[i][2]
    x_end = boxes[i+1][0]
    ax.arrow(x_start, 0.5, x_end - x_start - 0.2, 0, head_width=0.15, head_length=0.2, fc='black', ec='black')

# Style
ax.set_xlim(-0.5, 21)
ax.set_ylim(-0.5, 1.8)
ax.axis('off')
ax.set_title('Fruits-360 Project: Overall Data and Training Workflow', fontsize=14, fontweight='bold', color='#002b5c')

# Save to artifacts folder
save_path = Path(__file__).resolve().parents[1] / 'artifacts' / 'fruits360trainingworkflow.png'
save_path.parent.mkdir(parents=True, exist_ok=True)
plt.tight_layout()
plt.savefig(save_path, dpi=200)
plt.close()
print(f"[saved] Workflow diagram saved to {save_path}")

