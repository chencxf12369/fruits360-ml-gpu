import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path

# Output path
out_path = Path("../artifacts")
out_path.mkdir(parents=True, exist_ok=True)
save_path = out_path / "mobilenetv2_architecture.png"

# Create diagram
fig, ax = plt.subplots(figsize=(11, 5))
ax.set_xlim(0, 12)
ax.set_ylim(0, 6)
ax.axis('off')

# Define main blocks
boxes = [
    (0.5, 2.5, 'Input\n(224×224×3)'),
    (2.5, 2.5, 'Preprocessing\n(Padding + Normalization)'),
    (4.5, 2.5, 'MobileNetV2\n(Pretrained Base)'),
    (6.5, 3.5, 'Global Avg Pooling'),
    (6.5, 1.5, 'Flatten + Dropout (0.2)'),
    (8.5, 2.5, 'Dense\n(Softmax, 131 Classes)'),
    (10.5, 2.5, 'Output:\nPredicted Class')
]

# Draw boxes
for x, y, label in boxes:
    ax.add_patch(
        mpatches.FancyBboxPatch(
            (x, y), 1.6, 1.0,
            boxstyle="round,pad=0.2",
            fc="#A7C7E7", ec="black", lw=1.5
        )
    )
    ax.text(x + 0.8, y + 0.5, label, ha='center', va='center', fontsize=9)

# Draw arrows
for i in range(len(boxes) - 1):
    x1 = boxes[i][0] + 1.6
    y1 = boxes[i][1] + 0.5
    x2 = boxes[i + 1][0]
    y2 = boxes[i + 1][1] + 0.5
    ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle='->', lw=1.5))

# Title
ax.text(6, 5.5,
        'CNN Transfer Learning Architecture – MobileNetV2 Backbone',
        ha='center', va='center', fontsize=12, fontweight='bold')

# Save and display
plt.tight_layout()
plt.savefig(save_path, dpi=200, bbox_inches='tight')
plt.show()

print(f"[saved] CNN architecture diagram -> {save_path}")
