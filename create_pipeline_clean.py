import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

# Publication settings
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman', 'DejaVu Serif']
plt.rcParams['font.size'] = 10

# Figure setup
fig, ax = plt.subplots(1, 1, figsize=(15, 5))
ax.set_xlim(0, 15)
ax.set_ylim(0, 5)
ax.axis('off')

color_primary = '#2C3E50'
color_light_bg = '#F5F5F5'

# ============ INPUT ============
input_box = Rectangle((0.5, 1.8), 1.8, 1.4, edgecolor=color_primary,
                      facecolor=color_light_bg, linewidth=1.5, zorder=2)
ax.add_patch(input_box)
ax.text(1.4, 2.7, 'Input', ha='center', va='center',
        fontsize=11, fontweight='bold', color=color_primary)
ax.text(1.4, 2.3, 'Blood Smear\nImages', ha='center', va='center', fontsize=9)
ax.text(1.4, 1.85, '(640×640)', ha='center', va='center', fontsize=7, style='italic', color='#666')

# Arrow: Input → Detection
ax.arrow(2.4, 2.5, 0.5, 0, head_width=0.15, head_length=0.08,
         fc=color_primary, ec=color_primary, linewidth=1.2)

# ============ DETECTION ============
det_x = 3.0
det_y = [3.1, 2.5, 1.9]

ax.text(det_x + 0.85, 3.65, 'Detection Stage', ha='center', va='center',
        fontsize=10, fontweight='bold', color=color_primary)

for i, (label, y) in enumerate(zip(['YOLO v10', 'YOLO v11', 'YOLO v12'], det_y)):
    box = Rectangle((det_x, y - 0.25), 1.7, 0.45, edgecolor='#5DADE2',
                    facecolor='#EBF5FB', linewidth=1.2, zorder=2)
    ax.add_patch(box)
    ax.text(det_x + 0.85, y, label, ha='center', va='center', fontsize=8)

# Dashed arrows: Detection → Shared
for y in det_y:
    ax.plot([det_x + 1.75, 5.5], [y, 2.5], '--', linewidth=1.0,
            color='#95A5A6', alpha=0.6, zorder=1)
ax.plot(5.5, 2.5, '>', markersize=5, color='#95A5A6', alpha=0.6)

# ============ SHARED CROPS ============
shared_x = 5.6
shared_box = Rectangle((shared_x, 1.8), 2.0, 1.4, edgecolor='#F39C12',
                       facecolor='#FEF5E7', linewidth=2.0, linestyle='--', zorder=3)
ax.add_patch(shared_box)
ax.text(shared_x + 1.0, 2.7, 'Ground Truth', ha='center', va='center',
        fontsize=10, fontweight='bold', color=color_primary)
ax.text(shared_x + 1.0, 2.35, 'Crop Generation', ha='center', va='center', fontsize=9)
ax.text(shared_x + 1.0, 2.0, '(224×224)', ha='center', va='center',
        fontsize=7, style='italic', color='#666')
ax.text(shared_x + 1.0, 3.45, 'SHARED', ha='center', va='center',
        fontsize=7, fontweight='bold', color='#D68910',
        bbox=dict(boxstyle='round,pad=0.2', facecolor='white',
                 edgecolor='#D68910', linewidth=1.0))

# Arrow: Shared → Classification
ax.arrow(7.7, 2.5, 0.7, 0, head_width=0.15, head_length=0.08,
         fc=color_primary, ec=color_primary, linewidth=1.5)

# ============ CLASSIFICATION ============
cls_x = 8.5
cls_y_top = 3.0
cls_y_bot = 2.0

ax.text(cls_x + 1.9, 3.65, 'Classification Stage', ha='center', va='center',
        fontsize=10, fontweight='bold', color=color_primary)

# 6 CNN models (2 rows × 3 columns)
cnn_models = [
    ['DenseNet121', 'EfficientNet-B0', 'EfficientNet-B1'],
    ['EfficientNet-B2', 'ResNet50', 'ResNet101']
]

for row_idx, row in enumerate(cnn_models):
    y_pos = cls_y_top if row_idx == 0 else cls_y_bot
    for col_idx, model in enumerate(row):
        x = cls_x + col_idx * 1.25
        box = Rectangle((x, y_pos - 0.18), 1.15, 0.36, edgecolor='#9B59B6',
                       facecolor='#F4ECF7', linewidth=1.0, zorder=2)
        ax.add_patch(box)
        ax.text(x + 0.575, y_pos, model, ha='center', va='center', fontsize=7)

# Single arrow: Classification → Output (no crossing lines!)
ax.arrow(12.3, 2.5, 0.7, 0, head_width=0.15, head_length=0.08,
         fc=color_primary, ec=color_primary, linewidth=1.5)

# ============ OUTPUT ============
output_x = 13.1
output_box = Rectangle((output_x, 1.8), 1.4, 1.4, edgecolor='#27AE60',
                       facecolor='#E8F8F5', linewidth=1.5, zorder=2)
ax.add_patch(output_box)
ax.text(output_x + 0.7, 2.7, 'Output', ha='center', va='center',
        fontsize=11, fontweight='bold', color='#27AE60')
ax.text(output_x + 0.7, 2.3, 'Species/Stage\nClassification', ha='center', va='center', fontsize=8)
ax.text(output_x + 0.7, 1.85, '+ Metrics', ha='center', va='center',
        fontsize=7, style='italic', color='#666')

# ============ STAGE LABELS ============
labels = [
    (1.4, 1.5, '(a) Input'),
    (3.85, 1.5, '(b) Detection'),
    (6.6, 1.5, '(c) Shared Crops'),
    (10.15, 1.5, '(d) Classification'),
    (13.8, 1.5, '(e) Output')
]

for x, y, label in labels:
    ax.text(x, y, label, ha='center', va='center',
            fontsize=8, style='italic', color='#555555')

# Save
plt.tight_layout()
plt.savefig('luaran/figures/pipeline_architecture_horizontal.png',
            dpi=600, bbox_inches='tight', facecolor='white', pad_inches=0.1)
print("SUCCESS: Clean diagram with improved spacing")
print("         - Closer: Shared Crops <-> Classification")
print("         - Clean: Single arrow Classification -> Output")
print("         - No crossing lines!")
plt.close()
