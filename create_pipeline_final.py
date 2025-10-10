import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import Rectangle
import matplotlib.lines as mlines

# Publication-quality settings
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman', 'DejaVu Serif']
plt.rcParams['font.size'] = 10
plt.rcParams['axes.linewidth'] = 1.0

# Set up figure
fig, ax = plt.subplots(1, 1, figsize=(16, 5))
ax.set_xlim(0, 16)
ax.set_ylim(0, 5)
ax.axis('off')

# Professional color scheme
color_primary = '#2C3E50'
color_light_bg = '#F5F5F5'

# ============ STAGE 1: INPUT ============
input_box = Rectangle((0.5, 1.8), 2.0, 1.4,
                      edgecolor=color_primary, facecolor=color_light_bg,
                      linewidth=1.5, zorder=2)
ax.add_patch(input_box)
ax.text(1.5, 2.75, 'Input', ha='center', va='center',
        fontsize=11, fontweight='bold', color=color_primary)
ax.text(1.5, 2.35, 'Blood Smear', ha='center', va='center', fontsize=9)
ax.text(1.5, 2.05, 'Images', ha='center', va='center', fontsize=9)
ax.text(1.5, 1.85, '(640×640)', ha='center', va='center', fontsize=7, style='italic', color='#666')

# Arrow: Input → Detection
ax.arrow(2.6, 2.5, 0.55, 0, head_width=0.15, head_length=0.08,
         fc=color_primary, ec=color_primary, linewidth=1.2, zorder=1)

# ============ STAGE 2: DETECTION ============
det_x = 3.3
det_y_base = 2.5
det_spacing = 0.6

# Detection stage label
ax.text(det_x + 0.9, 3.65, 'Detection Stage', ha='center', va='center',
        fontsize=10, fontweight='bold', color=color_primary)

# 3 YOLO boxes
yolo_labels = ['YOLO v10', 'YOLO v11', 'YOLO v12']
for i, label in enumerate(yolo_labels):
    y_offset = (1 - i) * det_spacing
    box = Rectangle((det_x, det_y_base + y_offset - 0.25), 1.8, 0.45,
                    edgecolor='#5DADE2', facecolor='#EBF5FB',
                    linewidth=1.2, zorder=2)
    ax.add_patch(box)
    ax.text(det_x + 0.9, det_y_base + y_offset, label, ha='center', va='center',
            fontsize=8, color=color_primary)

# Arrows from detection to shared crop (dashed, subtle)
for i in range(3):
    y_offset = (1 - i) * det_spacing
    ax.plot([det_x + 1.85, 6.2], [det_y_base + y_offset, 2.5],
            linestyle='--', linewidth=1.0, color='#95A5A6', alpha=0.6, zorder=1)
    # Small arrow head
    ax.plot(6.2, 2.5, marker='>', markersize=5, color='#95A5A6', alpha=0.6, zorder=1)

# ============ STAGE 3: SHARED GROUND TRUTH ============
shared_x = 6.3
shared_box = Rectangle((shared_x, 1.8), 2.2, 1.4,
                       edgecolor='#F39C12', facecolor='#FEF5E7',
                       linewidth=2.0, zorder=3, linestyle='--')
ax.add_patch(shared_box)

ax.text(shared_x + 1.1, 2.75, 'Ground Truth', ha='center', va='center',
        fontsize=10, fontweight='bold', color=color_primary)
ax.text(shared_x + 1.1, 2.4, 'Crop Generation', ha='center', va='center',
        fontsize=9, color=color_primary)
ax.text(shared_x + 1.1, 2.05, '(224×224)', ha='center', va='center',
        fontsize=7, style='italic', color='#666')

# "Shared" label (more subtle)
ax.text(shared_x + 1.1, 3.45, 'SHARED', ha='center', va='center',
        fontsize=7, fontweight='bold', color='#D68910',
        bbox=dict(boxstyle='round,pad=0.2', facecolor='white',
                 edgecolor='#D68910', linewidth=1.0))

# Arrow: Shared → Classification
ax.arrow(8.6, 2.5, 1.05, 0, head_width=0.15, head_length=0.08,
         fc=color_primary, ec=color_primary, linewidth=1.5, zorder=1)

# ============ STAGE 4: CLASSIFICATION ============
cls_x = 9.9
cls_y_top = 3.05
cls_y_bot = 2.05
cls_spacing = 1.4

# Classification stage label
ax.text(cls_x + 2.05, 3.65, 'Classification Stage', ha='center', va='center',
        fontsize=10, fontweight='bold', color=color_primary)

# 6 CNN models (2 rows × 3 columns, more compact)
cnn_models = [
    ['DenseNet121', 'EfficientNet-B0', 'EfficientNet-B1'],
    ['EfficientNet-B2', 'ResNet50', 'ResNet101']
]

for row_idx, row in enumerate(cnn_models):
    y_pos = cls_y_top if row_idx == 0 else cls_y_bot
    for col_idx, model in enumerate(row):
        x_pos = cls_x + col_idx * cls_spacing
        box = Rectangle((x_pos, y_pos - 0.18), 1.3, 0.36,
                       edgecolor='#9B59B6', facecolor='#F4ECF7',
                       linewidth=1.0, zorder=2)
        ax.add_patch(box)
        ax.text(x_pos + 0.65, y_pos, model, ha='center', va='center',
                fontsize=7, color=color_primary)

# Arrows from classification to output (dashed, subtle)
for row_idx in range(2):
    y_pos = cls_y_top if row_idx == 0 else cls_y_bot
    for col_idx in range(3):
        x_pos = cls_x + col_idx * cls_spacing + 1.3
        ax.plot([x_pos, 14.3], [y_pos, 2.5],
                linestyle='--', linewidth=0.8, color='#95A5A6', alpha=0.5, zorder=1)

# Output arrow head
ax.plot(14.3, 2.5, marker='>', markersize=5, color='#95A5A6', alpha=0.6, zorder=1)

# ============ STAGE 5: OUTPUT ============
output_x = 14.4
output_box = Rectangle((output_x, 1.8), 1.4, 1.4,
                       edgecolor='#27AE60', facecolor='#E8F8F5',
                       linewidth=1.5, zorder=2)
ax.add_patch(output_box)
ax.text(output_x + 0.7, 2.75, 'Output', ha='center', va='center',
        fontsize=11, fontweight='bold', color='#27AE60')
ax.text(output_x + 0.7, 2.35, 'Species/Stage', ha='center', va='center', fontsize=8)
ax.text(output_x + 0.7, 2.05, 'Classification', ha='center', va='center', fontsize=8)
ax.text(output_x + 0.7, 1.85, '+ Metrics', ha='center', va='center',
        fontsize=7, style='italic', color='#666')

# ============ STAGE LABELS (Bottom) ============
stage_labels = [
    (1.5, 1.5, '(a) Input'),
    (4.2, 1.5, '(b) Detection'),
    (7.4, 1.5, '(c) Shared Crops'),
    (11.55, 1.5, '(d) Classification'),
    (15.1, 1.5, '(e) Output')
]

for x, y, label in stage_labels:
    ax.text(x, y, label, ha='center', va='center',
            fontsize=8, style='italic', color='#555555')

# Save with publication quality
plt.tight_layout()
plt.savefig('luaran/figures/pipeline_architecture_horizontal.png',
            dpi=600, bbox_inches='tight', facecolor='white', edgecolor='none',
            pad_inches=0.1)
print("SUCCESS: Final publication-quality diagram created")
print("         Resolution: 600 DPI")
print("         Style: Clean, professional, Q1/Q2 journal ready")
plt.close()
