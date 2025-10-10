import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle
import matplotlib.lines as mlines

# Publication-quality settings
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman', 'DejaVu Serif']
plt.rcParams['font.size'] = 10

# Set up figure - wider for publication
fig, ax = plt.subplots(1, 1, figsize=(16, 5))
ax.set_xlim(0, 16)
ax.set_ylim(0, 5)
ax.axis('off')

# Professional color scheme (muted, academic)
color_bg = '#FFFFFF'
color_primary = '#2C3E50'      # Dark blue-grey
color_accent1 = '#5DADE2'      # Soft blue
color_accent2 = '#F39C12'      # Soft orange
color_accent3 = '#9B59B6'      # Soft purple
color_shared = '#F8E45C'       # Soft yellow (emphasis)

# Title removed per user request

# ============ STAGE 1: INPUT ============
input_box = Rectangle((0.5, 1.8), 2.0, 1.4,
                      edgecolor=color_primary, facecolor='#F5F5F5',
                      linewidth=1.5, zorder=2)
ax.add_patch(input_box)
ax.text(1.5, 2.8, 'Input', ha='center', va='center',
        fontsize=11, fontweight='bold', color=color_primary)
ax.text(1.5, 2.4, 'Blood Smear', ha='center', va='center', fontsize=9)
ax.text(1.5, 2.1, 'Images', ha='center', va='center', fontsize=9)
ax.text(1.5, 1.85, '(640×640)', ha='center', va='center', fontsize=7, style='italic', color='#666')

# Arrow: Input → Detection
ax.annotate('', xy=(3.2, 2.5), xytext=(2.6, 2.5),
            arrowprops=dict(arrowstyle='->', lw=1.5, color=color_primary))

# ============ STAGE 2: DETECTION ============
det_x = 3.3
det_y_base = 2.5
det_spacing = 0.6

# Detection stage label (lowered)
ax.text(det_x + 1.0, 3.7, 'Detection Stage', ha='center', va='center',
        fontsize=10, fontweight='bold', color=color_primary)

# 3 YOLO boxes (compact, stacked)
yolo_labels = ['YOLO v10', 'YOLO v11', 'YOLO v12']
for i, label in enumerate(yolo_labels):
    y_offset = (1 - i) * det_spacing
    box = Rectangle((det_x, det_y_base + y_offset - 0.25), 1.8, 0.45,
                    edgecolor=color_accent1, facecolor='#EBF5FB',
                    linewidth=1.2, zorder=2)
    ax.add_patch(box)
    ax.text(det_x + 0.9, det_y_base + y_offset, label, ha='center', va='center',
            fontsize=8, color=color_primary)

# Arrows from detection to shared crop
for i in range(3):
    y_offset = (1 - i) * det_spacing
    ax.annotate('', xy=(6.2, 2.5), xytext=(det_x + 1.85, det_y_base + y_offset),
                arrowprops=dict(arrowstyle='->', lw=1.0, color='#95A5A6',
                               linestyle='--', alpha=0.7))

# ============ STAGE 3: SHARED GROUND TRUTH (KEY FEATURE) ============
shared_x = 6.3
shared_box = Rectangle((shared_x, 1.8), 2.2, 1.4,
                       edgecolor='#F39C12', facecolor=color_shared,
                       linewidth=2.5, zorder=3, linestyle='--')
ax.add_patch(shared_box)

ax.text(shared_x + 1.1, 2.8, 'Ground Truth', ha='center', va='center',
        fontsize=10, fontweight='bold', color=color_primary)
ax.text(shared_x + 1.1, 2.45, 'Crop Generation', ha='center', va='center',
        fontsize=9, color=color_primary)
ax.text(shared_x + 1.1, 2.1, '(224×224)', ha='center', va='center',
        fontsize=7, style='italic', color='#666')

# "Shared" badge
ax.text(shared_x + 1.1, 3.5, 'SHARED', ha='center', va='center',
        fontsize=8, fontweight='bold', color='#D68910',
        bbox=dict(boxstyle='round,pad=0.25', facecolor='#FEF9E7',
                 edgecolor='#D68910', linewidth=1.2))

# Arrow: Shared → Classification
ax.annotate('', xy=(9.8, 2.5), xytext=(8.6, 2.5),
            arrowprops=dict(arrowstyle='->', lw=1.8, color=color_primary))

# ============ STAGE 4: CLASSIFICATION ============
cls_x = 9.9
cls_y_top = 3.1
cls_y_bot = 2.0
cls_spacing = 1.45

# Classification stage label (lowered)
ax.text(cls_x + 2.1, 3.7, 'Classification Stage', ha='center', va='center',
        fontsize=10, fontweight='bold', color=color_primary)

# 6 CNN models (2 rows × 3 columns)
cnn_models = [
    ['DenseNet121', 'EfficientNet-B0', 'EfficientNet-B1'],
    ['EfficientNet-B2', 'ResNet50', 'ResNet101']
]

for row_idx, row in enumerate(cnn_models):
    y_pos = cls_y_top if row_idx == 0 else cls_y_bot
    for col_idx, model in enumerate(row):
        x_pos = cls_x + col_idx * cls_spacing
        box = Rectangle((x_pos, y_pos - 0.2), 1.35, 0.4,
                       edgecolor=color_accent3, facecolor='#F4ECF7',
                       linewidth=1.0, zorder=2)
        ax.add_patch(box)
        ax.text(x_pos + 0.675, y_pos, model, ha='center', va='center',
                fontsize=7, color=color_primary)

# Arrows from classification to output
for row_idx in range(2):
    y_pos = cls_y_top if row_idx == 0 else cls_y_bot
    for col_idx in range(3):
        x_pos = cls_x + col_idx * cls_spacing + 1.35
        ax.annotate('', xy=(14.3, 2.5), xytext=(x_pos, y_pos),
                    arrowprops=dict(arrowstyle='->', lw=0.8, color='#95A5A6',
                                   linestyle='--', alpha=0.6))

# ============ STAGE 5: OUTPUT ============
output_x = 14.4
output_box = Rectangle((output_x, 1.8), 1.4, 1.4,
                       edgecolor='#27AE60', facecolor='#E8F8F5',
                       linewidth=1.5, zorder=2)
ax.add_patch(output_box)
ax.text(output_x + 0.7, 2.8, 'Output', ha='center', va='center',
        fontsize=11, fontweight='bold', color='#27AE60')
ax.text(output_x + 0.7, 2.4, 'Species/Stage', ha='center', va='center', fontsize=8)
ax.text(output_x + 0.7, 2.1, 'Classification', ha='center', va='center', fontsize=8)
ax.text(output_x + 0.7, 1.85, '+ Metrics', ha='center', va='center',
        fontsize=7, style='italic', color='#666')

# ============ STAGE LABELS (Bottom) ============
stage_labels = [
    (1.5, 1.4, '(a) Input'),
    (4.2, 1.4, '(b) Detection'),
    (7.4, 1.4, '(c) Shared Crops'),
    (11.6, 1.4, '(d) Classification'),
    (15.1, 1.4, '(e) Output')
]

for x, y, label in stage_labels:
    ax.text(x, y, label, ha='center', va='center',
            fontsize=8, style='italic', color='#555555')

# Benefits removed per user request

# Save with publication quality
plt.tight_layout()
plt.savefig('luaran/figures/pipeline_architecture_horizontal.png',
            dpi=600, bbox_inches='tight', facecolor='white', edgecolor='none')
print("SUCCESS: Publication-quality diagram created")
print("         File: luaran/figures/pipeline_architecture_horizontal.png")
print("         DPI: 600 (publication standard)")
print("         Style: Professional, minimalist, Q1 journal quality")
plt.close()
