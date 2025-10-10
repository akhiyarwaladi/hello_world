import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import matplotlib.lines as mlines

# Set up the figure with better size for horizontal layout
fig, ax = plt.subplots(1, 1, figsize=(20, 8))
ax.set_xlim(0, 20)
ax.set_ylim(0, 8)
ax.axis('off')

# Define colors (modern palette)
color_input = '#E8F4F8'      # Light blue
color_detection = '#FFE5B4'  # Peach
color_shared = '#FFD700'     # Gold (emphasis on shared component)
color_classification = '#E6E6FA'  # Lavender
color_output = '#90EE90'     # Light green

# Border colors
border_input = '#5DADE2'
border_detection = '#F39C12'
border_shared = '#F1C40F'
border_classification = '#9B59B6'
border_output = '#27AE60'

# Title
ax.text(10, 7.5, 'Option A: Shared Classification Pipeline Architecture',
        ha='center', va='top', fontsize=20, fontweight='bold', family='sans-serif')

# Subtitle
ax.text(10, 7.0, 'Efficient Multi-Model Framework with 70% Storage Reduction & 60% Training Time Reduction',
        ha='center', va='top', fontsize=11, style='italic', color='#555555')

# ============ STAGE 1: INPUT ============
input_box = FancyBboxPatch((0.5, 3), 2.5, 2,
                           boxstyle="round,pad=0.1",
                           edgecolor=border_input, facecolor=color_input,
                           linewidth=3, zorder=2)
ax.add_patch(input_box)
ax.text(1.75, 4.5, 'INPUT', ha='center', va='center',
        fontsize=13, fontweight='bold', color=border_input)
ax.text(1.75, 4.0, 'Blood Smear\nImages', ha='center', va='center', fontsize=11)
ax.text(1.75, 3.4, '640×640 px', ha='center', va='center', fontsize=9, style='italic', color='#666')

# Arrow 1: Input → Detection
arrow1 = FancyArrowPatch((3.1, 4.0), (4.4, 4.0),
                        arrowstyle='->', mutation_scale=30,
                        linewidth=2.5, color='#34495E', zorder=1)
ax.add_patch(arrow1)

# ============ STAGE 2: DETECTION (3 YOLO models stacked) ============
yolo_models = ['YOLOv10', 'YOLOv11', 'YOLOv12']
yolo_colors = ['#5DADE2', '#E74C3C', '#52BE80']
yolo_y_positions = [5.2, 4.0, 2.8]

for i, (model, color, y_pos) in enumerate(zip(yolo_models, yolo_colors, yolo_y_positions)):
    detection_box = FancyBboxPatch((4.5, y_pos - 0.5), 2.0, 1.0,
                                   boxstyle="round,pad=0.08",
                                   edgecolor=color, facecolor=color_detection,
                                   linewidth=2.5, zorder=2)
    ax.add_patch(detection_box)
    ax.text(5.5, y_pos, model, ha='center', va='center',
            fontsize=11, fontweight='bold', color=color)

# Label for Detection Stage
ax.text(5.5, 6.0, 'DETECTION STAGE', ha='center', va='center',
        fontsize=12, fontweight='bold', color=border_detection)
ax.text(5.5, 1.8, '(3 parallel models)', ha='center', va='center',
        fontsize=9, style='italic', color='#888')

# Arrows from YOLO models converging to shared crop
for y_pos in yolo_y_positions:
    arrow_yolo = FancyArrowPatch((6.6, y_pos), (7.9, 4.0),
                                arrowstyle='->', mutation_scale=25,
                                linewidth=2, color='#7F8C8D', zorder=1,
                                linestyle='--', alpha=0.7)
    ax.add_patch(arrow_yolo)

# ============ STAGE 3: SHARED GROUND TRUTH CROPS (KEY FEATURE) ============
shared_box = FancyBboxPatch((8.0, 3.0), 2.8, 2.0,
                            boxstyle="round,pad=0.12",
                            edgecolor=border_shared, facecolor=color_shared,
                            linewidth=4, zorder=3, linestyle='--')  # Dashed to emphasize "shared"
ax.add_patch(shared_box)

ax.text(9.4, 4.5, 'GROUND TRUTH', ha='center', va='center',
        fontsize=12, fontweight='bold', color='#7D6608')
ax.text(9.4, 4.0, 'Crop Generation', ha='center', va='center',
        fontsize=11, fontweight='bold', color='#7D6608')
ax.text(9.4, 3.5, '224×224 px', ha='center', va='center',
        fontsize=9, style='italic', color='#7D6608')

# Label to indicate "shared"
ax.text(9.4, 5.3, '*** SHARED COMPONENT ***', ha='center', va='center',
        fontsize=10, fontweight='bold', color='#D68910',
        bbox=dict(boxstyle='round,pad=0.3', facecolor='#FFF9E6', edgecolor='#D68910', linewidth=1.5))

# Arrow: Shared Crops → Classification
arrow_shared = FancyArrowPatch((10.9, 4.0), (12.4, 4.0),
                              arrowstyle='->', mutation_scale=30,
                              linewidth=3, color='#34495E', zorder=1)
ax.add_patch(arrow_shared)

# ============ STAGE 4: CLASSIFICATION (6 CNN models in 2 rows) ============
cnn_models_row1 = ['DenseNet121', 'EfficientNet-B0', 'EfficientNet-B1']
cnn_models_row2 = ['EfficientNet-B2', 'ResNet50', 'ResNet101']
cnn_colors = ['#9B59B6', '#8E44AD', '#7D3C98']

# Row 1 (top)
for i, model in enumerate(cnn_models_row1):
    x_pos = 12.5 + i * 1.8
    cnn_box = FancyBboxPatch((x_pos, 4.8), 1.6, 0.9,
                             boxstyle="round,pad=0.06",
                             edgecolor=border_classification, facecolor=color_classification,
                             linewidth=2, zorder=2)
    ax.add_patch(cnn_box)
    ax.text(x_pos + 0.8, 5.25, model, ha='center', va='center',
            fontsize=9, fontweight='bold', color='#5B2C6F')

# Row 2 (bottom)
for i, model in enumerate(cnn_models_row2):
    x_pos = 12.5 + i * 1.8
    cnn_box = FancyBboxPatch((x_pos, 3.2), 1.6, 0.9,
                             boxstyle="round,pad=0.06",
                             edgecolor=border_classification, facecolor=color_classification,
                             linewidth=2, zorder=2)
    ax.add_patch(cnn_box)
    ax.text(x_pos + 0.8, 3.65, model, ha='center', va='center',
            fontsize=9, fontweight='bold', color='#5B2C6F')

# Label for Classification Stage
ax.text(14.9, 6.2, 'CLASSIFICATION STAGE', ha='center', va='center',
        fontsize=12, fontweight='bold', color=border_classification)
ax.text(14.9, 2.5, '(6 CNN architectures)', ha='center', va='center',
        fontsize=9, style='italic', color='#888')

# Arrows from CNNs converging to output
for i in range(3):
    x_pos = 12.5 + i * 1.8 + 0.8
    # Top row
    arrow_cnn_top = FancyArrowPatch((x_pos + 0.8, 5.25), (17.4, 4.0),
                                   arrowstyle='->', mutation_scale=20,
                                   linewidth=1.5, color='#7F8C8D', zorder=1,
                                   linestyle='--', alpha=0.6)
    ax.add_patch(arrow_cnn_top)

    # Bottom row
    arrow_cnn_bot = FancyArrowPatch((x_pos + 0.8, 3.65), (17.4, 4.0),
                                   arrowstyle='->', mutation_scale=20,
                                   linewidth=1.5, color='#7F8C8D', zorder=1,
                                   linestyle='--', alpha=0.6)
    ax.add_patch(arrow_cnn_bot)

# ============ STAGE 5: OUTPUT ============
output_box = FancyBboxPatch((17.5, 3.0), 2.2, 2.0,
                            boxstyle="round,pad=0.1",
                            edgecolor=border_output, facecolor=color_output,
                            linewidth=3, zorder=2)
ax.add_patch(output_box)
ax.text(18.6, 4.6, 'OUTPUT', ha='center', va='center',
        fontsize=13, fontweight='bold', color=border_output)
ax.text(18.6, 4.0, 'Species/Stage\nClassification', ha='center', va='center', fontsize=10)
ax.text(18.6, 3.3, '+ Metrics', ha='center', va='center',
        fontsize=9, style='italic', color='#666')

# ============ LEGEND ============
legend_y = 1.2
legend_elements = [
    mpatches.Rectangle((0, 0), 1, 1, fc=color_input, ec=border_input, lw=2, label='Input Data'),
    mpatches.Rectangle((0, 0), 1, 1, fc=color_detection, ec=border_detection, lw=2, label='YOLO Detection'),
    mpatches.Rectangle((0, 0), 1, 1, fc=color_shared, ec=border_shared, lw=2, label='Shared Ground Truth'),
    mpatches.Rectangle((0, 0), 1, 1, fc=color_classification, ec=border_classification, lw=2, label='CNN Classification'),
    mpatches.Rectangle((0, 0), 1, 1, fc=color_output, ec=border_output, lw=2, label='Output')
]

legend = ax.legend(handles=legend_elements, loc='lower center', ncol=5,
                  fontsize=10, frameon=True, fancybox=True, shadow=True,
                  bbox_to_anchor=(0.5, -0.05))

# ============ KEY BENEFITS (Bottom annotation) ============
benefits_text = "[+] 70% Storage Reduction  |  [+] 60% Training Time Reduction  |  [+] Consistent Classification"
ax.text(10, 0.3, benefits_text, ha='center', va='center',
        fontsize=10, color='#27AE60', fontweight='bold',
        bbox=dict(boxstyle='round,pad=0.5', facecolor='#E8F8F5', edgecolor='#27AE60', linewidth=2))

# Save with high quality
plt.tight_layout()
plt.savefig('luaran/figures/pipeline_architecture_horizontal.png',
            dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
print("SUCCESS: Diagram saved to luaran/figures/pipeline_architecture_horizontal.png")
plt.close()

print("\nPipeline Diagram Created Successfully!")
print("   - Format: Horizontal flow (left to right)")
print("   - Resolution: 300 DPI (publication quality)")
print("   - Size: 20x8 inches")
print("   - Highlights: Shared Ground Truth component emphasized")
