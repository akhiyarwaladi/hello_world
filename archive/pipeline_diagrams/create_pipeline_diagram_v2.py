import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import matplotlib.lines as mlines

# Set up the figure with better size for horizontal layout
fig, ax = plt.subplots(1, 1, figsize=(20, 6))
ax.set_xlim(0, 20)
ax.set_ylim(0, 6)
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
ax.text(10, 5.7, 'Option A: Shared Classification Pipeline Architecture',
        ha='center', va='top', fontsize=20, fontweight='bold', family='sans-serif')

# Subtitle
ax.text(10, 5.3, 'Efficient Multi-Model Framework for Malaria Detection and Classification',
        ha='center', va='top', fontsize=11, style='italic', color='#555555')

# ============ STAGE 1: INPUT ============
input_box = FancyBboxPatch((0.5, 2.0), 2.5, 2.0,
                           boxstyle="round,pad=0.1",
                           edgecolor=border_input, facecolor=color_input,
                           linewidth=3, zorder=2)
ax.add_patch(input_box)
ax.text(1.75, 3.5, 'INPUT', ha='center', va='center',
        fontsize=13, fontweight='bold', color=border_input)
ax.text(1.75, 3.0, 'Blood Smear\nImages', ha='center', va='center', fontsize=11)
ax.text(1.75, 2.4, '640×640 px', ha='center', va='center', fontsize=9, style='italic', color='#666')

# Arrow 1: Input → Detection
arrow1 = FancyArrowPatch((3.1, 3.0), (4.4, 3.0),
                        arrowstyle='->', mutation_scale=30,
                        linewidth=2.5, color='#34495E', zorder=1)
ax.add_patch(arrow1)

# ============ STAGE 2: DETECTION (3 YOLO models stacked) ============
yolo_models = ['YOLOv10', 'YOLOv11', 'YOLOv12']
yolo_colors = ['#5DADE2', '#E74C3C', '#52BE80']
yolo_y_positions = [3.8, 3.0, 2.2]

for i, (model, color, y_pos) in enumerate(zip(yolo_models, yolo_colors, yolo_y_positions)):
    detection_box = FancyBboxPatch((4.5, y_pos - 0.35), 2.0, 0.7,
                                   boxstyle="round,pad=0.08",
                                   edgecolor=color, facecolor=color_detection,
                                   linewidth=2.5, zorder=2)
    ax.add_patch(detection_box)
    ax.text(5.5, y_pos, model, ha='center', va='center',
            fontsize=11, fontweight='bold', color=color)

# Label for Detection Stage
ax.text(5.5, 4.6, 'DETECTION STAGE', ha='center', va='center',
        fontsize=12, fontweight='bold', color=border_detection)
ax.text(5.5, 1.5, '(3 parallel models)', ha='center', va='center',
        fontsize=9, style='italic', color='#888')

# Arrows from YOLO models converging to shared crop
for y_pos in yolo_y_positions:
    arrow_yolo = FancyArrowPatch((6.6, y_pos), (7.9, 3.0),
                                arrowstyle='->', mutation_scale=25,
                                linewidth=2, color='#7F8C8D', zorder=1,
                                linestyle='--', alpha=0.7)
    ax.add_patch(arrow_yolo)

# ============ STAGE 3: SHARED GROUND TRUTH CROPS (KEY FEATURE) ============
shared_box = FancyBboxPatch((8.0, 2.0), 2.8, 2.0,
                            boxstyle="round,pad=0.12",
                            edgecolor=border_shared, facecolor=color_shared,
                            linewidth=4, zorder=3, linestyle='--')  # Dashed to emphasize "shared"
ax.add_patch(shared_box)

ax.text(9.4, 3.5, 'GROUND TRUTH', ha='center', va='center',
        fontsize=12, fontweight='bold', color='#7D6608')
ax.text(9.4, 3.0, 'Crop Generation', ha='center', va='center',
        fontsize=11, fontweight='bold', color='#7D6608')
ax.text(9.4, 2.5, '224×224 px', ha='center', va='center',
        fontsize=9, style='italic', color='#7D6608')

# Label to indicate "shared"
ax.text(9.4, 4.3, '*** SHARED COMPONENT ***', ha='center', va='center',
        fontsize=10, fontweight='bold', color='#D68910',
        bbox=dict(boxstyle='round,pad=0.3', facecolor='#FFF9E6', edgecolor='#D68910', linewidth=1.5))

# Arrow: Shared Crops → Classification
arrow_shared = FancyArrowPatch((10.9, 3.0), (12.4, 3.0),
                              arrowstyle='->', mutation_scale=30,
                              linewidth=3, color='#34495E', zorder=1)
ax.add_patch(arrow_shared)

# ============ STAGE 4: CLASSIFICATION (6 CNN models in 2 rows) ============
cnn_models_row1 = ['DenseNet121', 'EfficientNet-B0', 'EfficientNet-B1']
cnn_models_row2 = ['EfficientNet-B2', 'ResNet50', 'ResNet101']

# Row 1 (top)
for i, model in enumerate(cnn_models_row1):
    x_pos = 12.5 + i * 1.8
    cnn_box = FancyBboxPatch((x_pos, 3.5), 1.6, 0.7,
                             boxstyle="round,pad=0.06",
                             edgecolor=border_classification, facecolor=color_classification,
                             linewidth=2, zorder=2)
    ax.add_patch(cnn_box)
    ax.text(x_pos + 0.8, 3.85, model, ha='center', va='center',
            fontsize=9, fontweight='bold', color='#5B2C6F')

# Row 2 (bottom)
for i, model in enumerate(cnn_models_row2):
    x_pos = 12.5 + i * 1.8
    cnn_box = FancyBboxPatch((x_pos, 2.3), 1.6, 0.7,
                             boxstyle="round,pad=0.06",
                             edgecolor=border_classification, facecolor=color_classification,
                             linewidth=2, zorder=2)
    ax.add_patch(cnn_box)
    ax.text(x_pos + 0.8, 2.65, model, ha='center', va='center',
            fontsize=9, fontweight='bold', color='#5B2C6F')

# Label for Classification Stage
ax.text(14.9, 4.6, 'CLASSIFICATION STAGE', ha='center', va='center',
        fontsize=12, fontweight='bold', color=border_classification)
ax.text(14.9, 1.5, '(6 CNN architectures)', ha='center', va='center',
        fontsize=9, style='italic', color='#888')

# Arrows from CNNs converging to output
for i in range(3):
    x_pos = 12.5 + i * 1.8 + 0.8
    # Top row
    arrow_cnn_top = FancyArrowPatch((x_pos + 0.8, 3.85), (17.4, 3.0),
                                   arrowstyle='->', mutation_scale=20,
                                   linewidth=1.5, color='#7F8C8D', zorder=1,
                                   linestyle='--', alpha=0.6)
    ax.add_patch(arrow_cnn_top)

    # Bottom row
    arrow_cnn_bot = FancyArrowPatch((x_pos + 0.8, 2.65), (17.4, 3.0),
                                   arrowstyle='->', mutation_scale=20,
                                   linewidth=1.5, color='#7F8C8D', zorder=1,
                                   linestyle='--', alpha=0.6)
    ax.add_patch(arrow_cnn_bot)

# ============ STAGE 5: OUTPUT (SEJAJAR DENGAN CLASSIFICATION) ============
# Centered vertically with classification boxes (between row 1 and row 2)
output_y_center = (3.85 + 2.65) / 2  # Average of top and bottom CNN box centers
output_box = FancyBboxPatch((17.5, output_y_center - 1.0), 2.2, 2.0,
                            boxstyle="round,pad=0.1",
                            edgecolor=border_output, facecolor=color_output,
                            linewidth=3, zorder=2)
ax.add_patch(output_box)
ax.text(18.6, output_y_center + 0.6, 'OUTPUT', ha='center', va='center',
        fontsize=13, fontweight='bold', color=border_output)
ax.text(18.6, output_y_center, 'Species/Stage\nClassification', ha='center', va='center', fontsize=10)
ax.text(18.6, output_y_center - 0.7, '+ Performance Metrics', ha='center', va='center',
        fontsize=9, style='italic', color='#666')

# Save with high quality
plt.tight_layout()
plt.savefig('luaran/figures/pipeline_architecture_horizontal.png',
            dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
print("SUCCESS: Diagram saved to luaran/figures/pipeline_architecture_horizontal.png")
plt.close()

print("\nPipeline Diagram Created Successfully!")
print("   - Format: Horizontal flow (left to right)")
print("   - Resolution: 300 DPI (publication quality)")
print("   - Cleaned: No legend, no benefits text")
print("   - Layout: OUTPUT box aligned with classification stage")
