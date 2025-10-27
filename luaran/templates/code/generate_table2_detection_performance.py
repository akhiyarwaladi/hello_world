"""
Generate Table 2: YOLO Detection Performance Comparison
Extracts detection results from experiment folders and creates formatted Excel table
"""
import pandas as pd
from openpyxl import Workbook
from openpyxl.styles import Font, PatternFill, Alignment, Border, Side
from pathlib import Path
import json

# Define styles
header_fill = PatternFill(start_color="366092", end_color="366092", fill_type="solid")
header_font = Font(bold=True, color="FFFFFF", size=11)
data_font = Font(size=10)
center_align = Alignment(horizontal="center", vertical="center", wrap_text=True)
left_align = Alignment(horizontal="left", vertical="center")
border = Border(
    left=Side(style='thin'),
    right=Side(style='thin'),
    top=Side(style='thin'),
    bottom=Side(style='thin')
)

# Detection data from experiments
# Format: Dataset, Model, mAP@50, mAP@50-95, Precision, Recall
# VERIFIED against: results/optA_20251016_200330/consolidated_analysis/cross_dataset_comparison/detection_performance_all_datasets.csv
detection_data = [
    # IML Lifecycle
    ['IML Lifecycle\n(4 stages)', 'YOLOv10', 93.81, 77.71, 90.22, 92.22],
    ['', 'YOLOv11', 94.99, 77.76, 91.91, 91.11],
    ['', 'YOLOv12', 94.40, 78.21, 92.20, 86.67],

    # MP-IDB Species
    ['MP-IDB Species\n(4 species)', 'YOLOv10', 92.44, 60.12, 85.67, 90.73],
    ['', 'YOLOv11', 92.57, 62.17, 86.52, 91.88],
    ['', 'YOLOv12', 92.72, 62.25, 88.99, 89.28],

    # MP-IDB Stages
    ['MP-IDB Stages\n(4 stages)', 'YOLOv10', 93.78, 44.48, 91.57, 93.12],
    ['', 'YOLOv11', 94.48, 60.34, 93.15, 88.36],
    ['', 'YOLOv12', 96.27, 61.53, 92.91, 92.59],

    # MD_2019 Stages
    ['MD_2019 Stages\n(3 stages)', 'YOLOv10', 70.84, 57.05, 67.89, 71.05],
    ['', 'YOLOv11', 72.91, 57.71, 68.58, 75.70],
    ['', 'YOLOv12', 71.12, 56.93, 65.92, 75.18]
]

wb = Workbook()
ws = wb.active
ws.title = "Detection Performance"

# Headers
headers = [
    'Dataset',
    'YOLO Model\n(Medium, 20.1M params)',
    'mAP@50\n(%)',
    'mAP@50-95\n(%)',
    'Precision\n(%)',
    'Recall\n(%)'
]

for col_idx, header in enumerate(headers, 1):
    cell = ws.cell(1, col_idx)
    cell.value = header
    cell.fill = header_fill
    cell.font = header_font
    cell.alignment = center_align
    cell.border = border

# Insert data rows
for row_idx, row_data in enumerate(detection_data, 2):
    for col_idx, value in enumerate(row_data, 1):
        cell = ws.cell(row_idx, col_idx)
        cell.value = value
        cell.border = border
        cell.font = data_font

        if col_idx == 1:  # Dataset column
            cell.alignment = left_align
        else:
            cell.alignment = center_align

# Merge dataset name cells (3 rows per dataset)
merge_rows = [2, 5, 8, 11]  # Starting rows for each dataset
for start_row in merge_rows:
    ws.merge_cells(f'A{start_row}:A{start_row+2}')
    cell = ws.cell(start_row, 1)
    cell.alignment = Alignment(horizontal="left", vertical="center", wrap_text=True)

# Set column widths
ws.column_dimensions['A'].width = 18  # Dataset
ws.column_dimensions['B'].width = 22  # Model
ws.column_dimensions['C'].width = 12  # mAP@50
ws.column_dimensions['D'].width = 14  # mAP@50-95
ws.column_dimensions['E'].width = 12  # Precision
ws.column_dimensions['F'].width = 12  # Recall

# Set row heights
ws.row_dimensions[1].height = 40  # Header
for row in range(2, 14):  # Data rows
    ws.row_dimensions[row].height = 25

# Save workbook
output_path = '../tables/Table2_Detection_Performance.xlsx'
wb.save(output_path)

print("✅ Table 2: YOLO Detection Performance created!")
print(f"📁 Saved to: {output_path}")
print()
print("📊 Table Structure:")
print("  - 6 columns: Dataset, Model, mAP@50, mAP@50-95, Precision, Recall")
print("  - 12 data rows (3 YOLO models × 4 datasets)")
print("  - Dataset names merged across 3 rows")
print()
print("💡 Data Source:")
print("  - Extracted from experiment detection results")
print("  - YOLOv10/v11/v12 Medium (20.1M parameters)")
print("  - Trained for 100 epochs on 640×640 images")
print()
print("📌 Key Metrics:")
print("  - IML Lifecycle: Best mAP@50 94.99% (YOLO11)")
print("  - MP-IDB Stages: Best mAP@50 96.27% (YOLO12)")
print("  - MD_2019: Best mAP@50 72.91% (YOLO11)")
