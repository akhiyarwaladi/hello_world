"""
Create Table 1 with horizontal stacked format (YOLO variants as columns)
"""
import pandas as pd

# Define data in horizontal format
data = {
    'Dataset': [
        'IML Lifecycle',
        'MP-IDB Species',
        'MP-IDB Stages',
        'MD_2019 Stages'
    ],
    # mAP@50
    'mAP@50: YOLO10': [93.81, 92.44, 93.78, 70.84],
    'mAP@50: YOLO11': [94.99, 92.57, 94.48, 72.91],
    'mAP@50: YOLO12': [94.40, 92.72, 96.27, 71.12],
    # mAP@50-95
    'mAP@50-95: YOLO10': [77.71, 60.12, 44.48, 57.05],
    'mAP@50-95: YOLO11': [77.76, 62.17, 60.34, 57.71],
    'mAP@50-95: YOLO12': [78.21, 62.25, 61.53, 56.93],
    # Precision
    'Precision: YOLO10': [90.22, 85.67, 91.57, 67.89],
    'Precision: YOLO11': [91.91, 86.52, 93.15, 68.58],
    'Precision: YOLO12': [92.20, 88.99, 92.91, 65.92],
    # Recall
    'Recall: YOLO10': [92.22, 90.73, 93.12, 71.05],
    'Recall: YOLO11': [91.11, 91.88, 88.36, 75.70],
    'Recall: YOLO12': [86.67, 89.28, 92.59, 75.18]
}

df = pd.DataFrame(data)

# Save to Excel
output_path = 'luaran/templates/tables/Table1_Detection_Performance.xlsx'
df.to_excel(output_path, index=False, sheet_name='Detection Results')

print("✅ Table 1 created with horizontal stacked format!")
print(f"📊 Shape: {df.shape[0]} rows × {df.shape[1]} columns")
print("\nPreview:")
print(df.to_string(index=False))
