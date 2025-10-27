import pandas as pd
import os

base = r"C:\Users\MyPC PRO\Documents\hello_world\luaran\templates\tables"

# Table 2: Detection Performance
print("="*80)
print("TABLE 2: DETECTION PERFORMANCE - VERIFICATION")
print("="*80)
df2 = pd.read_excel(os.path.join(base, "Table2_Detection_Performance.xlsx"))
print("\n[FULL TABLE 2]")
print(df2.to_string(index=False))

print("\n[PAPER CLAIM CHECK]: YOLO11 on IML should be 94.99% mAP@50")
iml_rows = df2[df2['Dataset'].str.contains('IML', case=False, na=False)]
yolo11_iml = iml_rows[iml_rows['Model'] == 'YOLOv11']
if not yolo11_iml.empty:
    actual = yolo11_iml['mAP@50'].values[0]
    print(f"ACTUAL: {actual}")
    print(f"MATCH: {'✅ YES' if abs(actual - 0.9499) < 0.0001 else '❌ NO'}")

print("\n[PAPER CLAIM CHECK]: YOLO11 on MD_2019 should be 72.91% mAP@50")
md_rows = df2[df2['Dataset'].str.contains('MD', case=False, na=False)]
yolo11_md = md_rows[md_rows['Model'] == 'YOLOv11']
if not yolo11_md.empty:
    actual = yolo11_md['mAP@50'].values[0]
    print(f"ACTUAL: {actual}")
    print(f"MATCH: {'✅ YES' if abs(actual - 0.7291) < 0.0001 else '❌ NO'}")

# Table 3: IML Classification
print("\n" + "="*80)
print("TABLE 3: IML CLASSIFICATION - VERIFICATION")
print("="*80)
df3 = pd.read_excel(os.path.join(base, "Table3_IML_Classification.xlsx"))
print("\n[FULL TABLE 3]")
print(df3.to_string(index=False))

print("\n[PAPER CLAIM CHECK]: EfficientNet-B1 should be 91.51% accuracy")
effb1 = df3[df3['Model'].str.contains('EfficientNet-B1', case=False, na=False)]
if not effb1.empty:
    actual = effb1['Overall Accuracy'].values[0]
    print(f"ACTUAL: {actual}")
    print(f"MATCH: {'✅ YES' if abs(actual - 0.9151) < 0.0001 else '❌ NO'}")

# Table 4: Species Classification
print("\n" + "="*80)
print("TABLE 4: SPECIES CLASSIFICATION - VERIFICATION")
print("="*80)
df4 = pd.read_excel(os.path.join(base, "Table4_Species_Classification.xlsx"))
print("\n[FULL TABLE 4]")
print(df4.to_string(index=False))

print("\n[PAPER CLAIM CHECK]: EfficientNet-B1 should be 98.28% accuracy")
effb1_species = df4[df4['Model'].str.contains('EfficientNet-B1', case=False, na=False)]
if not effb1_species.empty:
    actual = effb1_species['Overall Accuracy'].values[0]
    print(f"ACTUAL: {actual}")
    print(f"MATCH: {'✅ YES' if abs(actual - 0.9828) < 0.0001 else '❌ NO'}")

print("\n✅ VERIFICATION SCRIPT COMPLETE")
