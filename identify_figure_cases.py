import pandas as pd
import os
import json

base_path = r"C:\Users\MyPC PRO\Documents\hello_world\results\optA_20251207_233941\experiments"

# Output dictionary for all figure cases
figure_cases = {
    "Figure 5 - Detection Error Cases": {},
    "Figure 6 - Classification Error Cases": {}
}

print("=" * 100)
print("COMPREHENSIVE FIGURE CASE IDENTIFICATION")
print("=" * 100)

# =====================================================================
# FIGURE 5: DETECTION ERROR CASES
# =====================================================================

datasets = {
    'iml_lifecycle': 'experiment_iml_lifecycle',
    'md_2019_stages': 'experiment_md_2019_stages',
    'mp_idb_species': 'experiment_mp_idb_species',
    'mp_idb_stages': 'experiment_mp_idb_stages'
}

print("\n" + "=" * 100)
print("FIGURE 5: DETECTION ERROR CASES (YOLO11)")
print("=" * 100)

for ds_name, exp_folder in datasets.items():
    csv_path = os.path.join(base_path, exp_folder, 'visualizations', 'pred_detection_yolo11', 'detection_metadata.csv')

    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path)
        df = df.drop_duplicates(subset='image_name')

        # Figure 5(a) IML: FP only case
        if ds_name == 'iml_lifecycle':
            print("\n[Figure 5a] IML Lifecycle - False Positive Only Case")
            print("-" * 100)
            fp_only = df[(df['n_false_positives'] > 0) & (df['n_false_negatives'] == 0)].sort_values('n_false_positives', ascending=False)
            if len(fp_only) > 0:
                best = fp_only.iloc[0]
                figure_cases["Figure 5 - Detection Error Cases"]["5a_IML_FP_only"] = {
                    "image_name": best['image_name'],
                    "image_file": best['image_file'],
                    "n_gt_boxes": int(best['n_gt_boxes']),
                    "n_pred_boxes": int(best['n_pred_boxes']),
                    "n_false_positives": int(best['n_false_positives']),
                    "n_false_negatives": int(best['n_false_negatives']),
                    "avg_confidence": float(best['avg_confidence']),
                    "description": "False positive only (overdetection)"
                }
                print(f"✓ RECOMMENDED: {best['image_name']}")
                print(f"  GT boxes: {best['n_gt_boxes']}, Pred boxes: {best['n_pred_boxes']}")
                print(f"  FP: {best['n_false_positives']}, FN: {best['n_false_negatives']}, Confidence: {best['avg_confidence']:.3f}")
                print(f"  File: {best['image_file']}")

        # Figure 5(b) IML: FN only case
        if ds_name == 'iml_lifecycle':
            print("\n[Figure 5b] IML Lifecycle - False Negative Only Case")
            print("-" * 100)
            fn_only = df[(df['n_false_negatives'] > 0) & (df['n_false_positives'] == 0)].sort_values('n_false_negatives', ascending=False)
            if len(fn_only) > 0:
                best = fn_only.iloc[0]
                figure_cases["Figure 5 - Detection Error Cases"]["5b_IML_FN_only"] = {
                    "image_name": best['image_name'],
                    "image_file": best['image_file'],
                    "n_gt_boxes": int(best['n_gt_boxes']),
                    "n_pred_boxes": int(best['n_pred_boxes']),
                    "n_false_positives": int(best['n_false_positives']),
                    "n_false_negatives": int(best['n_false_negatives']),
                    "avg_confidence": float(best['avg_confidence']),
                    "description": "False negative only (missed detection)"
                }
                print(f"✓ RECOMMENDED: {best['image_name']}")
                print(f"  GT boxes: {best['n_gt_boxes']}, Pred boxes: {best['n_pred_boxes']}")
                print(f"  FP: {best['n_false_positives']}, FN: {best['n_false_negatives']}, Confidence: {best['avg_confidence']:.3f}")
                print(f"  File: {best['image_file']}")

        # Figure 5(c) MP-IDB Stages: overdetection with high FPs
        if ds_name == 'mp_idb_stages':
            print("\n[Figure 5c] MP-IDB Stages - Overdetection Case (High FP)")
            print("-" * 100)
            high_fp = df[df['n_false_positives'] >= 5].sort_values('n_false_positives', ascending=False)
            if len(high_fp) > 0:
                best = high_fp.iloc[0]
                figure_cases["Figure 5 - Detection Error Cases"]["5c_MPIDB_Stages_Overdetection"] = {
                    "image_name": best['image_name'],
                    "image_file": best['image_file'],
                    "n_gt_boxes": int(best['n_gt_boxes']),
                    "n_pred_boxes": int(best['n_pred_boxes']),
                    "n_false_positives": int(best['n_false_positives']),
                    "n_false_negatives": int(best['n_false_negatives']),
                    "avg_confidence": float(best['avg_confidence']),
                    "description": "Overdetection with high false positives"
                }
                print(f"✓ RECOMMENDED: {best['image_name']}")
                print(f"  GT boxes: {best['n_gt_boxes']}, Pred boxes: {best['n_pred_boxes']}")
                print(f"  FP: {best['n_false_positives']}, FN: {best['n_false_negatives']}, Confidence: {best['avg_confidence']:.3f}")
                print(f"  File: {best['image_file']}")

        # Figure 5(d) MP-IDB Species: mixed error case
        if ds_name == 'mp_idb_species':
            print("\n[Figure 5d] MP-IDB Species - Mixed Error Case (Both FP and FN)")
            print("-" * 100)
            mixed = df[(df['n_false_positives'] > 0) & (df['n_false_negatives'] > 0)]
            # Sort by total errors
            mixed['total_errors'] = mixed['n_false_positives'] + mixed['n_false_negatives']
            mixed = mixed.sort_values('total_errors', ascending=False)
            if len(mixed) > 0:
                best = mixed.iloc[0]
                figure_cases["Figure 5 - Detection Error Cases"]["5d_MPIDB_Species_Mixed"] = {
                    "image_name": best['image_name'],
                    "image_file": best['image_file'],
                    "n_gt_boxes": int(best['n_gt_boxes']),
                    "n_pred_boxes": int(best['n_pred_boxes']),
                    "n_false_positives": int(best['n_false_positives']),
                    "n_false_negatives": int(best['n_false_negatives']),
                    "avg_confidence": float(best['avg_confidence']),
                    "description": "Mixed error case (both FP and FN)"
                }
                print(f"✓ RECOMMENDED: {best['image_name']}")
                print(f"  GT boxes: {best['n_gt_boxes']}, Pred boxes: {best['n_pred_boxes']}")
                print(f"  FP: {best['n_false_positives']}, FN: {best['n_false_negatives']}, Confidence: {best['avg_confidence']:.3f}")
                print(f"  File: {best['image_file']}")

        # Figure 5(e) MD-2019: crowded mixed case
        if ds_name == 'md_2019_stages':
            print("\n[Figure 5e] MD-2019 Stages - Crowded Mixed Error Case")
            print("-" * 100)
            crowded_mixed = df[(df['n_gt_boxes'] >= 5) & (df['n_false_positives'] > 0) & (df['n_false_negatives'] > 0)]
            crowded_mixed['total_errors'] = crowded_mixed['n_false_positives'] + crowded_mixed['n_false_negatives']
            crowded_mixed = crowded_mixed.sort_values('total_errors', ascending=False)
            if len(crowded_mixed) > 0:
                best = crowded_mixed.iloc[0]
                figure_cases["Figure 5 - Detection Error Cases"]["5e_MD2019_Crowded_Mixed"] = {
                    "image_name": best['image_name'],
                    "image_file": best['image_file'],
                    "n_gt_boxes": int(best['n_gt_boxes']),
                    "n_pred_boxes": int(best['n_pred_boxes']),
                    "n_false_positives": int(best['n_false_positives']),
                    "n_false_negatives": int(best['n_false_negatives']),
                    "avg_confidence": float(best['avg_confidence']),
                    "description": "Crowded scene with mixed errors"
                }
                print(f"✓ RECOMMENDED: {best['image_name']}")
                print(f"  GT boxes: {best['n_gt_boxes']}, Pred boxes: {best['n_pred_boxes']}")
                print(f"  FP: {best['n_false_positives']}, FN: {best['n_false_negatives']}, Confidence: {best['avg_confidence']:.3f}")
                print(f"  File: {best['image_file']}")

        # Figure 5(f) MD-2019: FN case (atypical morphology)
        if ds_name == 'md_2019_stages':
            print("\n[Figure 5f] MD-2019 Stages - False Negative Case (Atypical Morphology)")
            print("-" * 100)
            fn_cases = df[df['n_false_negatives'] > 0].sort_values('n_false_negatives', ascending=False)
            if len(fn_cases) > 0:
                best = fn_cases.iloc[0]
                figure_cases["Figure 5 - Detection Error Cases"]["5f_MD2019_FN_Atypical"] = {
                    "image_name": best['image_name'],
                    "image_file": best['image_file'],
                    "n_gt_boxes": int(best['n_gt_boxes']),
                    "n_pred_boxes": int(best['n_pred_boxes']),
                    "n_false_positives": int(best['n_false_positives']),
                    "n_false_negatives": int(best['n_false_negatives']),
                    "avg_confidence": float(best['avg_confidence']),
                    "description": "False negatives due to atypical morphology"
                }
                print(f"✓ RECOMMENDED: {best['image_name']}")
                print(f"  GT boxes: {best['n_gt_boxes']}, Pred boxes: {best['n_pred_boxes']}")
                print(f"  FP: {best['n_false_positives']}, FN: {best['n_false_negatives']}, Confidence: {best['avg_confidence']:.3f}")
                print(f"  File: {best['image_file']}")

# =====================================================================
# FIGURE 6: CLASSIFICATION ERROR CASES
# =====================================================================

print("\n" + "=" * 100)
print("FIGURE 6: CLASSIFICATION ERROR CASES")
print("=" * 100)

datasets_models = {
    'iml_lifecycle': ('experiment_iml_lifecycle', 'efficientnet_b0_focal'),
    'md_2019_stages': ('experiment_md_2019_stages', 'efficientnet_b0_focal'),
    'mp_idb_species': ('experiment_mp_idb_species', 'efficientnet_b0_focal'),
    'mp_idb_stages': ('experiment_mp_idb_stages', 'efficientnet_b0_focal')
}

for ds_name, (exp_folder, model_name) in datasets_models.items():
    csv_path = os.path.join(base_path, exp_folder, 'visualizations', f'pred_classification_{model_name}', 'classification_metadata_images.csv')

    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path)
        df = df.drop_duplicates(subset='image_name')

        # Figure 6(a) IML: single error (n_incorrect=1, n_boxes=3)
        if ds_name == 'iml_lifecycle':
            print("\n[Figure 6a] IML Lifecycle - Single Error Case")
            print("-" * 100)
            single_err = df[(df['n_incorrect'] == 1) & (df['n_boxes'] == 3)]
            if len(single_err) > 0:
                best = single_err.iloc[0]
                figure_cases["Figure 6 - Classification Error Cases"]["6a_IML_Single_Error"] = {
                    "image_name": best['image_name'],
                    "image_file": best['image_file'],
                    "n_boxes": int(best['n_boxes']),
                    "n_correct": int(best['n_correct']),
                    "n_incorrect": int(best['n_incorrect']),
                    "accuracy": float(best['accuracy']),
                    "avg_confidence": float(best['avg_confidence']),
                    "description": "Single classification error (1 incorrect out of 3)"
                }
                print(f"✓ RECOMMENDED: {best['image_name']}")
                print(f"  Boxes: {best['n_boxes']}, Correct: {best['n_correct']}, Incorrect: {best['n_incorrect']}")
                print(f"  Accuracy: {best['accuracy']:.3f}, Confidence: {best['avg_confidence']:.3f}")
                print(f"  File: {best['image_file']}")

        # Figure 6(b) IML: moderate error (accuracy ~0.667, n_boxes=3)
        if ds_name == 'iml_lifecycle':
            print("\n[Figure 6b] IML Lifecycle - Moderate Error Case")
            print("-" * 100)
            # Use the same as 6a since they have same criteria (0.667 accuracy = 2/3 correct)
            moderate_err = df[(df['n_boxes'] == 3) & (df['n_incorrect'] == 1)]
            if len(moderate_err) > 0:
                # Pick a different one than 6a if possible
                if len(moderate_err) > 1:
                    best = moderate_err.iloc[1]
                else:
                    best = moderate_err.iloc[0]
                figure_cases["Figure 6 - Classification Error Cases"]["6b_IML_Moderate_Error"] = {
                    "image_name": best['image_name'],
                    "image_file": best['image_file'],
                    "n_boxes": int(best['n_boxes']),
                    "n_correct": int(best['n_correct']),
                    "n_incorrect": int(best['n_incorrect']),
                    "accuracy": float(best['accuracy']),
                    "avg_confidence": float(best['avg_confidence']),
                    "description": "Moderate error case (accuracy ~67%)"
                }
                print(f"✓ RECOMMENDED: {best['image_name']}")
                print(f"  Boxes: {best['n_boxes']}, Correct: {best['n_correct']}, Incorrect: {best['n_incorrect']}")
                print(f"  Accuracy: {best['accuracy']:.3f}, Confidence: {best['avg_confidence']:.3f}")
                print(f"  File: {best['image_file']}")

        # Figure 6(c) MP-IDB Stages: 4 incorrect among 14 (accuracy ~0.714)
        if ds_name == 'mp_idb_stages':
            print("\n[Figure 6c] MP-IDB Stages - Heavy Error Case (High Box Count)")
            print("-" * 100)
            # Look for high box count with moderate accuracy
            heavy_err = df[(df['n_boxes'] >= 10) & (df['n_incorrect'] >= 3)]
            if len(heavy_err) > 0:
                heavy_err = heavy_err.sort_values('n_incorrect', ascending=False)
                best = heavy_err.iloc[0]
                figure_cases["Figure 6 - Classification Error Cases"]["6c_MPIDB_Stages_Heavy_Error"] = {
                    "image_name": best['image_name'],
                    "image_file": best['image_file'],
                    "n_boxes": int(best['n_boxes']),
                    "n_correct": int(best['n_correct']),
                    "n_incorrect": int(best['n_incorrect']),
                    "accuracy": float(best['accuracy']),
                    "avg_confidence": float(best['avg_confidence']),
                    "description": "Heavy error case with high box count"
                }
                print(f"✓ RECOMMENDED: {best['image_name']}")
                print(f"  Boxes: {best['n_boxes']}, Correct: {best['n_correct']}, Incorrect: {best['n_incorrect']}")
                print(f"  Accuracy: {best['accuracy']:.3f}, Confidence: {best['avg_confidence']:.3f}")
                print(f"  File: {best['image_file']}")

        # Figure 6(d) MP-IDB Species: 100% wrong (accuracy=0)
        if ds_name == 'mp_idb_species':
            print("\n[Figure 6d] MP-IDB Species - Complete Failure (Species Confusion)")
            print("-" * 100)
            complete_fail = df[df['accuracy'] == 0].sort_values('n_boxes', ascending=False)
            if len(complete_fail) > 0:
                # Prefer single box case for clarity
                single_box_fail = complete_fail[complete_fail['n_boxes'] == 1]
                if len(single_box_fail) > 0:
                    best = single_box_fail.iloc[0]
                else:
                    best = complete_fail.iloc[0]
                figure_cases["Figure 6 - Classification Error Cases"]["6d_MPIDB_Species_Complete_Fail"] = {
                    "image_name": best['image_name'],
                    "image_file": best['image_file'],
                    "n_boxes": int(best['n_boxes']),
                    "n_correct": int(best['n_correct']),
                    "n_incorrect": int(best['n_incorrect']),
                    "accuracy": float(best['accuracy']),
                    "avg_confidence": float(best['avg_confidence']),
                    "description": "Complete failure - species confusion"
                }
                print(f"✓ RECOMMENDED: {best['image_name']}")
                print(f"  Boxes: {best['n_boxes']}, Correct: {best['n_correct']}, Incorrect: {best['n_incorrect']}")
                print(f"  Accuracy: {best['accuracy']:.3f}, Confidence: {best['avg_confidence']:.3f}")
                print(f"  File: {best['image_file']}")

        # Figure 6(e) MD-2019: heavy error (accuracy ~0.25, n_boxes=8)
        if ds_name == 'md_2019_stages':
            print("\n[Figure 6e] MD-2019 Stages - Heavy Error Case")
            print("-" * 100)
            heavy_err = df[(df['n_boxes'] >= 5) & (df['accuracy'] <= 0.3) & (df['accuracy'] > 0)]
            if len(heavy_err) > 0:
                heavy_err = heavy_err.sort_values('n_boxes', ascending=False)
                best = heavy_err.iloc[0]
                figure_cases["Figure 6 - Classification Error Cases"]["6e_MD2019_Heavy_Error"] = {
                    "image_name": best['image_name'],
                    "image_file": best['image_file'],
                    "n_boxes": int(best['n_boxes']),
                    "n_correct": int(best['n_correct']),
                    "n_incorrect": int(best['n_incorrect']),
                    "accuracy": float(best['accuracy']),
                    "avg_confidence": float(best['avg_confidence']),
                    "description": "Heavy error case (low accuracy)"
                }
                print(f"✓ RECOMMENDED: {best['image_name']}")
                print(f"  Boxes: {best['n_boxes']}, Correct: {best['n_correct']}, Incorrect: {best['n_incorrect']}")
                print(f"  Accuracy: {best['accuracy']:.3f}, Confidence: {best['avg_confidence']:.3f}")
                print(f"  File: {best['image_file']}")

        # Figure 6(f) MD-2019: perfect case (accuracy=1.0, n_boxes=10)
        if ds_name == 'md_2019_stages':
            print("\n[Figure 6f] MD-2019 Stages - Perfect Case (High Complexity)")
            print("-" * 100)
            perfect = df[(df['accuracy'] == 1.0) & (df['n_boxes'] >= 8)]
            if len(perfect) > 0:
                perfect = perfect.sort_values('n_boxes', ascending=False)
                best = perfect.iloc[0]
                figure_cases["Figure 6 - Classification Error Cases"]["6f_MD2019_Perfect"] = {
                    "image_name": best['image_name'],
                    "image_file": best['image_file'],
                    "n_boxes": int(best['n_boxes']),
                    "n_correct": int(best['n_correct']),
                    "n_incorrect": int(best['n_incorrect']),
                    "accuracy": float(best['accuracy']),
                    "avg_confidence": float(best['avg_confidence']),
                    "description": "Perfect classification on crowded image"
                }
                print(f"✓ RECOMMENDED: {best['image_name']}")
                print(f"  Boxes: {best['n_boxes']}, Correct: {best['n_correct']}, Incorrect: {best['n_incorrect']}")
                print(f"  Accuracy: {best['accuracy']:.3f}, Confidence: {best['avg_confidence']:.3f}")
                print(f"  File: {best['image_file']}")

# Save to JSON file
output_file = r"C:\Users\MyPC PRO\Documents\hello_world\figure_cases_identification.json"
with open(output_file, 'w') as f:
    json.dump(figure_cases, f, indent=2)

print("\n" + "=" * 100)
print(f"✓ Figure cases saved to: {output_file}")
print("=" * 100)
