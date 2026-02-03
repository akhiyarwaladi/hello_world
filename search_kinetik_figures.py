import pandas as pd
from pathlib import Path

exp_dir = Path('results/optA_20251207_233941/experiments')

print('=' * 80)
print('SEARCHING FOR KINETIK PAPER FIGURE MATCHES')
print('=' * 80)

# ============================================================================
# FIGURE 5c: MP-IDB Stages - exactly 8 false positives
# ============================================================================
print('\n' + '=' * 80)
print('FIGURE 5c: MP-IDB Stages - Detection with 8 False Positives')
print('=' * 80)

stages_vis = exp_dir / 'experiment_mp_idb_stages' / 'visualizations'
for det_model in ['pred_detection_yolo10', 'pred_detection_yolo11', 'pred_detection_yolo12']:
    csv_path = stages_vis / det_model / 'detection_metadata.csv'
    if csv_path.exists():
        df = pd.read_csv(csv_path)
        matches = df[df['n_false_positives'] == 8]
        if len(matches) > 0:
            print(f'\n>>> MODEL: {det_model}')
            print(f'>>> Found {len(matches)} images with exactly 8 false positives:')
            for _, row in matches.iterrows():
                print(f'  - Image: {row["image_name"]}')
                print(f'    GT boxes: {row["n_gt_boxes"]}, Pred boxes: {row["n_pred_boxes"]}')
                print(f'    Correct: {row["n_correct_matches"]}, FP: {row["n_false_positives"]}, FN: {row["n_false_negatives"]}')
                print(f'    Confidence: {row["avg_confidence"]:.3f}')
                print()
        else:
            # Show closest matches
            df['fp_diff'] = abs(df['n_false_positives'] - 8)
            closest = df.nsmallest(3, 'fp_diff')
            print(f'\n>>> MODEL: {det_model} (NO EXACT MATCH - showing closest 3)')
            for _, row in closest.iterrows():
                print(f'  - {row["image_name"]}: FP={row["n_false_positives"]} (target: 8)')

# ============================================================================
# FIGURE 6a & 6b: IML Lifecycle - n_boxes=3, n_incorrect=1, accuracy≈0.667
# ============================================================================
print('\n' + '=' * 80)
print('FIGURE 6a & 6b: IML Lifecycle - Classification (n_boxes=3, n_incorrect=1, acc≈0.667)')
print('=' * 80)

iml_vis = exp_dir / 'experiment_iml_lifecycle' / 'visualizations'
all_matches = []

for cls_dir in iml_vis.glob('pred_classification_*'):
    csv_path = cls_dir / 'classification_metadata_images.csv'
    if csv_path.exists():
        df = pd.read_csv(csv_path)
        # Filter: n_boxes=3, n_incorrect=1, accuracy between 0.6-0.7
        matches = df[
            (df['n_boxes'] == 3) &
            (df['n_incorrect'] == 1) &
            (df['accuracy'] >= 0.6) &
            (df['accuracy'] <= 0.7)
        ]
        if len(matches) > 0:
            for _, row in matches.iterrows():
                all_matches.append({
                    'model': cls_dir.name.replace('pred_classification_', ''),
                    'image': row['image_name'],
                    'n_boxes': row['n_boxes'],
                    'n_correct': row['n_correct'],
                    'n_incorrect': row['n_incorrect'],
                    'accuracy': row['accuracy'],
                    'avg_confidence': row.get('avg_confidence', 'N/A')
                })

if all_matches:
    print(f'\n>>> Found {len(all_matches)} matching images:')
    for m in all_matches:
        print(f'  - Model: {m["model"]}')
        print(f'    Image: {m["image"]}')
        print(f'    Stats: boxes={m["n_boxes"]}, correct={m["n_correct"]}, incorrect={m["n_incorrect"]}, acc={m["accuracy"]:.3f}')
        print()
else:
    print('>>> NO EXACT MATCHES FOUND - Searching for closest approximations...')
    for cls_dir in iml_vis.glob('pred_classification_*'):
        csv_path = cls_dir / 'classification_metadata_images.csv'
        if csv_path.exists():
            df = pd.read_csv(csv_path)
            # Find closest to acc=0.667, n_boxes=3
            df['score'] = abs(df['accuracy'] - 0.667) + abs(df['n_boxes'] - 3) * 0.1
            closest = df.nsmallest(3, 'score')
            print(f'\n>>> Model: {cls_dir.name.replace("pred_classification_", "")} (top 3 closest)')
            for _, row in closest.iterrows():
                print(f'  - {row["image_name"]}: boxes={row["n_boxes"]}, incorrect={row["n_incorrect"]}, acc={row["accuracy"]:.3f}')

# ============================================================================
# FIGURE 6c: MP-IDB Stages - n_boxes≈14, n_incorrect≈4, accuracy≈0.714
# ============================================================================
print('\n' + '=' * 80)
print('FIGURE 6c: MP-IDB Stages - Classification (n_boxes≈14, n_incorrect≈4, acc≈0.714)')
print('=' * 80)

stages_vis = exp_dir / 'experiment_mp_idb_stages' / 'visualizations'
all_matches = []

for cls_dir in stages_vis.glob('pred_classification_*'):
    csv_path = cls_dir / 'classification_metadata_images.csv'
    if csv_path.exists():
        df = pd.read_csv(csv_path)
        # Filter: n_boxes >= 10, accuracy between 0.65-0.80
        matches = df[
            (df['n_boxes'] >= 10) &
            (df['accuracy'] >= 0.65) &
            (df['accuracy'] <= 0.80)
        ]
        if len(matches) > 0:
            for _, row in matches.iterrows():
                all_matches.append({
                    'model': cls_dir.name.replace('pred_classification_', ''),
                    'image': row['image_name'],
                    'n_boxes': row['n_boxes'],
                    'n_correct': row['n_correct'],
                    'n_incorrect': row['n_incorrect'],
                    'accuracy': row['accuracy'],
                    'avg_confidence': row.get('avg_confidence', 'N/A')
                })

if all_matches:
    # Sort by how close to ideal (n_boxes=14, acc=0.714)
    for m in all_matches:
        m['score'] = abs(m['n_boxes'] - 14) * 0.05 + abs(m['accuracy'] - 0.714)
    all_matches.sort(key=lambda x: x['score'])

    print(f'\n>>> Found {len(all_matches)} candidates (showing top 10):')
    for m in all_matches[:10]:
        print(f'  - Model: {m["model"]}')
        print(f'    Image: {m["image"]}')
        print(f'    Stats: boxes={m["n_boxes"]}, correct={m["n_correct"]}, incorrect={m["n_incorrect"]}, acc={m["accuracy"]:.3f}')
        print()

# ============================================================================
# FIGURE 6e: MD-2019 - n_boxes=8, n_incorrect=6, accuracy=0.25
# ============================================================================
print('\n' + '=' * 80)
print('FIGURE 6e: MD-2019 Stages - Classification (n_boxes=8, n_incorrect=6, acc=0.25)')
print('=' * 80)

md2019_vis = exp_dir / 'experiment_md_2019_stages' / 'visualizations'
all_matches = []

for cls_dir in md2019_vis.glob('pred_classification_*'):
    csv_path = cls_dir / 'classification_metadata_images.csv'
    if csv_path.exists():
        df = pd.read_csv(csv_path)
        # Exact match: n_boxes=8, accuracy=0.25
        matches = df[
            (df['n_boxes'] == 8) &
            (df['accuracy'] == 0.25)
        ]
        if len(matches) > 0:
            for _, row in matches.iterrows():
                all_matches.append({
                    'model': cls_dir.name.replace('pred_classification_', ''),
                    'image': row['image_name'],
                    'n_boxes': row['n_boxes'],
                    'n_correct': row['n_correct'],
                    'n_incorrect': row['n_incorrect'],
                    'accuracy': row['accuracy'],
                    'avg_confidence': row.get('avg_confidence', 'N/A')
                })

if all_matches:
    print(f'\n>>> Found {len(all_matches)} exact matches:')
    for m in all_matches:
        print(f'  - Model: {m["model"]}')
        print(f'    Image: {m["image"]}')
        print(f'    Stats: boxes={m["n_boxes"]}, correct={m["n_correct"]}, incorrect={m["n_incorrect"]}, acc={m["accuracy"]:.3f}')
        print()
else:
    print('>>> NO EXACT MATCHES - Searching for closest...')
    for cls_dir in md2019_vis.glob('pred_classification_*'):
        csv_path = cls_dir / 'classification_metadata_images.csv'
        if csv_path.exists():
            df = pd.read_csv(csv_path)
            df['score'] = abs(df['n_boxes'] - 8) * 0.1 + abs(df['accuracy'] - 0.25)
            closest = df.nsmallest(5, 'score')
            print(f'\n>>> Model: {cls_dir.name.replace("pred_classification_", "")} (top 5 closest)')
            for _, row in closest.iterrows():
                print(f'  - {row["image_name"]}: boxes={row["n_boxes"]}, incorrect={row["n_incorrect"]}, acc={row["accuracy"]:.3f}')

# ============================================================================
# FIGURE 6f: MD-2019 - n_boxes≥8 (ideally 10), accuracy=1.0
# ============================================================================
print('\n' + '=' * 80)
print('FIGURE 6f: MD-2019 Stages - Classification (n_boxes≥8, accuracy=1.0)')
print('=' * 80)

all_matches = []

for cls_dir in md2019_vis.glob('pred_classification_*'):
    csv_path = cls_dir / 'classification_metadata_images.csv'
    if csv_path.exists():
        df = pd.read_csv(csv_path)
        # Filter: accuracy=1.0, n_boxes >= 8
        matches = df[
            (df['accuracy'] == 1.0) &
            (df['n_boxes'] >= 8)
        ]
        if len(matches) > 0:
            for _, row in matches.iterrows():
                all_matches.append({
                    'model': cls_dir.name.replace('pred_classification_', ''),
                    'image': row['image_name'],
                    'n_boxes': row['n_boxes'],
                    'n_correct': row['n_correct'],
                    'n_incorrect': row['n_incorrect'],
                    'accuracy': row['accuracy'],
                    'avg_confidence': row.get('avg_confidence', 'N/A')
                })

if all_matches:
    # Sort by n_boxes descending (prefer more boxes)
    all_matches.sort(key=lambda x: -x['n_boxes'])

    print(f'\n>>> Found {len(all_matches)} perfect accuracy images with ≥8 boxes (showing top 10):')
    for m in all_matches[:10]:
        print(f'  - Model: {m["model"]}')
        print(f'    Image: {m["image"]}')
        print(f'    Stats: boxes={m["n_boxes"]}, correct={m["n_correct"]}, incorrect={m["n_incorrect"]}, acc={m["accuracy"]:.3f}')
        print()

print('\n' + '=' * 80)
print('SEARCH COMPLETE')
print('=' * 80)
