import os
import json

exp_path = 'results/optA_20251207_233941/experiments'
datasets = [d for d in os.listdir(exp_path) if d.startswith('experiment_')]

print('='*60)
print('EXPERIMENT COMPLETION CHECK')
print('='*60)
print()

total_det = 0
total_cls = 0

for ds in sorted(datasets):
    ds_path = os.path.join(exp_path, ds)
    det_models = [d for d in os.listdir(ds_path) if d.startswith('det_') and os.path.isdir(os.path.join(ds_path, d))]
    cls_models = [d for d in os.listdir(ds_path) if d.startswith('cls_') and os.path.isdir(os.path.join(ds_path, d))]

    total_det += len(det_models)
    total_cls += len(cls_models)

    status_det = '✅' if len(det_models) == 3 else '⚠️'
    status_cls = '✅' if len(cls_models) == 6 else '⚠️'

    print(f'{ds.replace("experiment_", "")}:')
    print(f'  Detection models: {len(det_models)}/3 {status_det}')
    if det_models:
        print(f'    Models: {", ".join(sorted(det_models))}')
    print(f'  Classification models: {len(cls_models)}/6 {status_cls}')
    if cls_models:
        print(f'    Models: {", ".join(sorted(cls_models))}')
    print()

print('='*60)
print('OVERALL SUMMARY')
print('='*60)
print(f'Total Detection Models: {total_det}/12 (4 datasets × 3 YOLO models)')
print(f'Total Classification Models: {total_cls}/24 (4 datasets × 6 architectures)')
print()

completion_pct = ((total_det + total_cls) / (12 + 24)) * 100
print(f'Overall Completion: {completion_pct:.1f}%')

if total_det == 12 and total_cls == 24:
    print('\n✅ ALL EXPERIMENTS COMPLETED SUCCESSFULLY!')
else:
    print(f'\n⚠️ INCOMPLETE: {12-total_det} detection, {24-total_cls} classification models missing')
