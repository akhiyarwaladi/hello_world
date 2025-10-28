"""
Extract actual training times from classification experiments
NO HALLUCINATION - Direct extraction from results.txt files
"""
from pathlib import Path
import re

results_dir = Path("results/optA_20251016_200330/experiments")

datasets = {
    'iml_lifecycle': 'IML Lifecycle',
    'mp_idb_species': 'MP-IDB Species',
    'mp_idb_stages': 'MP-IDB Stages',
    'md_2019_stages': 'MD_2019'
}

models = [
    'densenet121',
    'efficientnet_b0',
    'efficientnet_b1',
    'efficientnet_b2',
    'resnet50',
    'resnet101'
]

print("="*80)
print("CLASSIFICATION TRAINING TIME EXTRACTION")
print("="*80)
print(f"Source: {results_dir}")
print()

all_times = {}

for dataset_key, dataset_name in datasets.items():
    print(f"\n{dataset_name} ({dataset_key}):")
    print("-" * 60)

    all_times[dataset_key] = {}

    for model in models:
        model_dir = results_dir / f"experiment_{dataset_key}" / f"cls_{model}_focal"
        results_file = model_dir / "results.txt"

        if results_file.exists():
            with open(results_file, 'r') as f:
                content = f.read()

            # Extract training time
            match = re.search(r'Training Time:\s+([\d.]+)\s+min', content)
            if match:
                training_time = float(match.group(1))
                all_times[dataset_key][model] = training_time
                print(f"  {model:20s}: {training_time:5.1f} min")
            else:
                print(f"  {model:20s}: NOT FOUND ❌")
                all_times[dataset_key][model] = None
        else:
            print(f"  {model:20s}: FILE NOT EXISTS ❌")
            all_times[dataset_key][model] = None

print("\n" + "="*80)
print("PYTHON DICT FORMAT (for table script)")
print("="*80)
print()

# Print as Python dict for easy copy-paste
print("training_times = {")
for dataset_key in datasets.keys():
    print(f"    '{dataset_key}': {{")
    for model in models:
        time_val = all_times[dataset_key].get(model)
        if time_val is not None:
            print(f"        '{model}': {time_val},")
        else:
            print(f"        '{model}': None,  # ⚠️ NOT FOUND")
    print("    },")
print("}")

print("\n" + "="*80)
print("SUMMARY")
print("="*80)

total_models = len(datasets) * len(models)
found_models = sum(1 for d in all_times.values() for t in d.values() if t is not None)

print(f"Total models: {total_models}")
print(f"Found: {found_models}")
print(f"Missing: {total_models - found_models}")

if found_models == total_models:
    print("\n✅ ALL training times found!")
else:
    print(f"\n⚠️ {total_models - found_models} training times missing!")

print("\n" + "="*80)
