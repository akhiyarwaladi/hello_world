"""
Extract ACTUAL detection training times from experiments
NO HALLUCINATION - Direct extraction from results.csv files
"""
from pathlib import Path
import pandas as pd

results_dir = Path("results/optA_20251016_200330/experiments")

datasets = {
    'iml_lifecycle': 'IML Lifecycle',
    'mp_idb_species': 'MP-IDB Species',
    'mp_idb_stages': 'MP-IDB Stages',
    'md_2019_stages': 'MD_2019'
}

models = ['yolo10', 'yolo11', 'yolo12']

print("="*80)
print("DETECTION TRAINING TIME EXTRACTION")
print("="*80)
print(f"Source: {results_dir}")
print("Format: results.csv - 'time' column (cumulative seconds at last epoch)")
print()

all_times = {}

for dataset_key, dataset_name in datasets.items():
    print(f"\n{dataset_name} ({dataset_key}):")
    print("-" * 60)

    all_times[dataset_key] = {}

    for model in models:
        model_dir = results_dir / f"experiment_{dataset_key}" / f"det_{model}"
        results_file = model_dir / "results.csv"

        if results_file.exists():
            try:
                df = pd.read_csv(results_file)

                # Get last epoch's cumulative time (in seconds)
                final_time_sec = df['time'].iloc[-1]
                final_time_min = final_time_sec / 60
                final_time_hr = final_time_sec / 3600

                all_times[dataset_key][model] = final_time_hr

                print(f"  {model:8s}: {final_time_sec:7.2f} sec = {final_time_min:5.2f} min = {final_time_hr:6.4f} hr")
            except Exception as e:
                print(f"  {model:8s}: ERROR reading CSV - {e}")
                all_times[dataset_key][model] = None
        else:
            print(f"  {model:8s}: FILE NOT EXISTS ❌")
            all_times[dataset_key][model] = None

print("\n" + "="*80)
print("PYTHON DICT FORMAT (for table script)")
print("="*80)
print()

# Print as Python dict for easy copy-paste (in HOURS)
print("detection_training_times = {")
for dataset_key in datasets.keys():
    print(f"    '{dataset_key}': {{")
    for model in models:
        time_val = all_times[dataset_key].get(model)
        if time_val is not None:
            print(f"        '{model}': {time_val:.4f},  # hours")
        else:
            print(f"        '{model}': None,  # ⚠️ NOT FOUND")
    print("    },")
print("}")

print("\n" + "="*80)
print("SUMMARY")
print("="*80)

total_models = len(datasets) * len(models)
found_models = sum(1 for d in all_times.values() for t in d.values() if t is not None)

print(f"Total detection models: {total_models}")
print(f"Found: {found_models}")
print(f"Missing: {total_models - found_models}")

if found_models == total_models:
    print("\n✅ ALL detection training times found!")
    print()
    print("TOTAL DETECTION TRAINING TIME (Shared Approach):")
    total_time = sum(t for d in all_times.values() for t in d.values() if t is not None)
    print(f"  {total_time:.4f} hours = {total_time * 60:.2f} minutes")
else:
    print(f"\n⚠️ {total_models - found_models} training times missing!")

print("\n" + "="*80)
