"""
Select best qualitative analysis images from each dataset
Criteria: Interesting cases with minority classes, multiple parasites, challenging scenarios
"""
import os
import shutil
from pathlib import Path

# Paths
base_exp = Path('results/optA_20251016_200330/experiments')
output_dir = Path('luaran/templates/figures/qualitative_analysis')

# Create output directory
output_dir.mkdir(parents=True, exist_ok=True)

# Dataset configurations
datasets = {
    'experiment_iml_lifecycle': {
        'yolo': 'yolo11',
        'output_name': 'iml_lifecycle_detection.png',
        'description': 'IML Lifecycle - Balanced detection across 4 stages',
        # Prefer images with interesting filenames
        'prefer_patterns': []  # Will pick manually
    },
    'experiment_mp_idb_species': {
        'yolo': 'yolo11',  # or yolo12 for best performance
        'output_name': 'mp_idb_species_detection.png',
        'description': 'MP-IDB Species - Minority species detection',
        # Look for patterns suggesting minority classes
        'prefer_patterns': ['ovale', 'malariae', 'vivax']
    },
    'experiment_mp_idb_stages': {
        'yolo': 'yolo12',  # Best performer (96.27% mAP@50)
        'output_name': 'mp_idb_stages_detection.png',
        'description': 'MP-IDB Stages - High-density overlapping parasites',
        # Look for gametocyte (G) or schizont (S) - minority classes
        'prefer_patterns': ['G', 'S', 'T']
    },
    'experiment_md_2019_stages': {
        'yolo': 'yolo11',
        'output_name': 'md_2019_stages_detection.png',
        'description': 'MD_2019 - Natural variation across patients',
        'prefer_patterns': []  # Natural variation dataset
    }
}

selected_images = {}

for dataset_name, config in datasets.items():
    viz_path = base_exp / dataset_name / 'visualizations' / f'pred_detection_{config["yolo"]}'

    if not viz_path.exists():
        print(f"⚠️  Path not found: {viz_path}")
        continue

    # Get all PNG images
    images = sorted([f for f in viz_path.iterdir() if f.suffix == '.png'])

    if not images:
        print(f"⚠️  No images found in {viz_path}")
        continue

    print(f"\n=== {dataset_name} ===")
    print(f"Total images: {len(images)}")
    print(f"Using YOLO variant: {config['yolo']}")

    # Selection strategy
    selected = None

    if config['prefer_patterns']:
        # Look for images with preferred patterns
        for pattern in config['prefer_patterns']:
            candidates = [img for img in images if pattern in img.name]
            if candidates:
                selected = candidates[0]  # Pick first match
                print(f"✓ Selected (pattern '{pattern}'): {selected.name}")
                break

    # If no pattern match, pick a middle image (often representative)
    if selected is None:
        selected = images[len(images) // 2]
        print(f"✓ Selected (middle): {selected.name}")

    # Copy to output folder
    output_path = output_dir / config['output_name']
    shutil.copy2(selected, output_path)

    selected_images[dataset_name] = {
        'original': selected.name,
        'output': config['output_name'],
        'description': config['description'],
        'yolo': config['yolo'],
        'path': str(output_path)
    }

    print(f"✓ Copied to: {output_path}")

# Print summary
print("\n" + "="*70)
print("QUALITATIVE ANALYSIS IMAGES SELECTED")
print("="*70)

for dataset, info in selected_images.items():
    print(f"\n{dataset}:")
    print(f"  Original: {info['original']}")
    print(f"  Output:   {info['output']}")
    print(f"  YOLO:     {info['yolo']}")
    print(f"  Desc:     {info['description']}")

# Save metadata
metadata_path = output_dir / 'selected_images_metadata.txt'
with open(metadata_path, 'w') as f:
    f.write("Qualitative Analysis Images - Selected for KINETIK Paper\n")
    f.write("="*70 + "\n\n")

    for dataset, info in selected_images.items():
        f.write(f"{dataset}:\n")
        f.write(f"  Original file: {info['original']}\n")
        f.write(f"  Output file:   {info['output']}\n")
        f.write(f"  YOLO variant:  {info['yolo']}\n")
        f.write(f"  Description:   {info['description']}\n")
        f.write(f"  Full path:     {info['path']}\n\n")

print(f"\n✅ Metadata saved to: {metadata_path}")
print(f"✅ All images saved to: {output_dir}")
