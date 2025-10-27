"""
Select classification visualization images showing:
- Detection bbox + Classification predictions
- Ground truth vs Predicted class
- Confidence scores
"""
import shutil
from pathlib import Path

base_exp = Path('results/optA_20251016_200330/experiments')
output_dir = Path('luaran/templates/figures/qualitative_analysis')
output_dir.mkdir(parents=True, exist_ok=True)

# For each dataset, use the SAME image as detection but from classification folder
# This shows detection + classification together

datasets = {
    'experiment_iml_lifecycle': {
        'detection_image': 'PA171862.png',
        'classifier': 'efficientnet_b1_focal',
        'output_name': 'iml_lifecycle_classification.png',
        'description': 'IML Lifecycle - EfficientNet-B1 classification on detected parasites'
    },
    'experiment_mp_idb_species': {
        'detection_image': '1701151546-0008-R-T.png',
        'classifier': 'efficientnet_b1_focal',
        'output_name': 'mp_idb_species_classification.png',
        'description': 'MP-IDB Species - EfficientNet-B1 species classification'
    },
    'experiment_mp_idb_stages': {
        'detection_image': '1704282807-0019-R_G.png',
        'classifier': 'resnet50_focal',  # Best performer
        'output_name': 'mp_idb_stages_classification.png',
        'description': 'MP-IDB Stages - ResNet50 classification on minority gametocyte'
    },
    'experiment_md_2019_stages': {
        'detection_image': 'Trip 064 Day 2 25-11-05 Image 7_5.png',
        'classifier': 'efficientnet_b0_focal',  # Best performer
        'output_name': 'md_2019_stages_classification.png',
        'description': 'MD_2019 - EfficientNet-B0 multi-stage classification'
    }
}

selected_images = {}

for dataset_name, config in datasets.items():
    cls_viz_path = base_exp / dataset_name / 'visualizations' / f'pred_classification_{config["classifier"]}'

    source_image = cls_viz_path / config['detection_image']

    if not source_image.exists():
        print(f"⚠️  Image not found: {source_image}")
        # Try to find similar image
        if cls_viz_path.exists():
            images = list(cls_viz_path.glob('*.png'))
            if images:
                source_image = images[len(images) // 2]  # Middle image
                print(f"   Using alternative: {source_image.name}")

    if source_image.exists():
        output_path = output_dir / config['output_name']
        shutil.copy2(source_image, output_path)

        selected_images[dataset_name] = {
            'original': source_image.name,
            'output': config['output_name'],
            'classifier': config['classifier'],
            'description': config['description'],
            'path': str(output_path)
        }

        print(f"✅ {dataset_name}:")
        print(f"   {source_image.name} → {config['output_name']}")
    else:
        print(f"❌ No images found for {dataset_name}")

# Save metadata
print("\n" + "="*80)
print("CLASSIFICATION VISUALIZATION IMAGES SELECTED")
print("="*80)

metadata_path = output_dir / 'classification_images_metadata.txt'
with open(metadata_path, 'w') as f:
    f.write("Classification Visualization Images - Selected for KINETIK Paper\n")
    f.write("="*80 + "\n\n")

    for dataset, info in selected_images.items():
        f.write(f"{dataset}:\n")
        f.write(f"  Original file: {info['original']}\n")
        f.write(f"  Output file:   {info['output']}\n")
        f.write(f"  Classifier:    {info['classifier']}\n")
        f.write(f"  Description:   {info['description']}\n")
        f.write(f"  Full path:     {info['path']}\n\n")

        print(f"\n{dataset}:")
        print(f"  Classifier: {info['classifier']}")
        print(f"  Output: {info['output']}")

print(f"\n✅ Metadata saved to: {metadata_path}")
print(f"✅ All classification images saved to: {output_dir}")
