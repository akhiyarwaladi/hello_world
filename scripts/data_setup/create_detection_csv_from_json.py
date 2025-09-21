#!/usr/bin/env python3
"""
Extract detection annotations from processed JSON data and create CSV files
like Falciparum format for Vivax, Malariae, and Ovale species.
"""

import json
import csv
from pathlib import Path
from collections import defaultdict

def load_annotations(json_file):
    """Load unified annotations from JSON file"""
    with open(json_file, 'r') as f:
        return json.load(f)

def create_detection_csv(annotations, species_name, output_dir):
    """Create detection CSV file for a specific species"""

    # Filter annotations for this species
    species_annotations = [
        ann for ann in annotations
        if ann.get('original_species', '').lower() == species_name.lower()
    ]

    if not species_annotations:
        print(f"⚠️ No annotations found for {species_name}")
        return None

    # Group by original image
    image_groups = defaultdict(list)
    for ann in species_annotations:
        original_path = ann['original_path']
        image_name = Path(original_path).name
        image_groups[image_name].append(ann)

    # Create output directory
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Create CSV file
    csv_file = output_dir / f"mp-idb-{species_name.lower()}.csv"

    print(f"Creating {csv_file} with {len(species_annotations)} objects from {len(image_groups)} images")

    with open(csv_file, 'w', newline='') as f:
        writer = csv.writer(f)

        # Write header (same format as Falciparum)
        writer.writerow(['filename', 'parasite_type', 'xmin', 'xmax', 'ymin', 'ymax'])

        # Write annotations
        for image_name, objects in image_groups.items():
            for obj in objects:
                # Extract original bbox coordinates
                x = int(obj['original_bbox_x'])
                y = int(obj['original_bbox_y'])
                w = int(obj['original_bbox_w'])
                h = int(obj['original_bbox_h'])

                # Convert to xmin, xmax, ymin, ymax format
                xmin = x
                xmax = x + w
                ymin = y
                ymax = y + h

                # Use generic parasite type for detection (single class)
                parasite_type = "parasite"

                writer.writerow([image_name, parasite_type, xmin, xmax, ymin, ymax])

    print(f"✅ Created {csv_file}")
    return csv_file

def main():
    # Paths
    annotations_file = Path("data/integrated/annotations/unified_annotations.json")
    output_base_dir = Path("data/raw/mp_idb")

    # Load annotations
    print("Loading unified annotations...")
    annotations = load_annotations(annotations_file)
    print(f"Loaded {len(annotations)} total annotations")

    # Create CSV files for each species (except Falciparum which already has CSV)
    species_to_process = ['P_vivax', 'P_malariae', 'P_ovale']

    for species in species_to_process:
        species_dir = output_base_dir / species.split('_')[1].capitalize()
        csv_file = create_detection_csv(annotations, species, species_dir)

        if csv_file:
            print(f"✅ {species}: {csv_file}")
        else:
            print(f"❌ {species}: No data found")

    print("\n🎉 Detection CSV files created! Now you can run the detection preprocessing.")

if __name__ == "__main__":
    main()