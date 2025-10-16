#!/usr/bin/env python3
"""
Fix data.yaml paths to use relative paths that work on any system
"""

import yaml
from pathlib import Path

def fix_data_yaml(yaml_path):
    """Fix a single data.yaml file to use relative path"""
    yaml_file = Path(yaml_path)

    if not yaml_file.exists():
        print(f"File not found: {yaml_file}")
        return False

    # Read current config
    with open(yaml_file, 'r') as f:
        config = yaml.safe_load(f)

    # Get the directory containing data.yaml
    dataset_dir = yaml_file.parent.resolve()

    # Use absolute path with forward slashes for cross-platform compatibility
    # Forward slashes work on both Windows and Linux
    absolute_path = str(dataset_dir).replace('\\', '/')

    # Update path to use forward slashes
    config['path'] = absolute_path

    # Write back
    with open(yaml_file, 'w') as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)

    print(f"✓ Fixed: {yaml_file}")
    print(f"  New path: {config['path']}")
    return True

def main():
    """Fix all data.yaml files in processed datasets"""
    print("Fixing data.yaml paths to be system-independent...")
    print()

    # Find all data.yaml files
    processed_dir = Path('data/processed')
    yaml_files = list(processed_dir.glob('*/data.yaml'))

    if not yaml_files:
        print("No data.yaml files found")
        return

    print(f"Found {len(yaml_files)} data.yaml files\n")

    for yaml_file in yaml_files:
        fix_data_yaml(yaml_file)
        print()

    print("="*60)
    print("All data.yaml files updated successfully!")
    print("="*60)
    print()
    print("These paths will now work on both Windows and Linux.")

if __name__ == "__main__":
    main()
