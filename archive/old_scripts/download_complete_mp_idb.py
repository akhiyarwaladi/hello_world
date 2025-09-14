#!/usr/bin/env python3
"""
Download complete MP-IDB dataset from Dataset Ninja
Try to get the version with complete annotations for all 4 species
"""

import requests
import os
from pathlib import Path
import subprocess

def download_mp_idb_complete():
    """Download complete MP-IDB from Dataset Ninja or alternative sources"""

    print("🔍 ATTEMPTING TO DOWNLOAD COMPLETE MP-IDB DATASET")
    print("=" * 60)

    # Try dataset-tools if available
    try:
        import dataset_tools as dtools
        print("✅ dataset-tools found, attempting download...")

        output_dir = Path("data/raw/mp_idb_complete")
        output_dir.mkdir(parents=True, exist_ok=True)

        dtools.download(dataset='MP IDB', dst_dir=str(output_dir))

        print(f"✅ Downloaded to: {output_dir}")
        return True

    except ImportError:
        print("❌ dataset-tools not available")
    except Exception as e:
        print(f"❌ Error with dataset-tools: {e}")

    # Try alternative download methods
    print("\n🔄 Trying alternative download methods...")

    # Method 1: Try direct download from Dataset Ninja
    try:
        print("Attempting Dataset Ninja direct download...")

        # This would require specific API or URL - placeholder for now
        print("⚠️  Direct download requires specific API access")

    except Exception as e:
        print(f"❌ Alternative download failed: {e}")

    # Method 2: Check if we can find annotations in current dataset
    print("\n🔍 ANALYZING CURRENT MP-IDB STRUCTURE")

    current_mpidb = Path("data/raw/mp_idb")
    if current_mpidb.exists():
        print(f"Current MP-IDB found at: {current_mpidb}")

        # Look for any annotation files we might have missed
        all_files = list(current_mpidb.rglob("*"))

        annotation_files = []
        for file in all_files:
            if file.suffix.lower() in ['.csv', '.json', '.xml', '.txt']:
                if 'git' not in str(file).lower():
                    annotation_files.append(file)

        print(f"Found annotation files:")
        for file in annotation_files:
            print(f"  - {file}")

        # Check if there might be hidden/compressed annotations
        compressed_files = list(current_mpidb.rglob("*.zip")) + list(current_mpidb.rglob("*.tar*"))
        if compressed_files:
            print(f"Found compressed files that might contain annotations:")
            for file in compressed_files:
                print(f"  - {file}")

    return False

def check_supervisely_format():
    """Check if we can work with Supervisely format annotations"""

    print("\n🔍 CHECKING FOR SUPERVISELY FORMAT DATA")

    # Look for supervisely format files
    mpidb_path = Path("data/raw/mp_idb")

    if mpidb_path.exists():
        # Check for supervisely-style annotation directories
        sly_dirs = ['ann', 'annotations', 'meta.json']

        for species_dir in ['Falciparum', 'Vivax', 'Malariae', 'Ovale']:
            species_path = mpidb_path / species_dir
            if species_path.exists():
                print(f"\n📁 Checking {species_dir}:")

                # List all files to see what we have
                all_files = list(species_path.rglob("*"))
                for file in all_files:
                    if file.is_file():
                        print(f"  - {file.name} ({file.suffix})")

def main():
    """Main function to attempt complete MP-IDB download"""

    # First attempt download
    success = download_mp_idb_complete()

    # If download failed, analyze what we have
    if not success:
        check_supervisely_format()

        print("\n💡 RECOMMENDATIONS:")
        print("1. Try installing dataset-tools manually:")
        print("   pip install dataset-tools --no-deps")
        print("   (install only core package without heavy dependencies)")

        print("\n2. Check if MP-IDB repository has releases with complete data:")
        print("   https://github.com/andrealoddo/MP-IDB-The-Malaria-Parasite-Image-Database-for-Image-Processing-and-Analysis/releases")

        print("\n3. Try contacting dataset authors for complete annotations")

        print("\n4. Use current Falciparum data for life-stage classification as proof-of-concept")

if __name__ == "__main__":
    main()