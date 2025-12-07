#!/usr/bin/env python3
"""
MASTER DATA PREPARATION PIPELINE
Complete automated setup from raw downloads to ready-for-training datasets.

This script automates ALL preprocessing steps:
1. Download raw datasets (IML, MP-IDB, MD-2019)
2. Setup detection datasets (YOLO format with 60/20/20 split)
3. Generate ground truth crops for classification (224x224)

Usage:
    python setup_all_data.py                    # Full automated setup
    python setup_all_data.py --quick            # Skip downloads if data exists
    python setup_all_data.py --datasets iml,md  # Specific datasets only

Author: Malaria Detection Team
Date: 2025-12-07
"""

import os
import sys
import subprocess
import argparse
import time
import stat
import shutil
import zipfile
import urllib.request
from pathlib import Path
from datetime import datetime

def remove_readonly(func, path, excinfo):
    """Handle readonly files on Windows (e.g., Git objects)"""
    try:
        os.chmod(path, stat.S_IWRITE)
        func(path)
    except Exception:
        pass  # If still fails, just continue

def download_and_extract_md2019():
    """Download and extract MD-2019 dataset from S3 (or fallback to cached zip)

    Priority:
    1. Download from URL (always get fresh data)
    2. Fallback to cached zip if download fails

    Returns:
        bool: True if successful, False otherwise
    """
    md2019_url = "https://prod-dcd-datasets-cache-zipfiles.s3.eu-west-1.amazonaws.com/5bf2kmwvfn-1.zip"
    download_dir = Path("data/download")
    zip_path = download_dir / "malaria_2019.zip"
    extract_to = Path("data/raw/md_2019")

    # Check if already extracted
    if extract_to.exists():
        return True

    # Create download directory
    download_dir.mkdir(parents=True, exist_ok=True)

    # PRIORITY 1: Try to download from URL first (always get fresh data)
    try:
        print_info(f"Downloading MD-2019 from S3 (~1.7 GB)...")
        print_info(f"  URL: {md2019_url}")
        print_info(f"  Saving to: {zip_path}")

        # Download from URL
        urllib.request.urlretrieve(md2019_url, zip_path)
        print_success(f"Downloaded: {zip_path}")

    except Exception as e:
        print_warning(f"Download from URL failed: {e}")

        # PRIORITY 2: Fallback to cached zip if exists
        if zip_path.exists():
            print_info(f"Falling back to cached zip: {zip_path}")
        else:
            print_error("No cached zip found and download failed!")
            return False

    # Extract the zip
    try:
        print_info(f"Extracting to: {extract_to}")
        extract_to.mkdir(parents=True, exist_ok=True)

        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(extract_to)

        print_success(f"Extracted MD-2019 to: {extract_to}")
        return True

    except Exception as e:
        print_error(f"Failed to extract MD-2019: {e}")
        return False

class Colors:
    """Terminal colors"""
    HEADER = '\033[95m'
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    END = '\033[0m'
    BOLD = '\033[1m'

def print_header(text):
    """Print section header"""
    print(f"\n{Colors.HEADER}{'='*80}{Colors.END}")
    print(f"{Colors.HEADER}{Colors.BOLD}{text.center(80)}{Colors.END}")
    print(f"{Colors.HEADER}{'='*80}{Colors.END}\n")

def print_success(text):
    """Print success message"""
    print(f"{Colors.GREEN}✓ {text}{Colors.END}")

def print_error(text):
    """Print error message"""
    print(f"{Colors.RED}✗ {text}{Colors.END}")

def print_info(text):
    """Print info message"""
    print(f"{Colors.CYAN}ℹ {text}{Colors.END}")

def print_warning(text):
    """Print warning message"""
    print(f"{Colors.YELLOW}⚠ {text}{Colors.END}")

def print_step(step_num, total_steps, description):
    """Print step progress"""
    print(f"\n{Colors.BOLD}[STEP {step_num}/{total_steps}] {description}{Colors.END}")

def run_command(cmd, description, critical=True):
    """Run command and handle errors"""
    print(f"\n{Colors.BLUE}► {description}{Colors.END}")
    print(f"  Command: {' '.join(cmd)}")

    try:
        result = subprocess.run(
            cmd,
            check=True,
            capture_output=True,
            text=True,
            encoding='utf-8',
            errors='replace'
        )

        if result.stdout:
            # Print last few lines of output
            lines = result.stdout.strip().split('\n')
            for line in lines[-5:]:
                print(f"  {line}")

        print_success(f"{description} - Complete")
        return True

    except subprocess.CalledProcessError as e:
        print_error(f"{description} - Failed")
        if e.stderr:
            print(f"{Colors.RED}Error: {e.stderr[:500]}{Colors.END}")

        if critical:
            print_error("Critical step failed! Stopping pipeline.")
            return False
        else:
            print_warning("Non-critical step failed, continuing...")
            return True

    except Exception as e:
        print_error(f"Unexpected error: {e}")
        return False if critical else True

def check_prerequisites():
    """Check if required tools are available"""
    print_header("CHECKING PREREQUISITES")

    checks = {
        "Python": ["python", "--version"],
        "PyTorch": ["python", "-c", "import torch; print(torch.__version__)"],
        "Ultralytics": ["python", "-c", "import ultralytics; print(ultralytics.__version__)"],
    }

    all_ok = True
    for name, cmd in checks.items():
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
            if result.returncode == 0:
                version = result.stdout.strip()
                print_success(f"{name}: {version}")
            else:
                print_error(f"{name}: Not found")
                all_ok = False
        except Exception as e:
            print_error(f"{name}: Error ({e})")
            all_ok = False

    # Check Kaggle (optional)
    try:
        result = subprocess.run(
            ["python", "-c", "from kaggle.api.kaggle_api_extended import KaggleApi; api = KaggleApi(); api.authenticate()"],
            capture_output=True,
            text=True,
            timeout=10
        )
        if result.returncode == 0:
            print_success("Kaggle API: Configured ✓")
        else:
            print_warning("Kaggle API: Not configured (MP-IDB download will fail)")
            print_info("Setup: https://www.kaggle.com/settings → Create API Token")
    except:
        print_warning("Kaggle: Not installed or not configured")

    return all_ok

def cleanup_existing_data(datasets, force=False, auto_confirm=False, clean_raw=False):
    """Clean up existing processed data before regeneration"""
    print_header("CLEANUP CHECK")

    paths_to_check = {
        "iml": [
            Path("data/processed/lifecycle"),
            Path("data/ground_truth_crops_224/lifecycle")
        ],
        "species": [
            Path("data/processed/species"),
            Path("data/ground_truth_crops_224/species")
        ],
        "stages": [
            Path("data/processed/stages"),
            Path("data/ground_truth_crops_224/stages")
        ],
        "md": [
            Path("data/processed/md_2019_stages"),
            Path("data/ground_truth_crops_224/md_2019_stages")
        ]
    }

    # Add raw data paths if clean_raw is True
    if clean_raw:
        raw_paths = {
            "iml": [Path("data/raw/malaria_lifecycle")],
            "species": [Path("data/raw/kaggle_dataset")],
            "stages": [Path("data/raw/kaggle_dataset")],  # Same as species
            "md": [Path("data/raw/md_2019")]
        }
        for ds in datasets:
            if ds in raw_paths:
                paths_to_check[ds].extend(raw_paths[ds])

        # Remove duplicates
        for ds in datasets:
            if ds in paths_to_check:
                paths_to_check[ds] = list(set(paths_to_check[ds]))

    existing_paths = []
    for ds in datasets:
        if ds in paths_to_check:
            for path in paths_to_check[ds]:
                if path.exists():
                    existing_paths.append(path)

    if not existing_paths:
        print_success("No existing processed data found, clean start")
        return True

    print_warning(f"Found {len(existing_paths)} existing processed folders:")
    for path in existing_paths:
        print(f"  • {path}")

    if force or auto_confirm:
        if force and clean_raw:
            print_info("COMPLETE CLEANUP mode enabled (including raw data), removing...")
        elif force:
            print_info("Force cleanup mode enabled, removing...")
        else:
            print_info("Auto-confirm mode enabled, removing existing data...")

        for path in existing_paths:
            try:
                shutil.rmtree(path, onerror=remove_readonly)
                print_success(f"Removed: {path}")
            except Exception as e:
                print_error(f"Failed to remove {path}: {e}")
        return True

    print_warning("\nRunning setup again will:")
    if clean_raw:
        print("  ⚠ RE-DOWNLOAD all raw data from scratch")
    print("  ⚠ Regenerate detection datasets (new random splits)")
    print("  ⚠ Regenerate ground truth crops (new random splits)")
    print("  ⚠ Results will be DIFFERENT from previous runs")

    response = input(f"\n{Colors.YELLOW}Clean up and regenerate? (y/n): {Colors.END}").strip().lower()
    if response == 'y':
        for path in existing_paths:
            try:
                shutil.rmtree(path, onerror=remove_readonly)
                print_success(f"Removed: {path}")
            except Exception as e:
                print_error(f"Failed to remove {path}: {e}")
        return True
    else:
        print_info("Keeping existing data, some steps may be skipped")
        return True

def step1_download_datasets(datasets, quick_mode=False, auto_confirm=False):
    """Step 1: Download raw datasets"""
    print_step(1, 3, "DOWNLOAD RAW DATASETS")

    # Check what's already downloaded
    raw_dir = Path("data/raw")
    existing = {
        "iml": (raw_dir / "malaria_lifecycle").exists(),
        "species": (raw_dir / "kaggle_dataset" / "MP-IDB-YOLO").exists(),
        "stages": (raw_dir / "kaggle_dataset" / "MP-IDB-YOLO").exists(),
        "md": (raw_dir / "md_2019").exists()
    }

    print_info(f"Existing raw data:")
    for ds, exists in existing.items():
        status = "✓" if exists else "✗"
        print(f"  [{status}] {ds.upper()}")

    if quick_mode and all(existing[ds] for ds in datasets):
        print_success("All required datasets already exist, skipping download")
        return True

    # Download IML + Kaggle MP-IDB if needed
    download_list = []
    if "iml" in datasets and not existing["iml"]:
        download_list.append("malaria_lifecycle")

    if ("species" in datasets or "stages" in datasets) and not existing["species"]:
        download_list.append("kaggle_mp_idb")

    if download_list:
        cmd = [
            "python", "scripts/data_setup/01_download_datasets.py",
            "--dataset", ",".join(download_list)
        ]

        if not run_command(cmd, f"Download {', '.join(download_list)}", critical=False):
            print_warning("Some downloads failed, but continuing...")

    # Download MD-2019 (auto-download and extract)
    if "md" in datasets and not existing["md"]:
        print_info("")
        print_info("MD-2019 dataset not found, attempting auto-download...")

        success = download_and_extract_md2019()
        if not success:
            print_warning("MD-2019 auto-download failed!")

            if auto_confirm:
                print_warning("Auto-confirm mode: Continuing without MD-2019")
            else:
                response = input(f"\n{Colors.YELLOW}Continue without MD-2019? (y/n): {Colors.END}").strip().lower()
                if response != 'y':
                    return False

    return True

def step2_setup_detection(datasets, train_ratio, val_ratio, test_ratio):
    """Step 2: Setup detection datasets (YOLO format)"""
    print_step(2, 3, "SETUP DETECTION DATASETS (YOLO FORMAT)")

    print_info(f"Split ratios: {train_ratio:.0%} / {val_ratio:.0%} / {test_ratio:.0%}")

    dataset_scripts = {
        "iml": ("IML Lifecycle", "scripts/data_setup/08_setup_lifecycle_for_pipeline.py"),
        "species": ("MP-IDB Species", "scripts/data_setup/07_setup_kaggle_species_for_pipeline.py"),
        "stages": ("MP-IDB Stages", "scripts/data_setup/09_setup_kaggle_stage_for_pipeline.py"),
        "md": ("MD-2019", "scripts/data_setup/10_setup_md2019_stage_for_pipeline.py")
    }

    for ds_key in datasets:
        if ds_key not in dataset_scripts:
            continue

        name, script = dataset_scripts[ds_key]

        cmd = [
            "python", script,
            "--train-ratio", str(train_ratio),
            "--val-ratio", str(val_ratio),
            "--test-ratio", str(test_ratio)
        ]

        if not run_command(cmd, f"Setup {name}", critical=True):
            return False

    return True

def step3_generate_crops(datasets, train_ratio, val_ratio, test_ratio):
    """Step 3: Generate ground truth crops"""
    print_step(3, 3, "GENERATE GROUND TRUTH CROPS (224x224)")

    print_info(f"Split ratios: {train_ratio:.0%} / {val_ratio:.0%} / {test_ratio:.0%}")

    crop_configs = {
        "iml": {
            "name": "IML Lifecycle",
            "dataset": "data/processed/lifecycle",
            "output": "data/ground_truth_crops_224/lifecycle",
            "type": "iml_lifecycle"
        },
        "species": {
            "name": "MP-IDB Species",
            "dataset": "data/processed/species",
            "output": "data/ground_truth_crops_224/species",
            "type": "mp_idb_species"
        },
        "stages": {
            "name": "MP-IDB Stages",
            "dataset": "data/processed/stages",
            "output": "data/ground_truth_crops_224/stages",
            "type": "mp_idb_stages"
        },
        "md": {
            "name": "MD-2019",
            "dataset": "data/processed/md_2019_stages",
            "output": "data/ground_truth_crops_224/md_2019_stages",
            "type": "md_2019_stages"
        }
    }

    for ds_key in datasets:
        if ds_key not in crop_configs:
            continue

        config = crop_configs[ds_key]

        cmd = [
            "python", "scripts/training/generate_ground_truth_crops.py",
            "--dataset", config["dataset"],
            "--output", config["output"],
            "--type", config["type"],
            "--train-ratio", str(train_ratio),
            "--val-ratio", str(val_ratio),
            "--test-ratio", str(test_ratio)
        ]

        if not run_command(cmd, f"Generate crops for {config['name']}", critical=True):
            return False

    return True

def verify_setup(datasets):
    """Verify all data is ready"""
    print_header("VERIFYING SETUP")

    checks = {
        "iml": [
            ("Detection dataset", Path("data/processed/lifecycle/data.yaml")),
            ("Ground truth crops", Path("data/ground_truth_crops_224/lifecycle/crops"))
        ],
        "species": [
            ("Detection dataset", Path("data/processed/species/data.yaml")),
            ("Ground truth crops", Path("data/ground_truth_crops_224/species/crops"))
        ],
        "stages": [
            ("Detection dataset", Path("data/processed/stages/data.yaml")),
            ("Ground truth crops", Path("data/ground_truth_crops_224/stages/crops"))
        ],
        "md": [
            ("Detection dataset", Path("data/processed/md_2019_stages/data.yaml")),
            ("Ground truth crops", Path("data/ground_truth_crops_224/md_2019_stages/crops"))
        ]
    }

    all_ok = True
    for ds_key in datasets:
        if ds_key not in checks:
            continue

        print(f"\n{Colors.BOLD}{ds_key.upper()}:{Colors.END}")
        for name, path in checks[ds_key]:
            if path.exists():
                print_success(f"{name}: {path}")
            else:
                print_error(f"{name}: Missing - {path}")
                all_ok = False

    return all_ok

def print_summary(datasets, elapsed_time):
    """Print completion summary"""
    print_header("SETUP COMPLETE!")

    print(f"{Colors.GREEN}{Colors.BOLD}✓ All preprocessing complete!{Colors.END}\n")

    print(f"{Colors.BOLD}Datasets ready:{Colors.END}")
    for ds in datasets:
        print(f"  ✓ {ds.upper()}")

    print(f"\n{Colors.BOLD}Total time:{Colors.END} {elapsed_time/60:.1f} minutes")

    print(f"\n{Colors.BOLD}Directory structure:{Colors.END}")
    print("  data/")
    print("  ├── raw/                      # Downloaded raw data")
    print("  ├── processed/                # YOLO detection datasets (60/20/20)")
    print("  └── ground_truth_crops_224/   # Classification crops (224x224)")

    print(f"\n{Colors.BOLD}Next steps:{Colors.END}")
    print("  1. Quick test (5 minutes):")
    print("     python main_pipeline.py --dataset iml_lifecycle --include yolo11 \\")
    print("            --classification-models densenet121 --epochs-det 5 --epochs-cls 5")
    print("\n  2. Full experiment:")
    print("     python main_pipeline.py --dataset all")

def main():
    """Main pipeline"""
    parser = argparse.ArgumentParser(
        description="Master data preparation pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    parser.add_argument(
        "--datasets",
        nargs="+",
        choices=["iml", "species", "stages", "md", "all"],
        default=["all"],
        help="Datasets to setup (default: all)"
    )

    parser.add_argument(
        "--quick",
        action="store_true",
        help="Skip downloads if data already exists"
    )

    parser.add_argument(
        "--yes", "-y",
        action="store_true",
        help="Auto-confirm all prompts (for automation/background mode)"
    )

    parser.add_argument(
        "--train-ratio",
        type=float,
        default=0.60,
        help="Training set ratio (default: 0.60)"
    )

    parser.add_argument(
        "--val-ratio",
        type=float,
        default=0.20,
        help="Validation set ratio (default: 0.20)"
    )

    parser.add_argument(
        "--test-ratio",
        type=float,
        default=0.20,
        help="Test set ratio (default: 0.20)"
    )

    args = parser.parse_args()

    # Resolve "all" datasets
    if "all" in args.datasets:
        datasets = ["iml", "species", "stages", "md"]
    else:
        datasets = args.datasets

    # Validate split ratios
    total_ratio = args.train_ratio + args.val_ratio + args.test_ratio
    if abs(total_ratio - 1.0) > 0.001:
        print_error(f"Split ratios must sum to 1.0 (current: {total_ratio})")
        return 1

    # Print welcome
    print_header("MASTER DATA PREPARATION PIPELINE")
    print(f"Datasets: {', '.join([d.upper() for d in datasets])}")
    print(f"Split ratios: {args.train_ratio:.0%} / {args.val_ratio:.0%} / {args.test_ratio:.0%}")
    print(f"Quick mode: {'Yes' if args.quick else 'No'}")
    print(f"Auto-confirm: {'Yes' if args.yes else 'No'}")
    print(f"{Colors.YELLOW}Clean mode: ALWAYS (raw + processed data){Colors.END}")

    if not args.quick and not args.yes:
        print(f"\n{Colors.YELLOW}This will:{Colors.END}")
        print("  1. Download raw datasets (~2 GB)")
        print("  2. Setup YOLO detection datasets")
        print("  3. Generate classification crops")
        print(f"\n{Colors.YELLOW}Estimated time: 20-30 minutes{Colors.END}")

        response = input(f"\n{Colors.BOLD}Continue? (y/n): {Colors.END}").strip().lower()
        if response != 'y':
            print("Setup cancelled")
            return 0
    elif args.yes:
        print_success("Auto-confirm mode: Proceeding without prompts")

    start_time = time.time()

    # Check prerequisites
    if not check_prerequisites():
        print_error("\nPrerequisite check failed!")
        print_info("Install missing packages: pip install -r requirements.txt")
        return 1

    # Cleanup check (always clean everything: raw + processed)
    if not cleanup_existing_data(datasets, force=True, auto_confirm=args.yes, clean_raw=True):
        print_info("Setup cancelled by user")
        return 0

    # Step 1: Download
    if not step1_download_datasets(datasets, args.quick, auto_confirm=args.yes):
        print_error("\nDownload step failed!")
        return 1

    # Step 2: Setup detection
    if not step2_setup_detection(datasets, args.train_ratio, args.val_ratio, args.test_ratio):
        print_error("\nDetection setup failed!")
        return 1

    # Step 3: Generate crops
    if not step3_generate_crops(datasets, args.train_ratio, args.val_ratio, args.test_ratio):
        print_error("\nCrop generation failed!")
        return 1

    # Verify
    if not verify_setup(datasets):
        print_warning("\nSome verification checks failed!")
        print_info("Review the errors above and fix manually if needed")

    # Summary
    elapsed = time.time() - start_time
    print_summary(datasets, elapsed)

    return 0

if __name__ == "__main__":
    sys.exit(main())
