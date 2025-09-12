#!/usr/bin/env python3
"""
Automated pipeline runner for malaria detection project
Runs the complete pipeline from preprocessing to training
"""

import os
import sys
import subprocess
import time
from pathlib import Path

def run_script(script_path, description):
    """Run a Python script and handle errors"""
    print(f"\n{'='*60}")
    print(f" {description}")
    print(f"{'='*60}")
    
    try:
        result = subprocess.run([
            sys.executable, script_path
        ], check=True, capture_output=True, text=True)
        
        print(f"✓ {description} completed successfully")
        if result.stdout:
            print("Output:", result.stdout[-500:])  # Last 500 chars
        return True
        
    except subprocess.CalledProcessError as e:
        print(f"✗ {description} failed with error:")
        print(f"Return code: {e.returncode}")
        if e.stdout:
            print("Stdout:", e.stdout[-500:])
        if e.stderr:
            print("Stderr:", e.stderr[-500:])
        return False
    except Exception as e:
        print(f"✗ Unexpected error in {description}: {e}")
        return False

def check_preprocessing_completed():
    """Check if preprocessing is completed"""
    processed_file = Path("data/processed/processed_samples.csv")
    return processed_file.exists()

def main():
    """Run the complete pipeline"""
    
    scripts = [
        ("scripts/02_preprocess_data.py", "Data Preprocessing"),
        ("scripts/03_integrate_datasets.py", "Dataset Integration"),
        ("scripts/04_convert_to_yolo.py", "YOLO Format Conversion"),
        ("scripts/05_augment_data.py", "Data Augmentation"),
        ("scripts/06_split_dataset.py", "Dataset Splitting")
    ]
    
    print("MALARIA DETECTION PIPELINE")
    print("="*60)
    
    # Check if preprocessing is already running
    if not check_preprocessing_completed():
        print("Waiting for preprocessing to complete...")
        while not check_preprocessing_completed():
            time.sleep(30)  # Check every 30 seconds
            print("Still waiting for preprocessing...")
    
    success_count = 0
    
    for script_path, description in scripts:
        if not Path(script_path).exists():
            print(f"✗ Script not found: {script_path}")
            continue
            
        if run_script(script_path, description):
            success_count += 1
        else:
            print(f"\n⚠️ Pipeline stopped at: {description}")
            print("Please fix the error and run again")
            break
    
    print(f"\n{'='*60}")
    print(f"PIPELINE SUMMARY")
    print(f"{'='*60}")
    print(f"Completed steps: {success_count}/{len(scripts)}")
    
    if success_count == len(scripts):
        print("🎉 Complete pipeline executed successfully!")
        print("\nNext steps:")
        print("1. Review the processed datasets")
        print("2. Start model training")
        print("3. Evaluate model performance")
    else:
        print("❌ Pipeline incomplete - please check errors above")

if __name__ == "__main__":
    main()