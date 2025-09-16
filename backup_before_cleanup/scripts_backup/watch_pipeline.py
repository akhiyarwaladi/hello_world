#!/usr/bin/env python3
"""
Pipeline watcher - automatically runs next steps when preprocessing completes
"""

import os
import sys
import time
import subprocess
from pathlib import Path

def wait_for_preprocessing():
    """Wait for preprocessing to complete by checking for CSV file"""
    csv_file = Path("data/processed/processed_samples.csv")
    
    print("🔍 Watching for preprocessing completion...")
    print(f"Looking for: {csv_file}")
    
    while not csv_file.exists():
        # Check processed images count
        try:
            image_count = len(list(Path("data/processed/images").glob("*.jpg")))
            print(f"📊 Current processed images: {image_count}")
        except:
            image_count = 0
            print("📊 Processed directory not ready yet")
        
        print("⏳ Preprocessing still running, checking again in 30 seconds...")
        time.sleep(30)
    
    print("✅ Preprocessing completed! CSV file found.")
    return True

def run_remaining_pipeline():
    """Run the remaining pipeline steps"""
    
    scripts = [
        ("scripts/03_integrate_datasets.py", "Dataset Integration"),
        ("scripts/04_convert_to_yolo.py", "YOLO Format Conversion"), 
        ("scripts/05_augment_data.py", "Data Augmentation"),
        ("scripts/06_split_dataset.py", "Dataset Splitting")
    ]
    
    print("\n🚀 Starting automatic pipeline execution...")
    
    for script_path, description in scripts:
        print(f"\n{'='*60}")
        print(f"▶️  {description}")
        print(f"{'='*60}")
        
        try:
            result = subprocess.run([
                sys.executable, script_path
            ], check=True, text=True)
            
            print(f"✅ {description} completed successfully!")
            
        except subprocess.CalledProcessError as e:
            print(f"❌ {description} failed!")
            print(f"Return code: {e.returncode}")
            print("🛑 Stopping pipeline execution.")
            return False
        except Exception as e:
            print(f"❌ Unexpected error in {description}: {e}")
            print("🛑 Stopping pipeline execution.")
            return False
    
    print(f"\n{'='*60}")
    print("🎉 COMPLETE PIPELINE EXECUTION FINISHED!")
    print(f"{'='*60}")
    print("✨ All steps completed successfully:")
    print("  ✅ Data Download")
    print("  ✅ Data Preprocessing") 
    print("  ✅ Dataset Integration")
    print("  ✅ YOLO Format Conversion")
    print("  ✅ Data Augmentation")
    print("  ✅ Dataset Splitting")
    print("\n🏁 Pipeline ready for model training!")
    
    return True

def main():
    """Main pipeline watcher"""
    print("🎯 MALARIA DETECTION PIPELINE WATCHER")
    print("=" * 60)
    
    # Wait for preprocessing to complete
    if wait_for_preprocessing():
        # Run remaining pipeline
        success = run_remaining_pipeline()
        
        if success:
            print("\n🚀 Ready to start model training!")
        else:
            print("\n⚠️  Pipeline incomplete - manual intervention needed")
    else:
        print("❌ Error waiting for preprocessing")

if __name__ == "__main__":
    main()