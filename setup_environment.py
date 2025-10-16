#!/usr/bin/env python
"""
Setup script for Malaria Detection Pipeline Environment
Installs all required packages in conda base environment
"""

import subprocess
import sys
import os

def run_command(cmd, description):
    """Run command and print status"""
    print(f"\n{'='*60}")
    print(f"{description}")
    print(f"{'='*60}")
    print(f"Command: {cmd}")

    try:
        result = subprocess.run(
            cmd,
            shell=True,
            check=False,
            capture_output=True,
            text=True,
            encoding='utf-8',
            errors='replace'
        )

        if result.stdout:
            print(result.stdout)
        if result.stderr:
            print("STDERR:", result.stderr)

        if result.returncode == 0:
            print(f"✓ SUCCESS: {description}")
            return True
        else:
            print(f"✗ FAILED: {description} (return code: {result.returncode})")
            return False
    except Exception as e:
        print(f"✗ EXCEPTION: {e}")
        return False

def main():
    print("""
    ========================================
    Malaria Detection Pipeline Environment Setup
    ========================================

    This script will install:
    - PyTorch 2.8.0 with CUDA 12.8
    - Ultralytics (YOLO 10, 11, 12)
    - Computer vision packages (OpenCV, scikit-image, albumentations)
    - Data science packages (pandas, numpy, scipy, scikit-learn)
    - All other dependencies from requirements.txt
    """)

    # Step 1: Install PyTorch with CUDA
    print("\n[STEP 1/4] Installing PyTorch with CUDA 12.8...")
    pytorch_success = run_command(
        "pip install torch==2.8.0 torchvision==0.23.0 --index-url https://download.pytorch.org/whl/cu128",
        "PyTorch + CUDA 12.8 Installation"
    )

    if not pytorch_success:
        print("\n⚠ WARNING: PyTorch installation may have issues. Continuing...")

    # Step 2: Install Ultralytics (YOLO)
    print("\n[STEP 2/4] Installing Ultralytics (YOLO)...")
    yolo_success = run_command(
        "pip install ultralytics==8.3.202",
        "Ultralytics YOLO Installation"
    )

    # Step 3: Install remaining dependencies
    print("\n[STEP 3/4] Installing remaining dependencies from requirements.txt...")
    deps_success = run_command(
        "pip install -r requirements.txt",
        "Dependencies Installation"
    )

    # Step 4: Verify installation
    print("\n[STEP 4/4] Verifying installation...")

    verification_tests = [
        ("import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA Available: {torch.cuda.is_available()}'); print(f'CUDA Version: {torch.version.cuda if torch.cuda.is_available() else \"N/A\"}')", "PyTorch"),
        ("import torchvision; print(f'TorchVision: {torchvision.__version__}')", "TorchVision"),
        ("import ultralytics; print(f'Ultralytics: {ultralytics.__version__}')", "Ultralytics"),
        ("import cv2; print(f'OpenCV: {cv2.__version__}')", "OpenCV"),
        ("import pandas; print(f'Pandas: {pandas.__version__}')", "Pandas"),
        ("import numpy; print(f'NumPy: {numpy.__version__}')", "NumPy"),
        ("import albumentations; print(f'Albumentations: {albumentations.__version__}')", "Albumentations"),
        ("import sklearn; print(f'Scikit-learn: {sklearn.__version__}')", "Scikit-learn"),
    ]

    all_verified = True
    for test_code, package_name in verification_tests:
        try:
            result = subprocess.run(
                [sys.executable, "-c", test_code],
                capture_output=True,
                text=True,
                check=True
            )
            print(f"✓ {package_name}: {result.stdout.strip()}")
        except subprocess.CalledProcessError:
            print(f"✗ {package_name}: NOT INSTALLED")
            all_verified = False
        except Exception as e:
            print(f"✗ {package_name}: ERROR - {e}")
            all_verified = False

    # Final summary
    print("\n" + "="*60)
    if all_verified:
        print("✓ SETUP COMPLETE - All packages verified!")
    else:
        print("⚠ SETUP COMPLETE - Some packages may need manual installation")
    print("="*60)

    print("""
    You can now run the main pipeline:

    Full experiment (all datasets, all models):
      python main_pipeline.py

    Quick test (single dataset, single models):
      python main_pipeline.py --dataset iml_lifecycle --include yolo11 --classification-models densenet121 --epochs-det 5 --epochs-cls 5 --no-zip

    Single dataset full experiment:
      python main_pipeline.py --dataset iml_lifecycle --epochs-det 100 --epochs-cls 75
    """)

if __name__ == "__main__":
    main()
