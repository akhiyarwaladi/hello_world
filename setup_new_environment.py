#!/usr/bin/env python
"""
Complete Environment Setup Script for Malaria Detection Pipeline
================================================================

This script creates a NEW conda environment and installs all dependencies
for transferable, reproducible deployment on any Windows PC with NVIDIA GPU.

Requirements:
- Anaconda or Miniconda installed
- NVIDIA GPU with CUDA support (RTX 2060 or newer recommended)
- Windows 10/11 64-bit
- ~10GB disk space for environment

Usage:
------
1. Open Anaconda Prompt
2. cd to project directory
3. python setup_new_environment.py

Or with custom environment name:
    python setup_new_environment.py --name my_malaria_env

Created: 2025-12-07
Author: Malaria Detection Pipeline Team
"""

import subprocess
import sys
import os
import argparse
import platform
import shutil
from pathlib import Path

# ============================================================================
# CONFIGURATION
# ============================================================================
DEFAULT_ENV_NAME = "malaria_detection"
PYTHON_VERSION = "3.11"  # Stable version with good PyTorch support
PYTORCH_VERSION = "2.8.0"
TORCHVISION_VERSION = "0.23.0"
CUDA_VERSION = "cu128"  # CUDA 12.8
ULTRALYTICS_VERSION = "8.3.202"


def print_header(text, char="="):
    """Print formatted header"""
    print(f"\n{char * 60}")
    print(f" {text}")
    print(f"{char * 60}")


def print_step(step_num, total_steps, description):
    """Print step indicator"""
    print(f"\n[STEP {step_num}/{total_steps}] {description}")
    print("-" * 50)


def run_command(cmd, description=None, shell=True, capture_output=False):
    """Run command with error handling"""
    if description:
        print(f"  {description}")
    print(f"  Command: {cmd}")

    try:
        if capture_output:
            result = subprocess.run(
                cmd, shell=shell, capture_output=True, text=True,
                encoding='utf-8', errors='replace'
            )
            return result.returncode == 0, result.stdout, result.stderr
        else:
            result = subprocess.run(cmd, shell=shell, check=False)
            return result.returncode == 0, None, None
    except Exception as e:
        print(f"  [ERROR] {e}")
        return False, None, str(e)


def check_conda():
    """Check if conda is available"""
    success, stdout, _ = run_command("conda --version", capture_output=True)
    if success and stdout:
        print(f"  [OK] Conda found: {stdout.strip()}")
        return True
    print("  [ERROR] Conda not found! Please install Anaconda or Miniconda.")
    return False


def check_nvidia_gpu():
    """Check for NVIDIA GPU"""
    success, stdout, _ = run_command("nvidia-smi --query-gpu=name --format=csv,noheader",
                                     capture_output=True)
    if success and stdout:
        gpu_name = stdout.strip().split('\n')[0]
        print(f"  [OK] NVIDIA GPU found: {gpu_name}")
        return True
    print("  [WARNING] No NVIDIA GPU detected. CUDA features will not work.")
    return False


def check_existing_env(env_name):
    """Check if environment already exists"""
    success, stdout, _ = run_command("conda env list", capture_output=True)
    if success and stdout:
        if env_name in stdout:
            return True
    return False


def create_conda_env(env_name, python_version):
    """Create new conda environment"""
    print(f"  Creating conda environment: {env_name} (Python {python_version})")

    cmd = f"conda create -n {env_name} python={python_version} -y"
    success, _, _ = run_command(cmd)
    return success


def remove_conda_env(env_name):
    """Remove existing conda environment"""
    print(f"  Removing existing environment: {env_name}")
    cmd = f"conda env remove -n {env_name} -y"
    success, _, _ = run_command(cmd)
    return success


def install_pytorch(env_name):
    """Install PyTorch with CUDA support"""
    print(f"  Installing PyTorch {PYTORCH_VERSION} with CUDA {CUDA_VERSION}")

    # Use pip inside conda environment
    cmd = (
        f"conda run -n {env_name} pip install "
        f"torch=={PYTORCH_VERSION} torchvision=={TORCHVISION_VERSION} "
        f"--index-url https://download.pytorch.org/whl/{CUDA_VERSION}"
    )
    success, _, _ = run_command(cmd)
    return success


def install_ultralytics(env_name):
    """Install Ultralytics YOLO"""
    print(f"  Installing Ultralytics {ULTRALYTICS_VERSION}")

    cmd = f"conda run -n {env_name} pip install ultralytics=={ULTRALYTICS_VERSION}"
    success, _, _ = run_command(cmd)
    return success


def install_requirements(env_name, requirements_file="requirements.txt"):
    """Install remaining dependencies from requirements.txt"""
    print(f"  Installing dependencies from {requirements_file}")

    if not Path(requirements_file).exists():
        print(f"  [ERROR] {requirements_file} not found!")
        return False

    cmd = f"conda run -n {env_name} pip install -r {requirements_file}"
    success, _, _ = run_command(cmd)
    return success


def install_core_packages(env_name):
    """Install core packages individually (fallback if requirements.txt fails)"""
    packages = [
        # Computer Vision
        "opencv-python>=4.9.0",
        "albumentations>=2.0.0",
        "pillow>=10.0.0",
        "scikit-image>=0.22.0",

        # Data Science
        "pandas>=2.0.0",
        "numpy>=1.26.0",
        "scipy>=1.12.0",
        "scikit-learn>=1.4.0",

        # Visualization
        "matplotlib>=3.8.0",
        "seaborn>=0.13.0",

        # Excel handling
        "openpyxl>=3.1.0",
        "xlsxwriter>=3.1.0",

        # Utilities
        "tqdm>=4.66.0",
        "pyyaml>=6.0.0",
        "psutil>=5.9.0",
        "requests>=2.31.0",

        # Data loading
        "kaggle>=1.6.0",
        "gdown>=5.0.0",
    ]

    for package in packages:
        print(f"  Installing: {package}")
        cmd = f"conda run -n {env_name} pip install \"{package}\""
        run_command(cmd)

    return True


def verify_installation(env_name):
    """Verify all critical packages are installed"""
    print("\n  Verifying installation...")

    verification_tests = [
        ("import torch; print(f'PyTorch {torch.__version__}')", "PyTorch"),
        ("import torch; print(f'CUDA: {torch.cuda.is_available()}')", "CUDA"),
        ("import torchvision; print(f'TorchVision {torchvision.__version__}')", "TorchVision"),
        ("import ultralytics; print(f'Ultralytics {ultralytics.__version__}')", "Ultralytics"),
        ("import cv2; print(f'OpenCV {cv2.__version__}')", "OpenCV"),
        ("import pandas; print(f'Pandas {pandas.__version__}')", "Pandas"),
        ("import numpy; print(f'NumPy {numpy.__version__}')", "NumPy"),
        ("import sklearn; print(f'Scikit-learn {sklearn.__version__}')", "Scikit-learn"),
        ("import albumentations; print(f'Albumentations {albumentations.__version__}')", "Albumentations"),
        ("import matplotlib; print(f'Matplotlib {matplotlib.__version__}')", "Matplotlib"),
        ("import openpyxl; print(f'Openpyxl {openpyxl.__version__}')", "Openpyxl"),
    ]

    all_passed = True
    results = []

    for test_code, package_name in verification_tests:
        cmd = f'conda run -n {env_name} python -c "{test_code}"'
        success, stdout, _ = run_command(cmd, capture_output=True)

        if success and stdout:
            status = f"[OK] {stdout.strip()}"
            results.append((package_name, True, stdout.strip()))
        else:
            status = f"[FAIL] Not installed or error"
            results.append((package_name, False, "Failed"))
            all_passed = False

        print(f"    {package_name}: {status}")

    return all_passed, results


def fix_data_yaml_paths(env_name):
    """Fix data.yaml paths for Windows compatibility"""
    print("  Fixing data.yaml paths...")

    if Path("fix_data_yaml_paths.py").exists():
        cmd = f"conda run -n {env_name} python fix_data_yaml_paths.py"
        success, _, _ = run_command(cmd)
        return success
    else:
        print("  [INFO] fix_data_yaml_paths.py not found, skipping...")
        return True


def create_activation_script(env_name):
    """Create batch script for easy environment activation"""
    script_content = f"""@echo off
REM Activate Malaria Detection Environment
REM Usage: Double-click this file or run from cmd

echo ============================================
echo Activating Malaria Detection Environment
echo ============================================

call conda activate {env_name}

echo.
echo Environment activated: {env_name}
echo.
echo Quick commands:
echo   python main_pipeline.py --help    (show pipeline options)
echo   python main_pipeline.py           (run full pipeline)
echo.

cmd /k
"""

    script_path = Path("activate_env.bat")
    with open(script_path, 'w') as f:
        f.write(script_content)

    print(f"  Created: {script_path}")
    return True


def main():
    parser = argparse.ArgumentParser(
        description="Setup complete environment for Malaria Detection Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python setup_new_environment.py
  python setup_new_environment.py --name my_malaria_env
  python setup_new_environment.py --force
        """
    )
    parser.add_argument("--name", default=DEFAULT_ENV_NAME,
                        help=f"Environment name (default: {DEFAULT_ENV_NAME})")
    parser.add_argument("--force", action="store_true",
                        help="Remove existing environment if exists")
    parser.add_argument("--skip-verify", action="store_true",
                        help="Skip verification step")
    args = parser.parse_args()

    env_name = args.name
    total_steps = 8

    # ========================================================================
    # HEADER
    # ========================================================================
    print_header("MALARIA DETECTION PIPELINE - ENVIRONMENT SETUP")
    print(f"""
    This script will:
    1. Check system requirements (Conda, NVIDIA GPU)
    2. Create conda environment: {env_name}
    3. Install PyTorch {PYTORCH_VERSION} with CUDA {CUDA_VERSION}
    4. Install Ultralytics (YOLO) {ULTRALYTICS_VERSION}
    5. Install all dependencies from requirements.txt
    6. Verify installation
    7. Fix data paths for Windows
    8. Create activation script

    Estimated time: 10-15 minutes
    Required disk space: ~10GB
    """)

    input("Press Enter to continue or Ctrl+C to cancel...")

    # ========================================================================
    # STEP 1: Check System Requirements
    # ========================================================================
    print_step(1, total_steps, "Checking System Requirements")

    print(f"  System: {platform.system()} {platform.release()}")
    print(f"  Python: {sys.version}")

    if not check_conda():
        print("\n[FATAL] Conda is required. Please install Anaconda or Miniconda.")
        print("Download: https://www.anaconda.com/download")
        sys.exit(1)

    has_gpu = check_nvidia_gpu()

    # ========================================================================
    # STEP 2: Handle Existing Environment
    # ========================================================================
    print_step(2, total_steps, "Preparing Environment")

    if check_existing_env(env_name):
        if args.force:
            print(f"  Environment '{env_name}' exists. Removing (--force)...")
            if not remove_conda_env(env_name):
                print("[ERROR] Failed to remove existing environment")
                sys.exit(1)
        else:
            print(f"  [WARNING] Environment '{env_name}' already exists!")
            response = input(f"  Remove and recreate? (y/N): ").strip().lower()
            if response == 'y':
                if not remove_conda_env(env_name):
                    print("[ERROR] Failed to remove existing environment")
                    sys.exit(1)
            else:
                print("  Skipping environment creation. Using existing environment.")
                # Skip to installation

    # ========================================================================
    # STEP 3: Create Conda Environment
    # ========================================================================
    print_step(3, total_steps, f"Creating Conda Environment ({env_name})")

    if not check_existing_env(env_name):
        if not create_conda_env(env_name, PYTHON_VERSION):
            print("[ERROR] Failed to create conda environment")
            sys.exit(1)
        print(f"  [OK] Environment '{env_name}' created successfully")
    else:
        print(f"  [OK] Using existing environment '{env_name}'")

    # ========================================================================
    # STEP 4: Install PyTorch
    # ========================================================================
    print_step(4, total_steps, "Installing PyTorch with CUDA")

    if not install_pytorch(env_name):
        print("[WARNING] PyTorch installation may have issues. Continuing...")

    # ========================================================================
    # STEP 5: Install Ultralytics
    # ========================================================================
    print_step(5, total_steps, "Installing Ultralytics (YOLO)")

    if not install_ultralytics(env_name):
        print("[WARNING] Ultralytics installation may have issues. Continuing...")

    # ========================================================================
    # STEP 6: Install Dependencies
    # ========================================================================
    print_step(6, total_steps, "Installing Dependencies")

    if Path("requirements.txt").exists():
        if not install_requirements(env_name):
            print("  [WARNING] Some dependencies may have failed. Installing core packages...")
            install_core_packages(env_name)
    else:
        print("  [WARNING] requirements.txt not found. Installing core packages...")
        install_core_packages(env_name)

    # ========================================================================
    # STEP 7: Verify Installation
    # ========================================================================
    if not args.skip_verify:
        print_step(7, total_steps, "Verifying Installation")
        all_passed, results = verify_installation(env_name)

        if not all_passed:
            print("\n  [WARNING] Some packages failed verification!")
            print("  The pipeline may still work. Try running a quick test.")
    else:
        print_step(7, total_steps, "Verification (SKIPPED)")
        all_passed = True

    # ========================================================================
    # STEP 8: Post-Installation Setup
    # ========================================================================
    print_step(8, total_steps, "Post-Installation Setup")

    fix_data_yaml_paths(env_name)
    create_activation_script(env_name)

    # ========================================================================
    # FINAL SUMMARY
    # ========================================================================
    print_header("SETUP COMPLETE!", "=")

    print(f"""
    Environment Name: {env_name}
    Python Version: {PYTHON_VERSION}
    PyTorch Version: {PYTORCH_VERSION}
    CUDA Support: {"Yes" if has_gpu else "No (CPU only)"}

    TO ACTIVATE THE ENVIRONMENT:
    ============================
    Option 1: conda activate {env_name}
    Option 2: Double-click activate_env.bat

    QUICK TEST (5 minutes):
    =======================
    conda activate {env_name}
    python main_pipeline.py --dataset iml_lifecycle --include yolo11 --classification-models densenet121 --epochs-det 5 --epochs-cls 5

    FULL PIPELINE:
    ==============
    conda activate {env_name}
    python main_pipeline.py

    TRANSFER TO ANOTHER PC:
    =======================
    1. Copy the entire project folder
    2. Run: python setup_new_environment.py
    3. Activate environment and run pipeline
    """)


if __name__ == "__main__":
    main()
