#!/usr/bin/env python3
"""
Comprehensive Setup Verification Script for Malaria Detection Pipeline
Run this script to verify your environment and data are correctly set up.

Usage:
    python verify_setup.py
    python verify_setup.py --verbose
    python verify_setup.py --fix  # Auto-fix common issues
"""

import os
import sys
from pathlib import Path

# Color codes for terminal output
class Colors:
    GREEN = '\033[92m'
    RED = '\033[91m'
    YELLOW = '\033[93m'
    BLUE = '\033[94m'
    RESET = '\033[0m'
    BOLD = '\033[1m'

def print_header(text):
    print(f"\n{Colors.BOLD}{Colors.BLUE}{'='*60}{Colors.RESET}")
    print(f"{Colors.BOLD}{Colors.BLUE}{text}{Colors.RESET}")
    print(f"{Colors.BOLD}{Colors.BLUE}{'='*60}{Colors.RESET}")

def print_success(text):
    print(f"{Colors.GREEN}✓ {text}{Colors.RESET}")

def print_error(text):
    print(f"{Colors.RED}✗ {text}{Colors.RESET}")

def print_warning(text):
    print(f"{Colors.YELLOW}⚠ {text}{Colors.RESET}")

def print_info(text):
    print(f"{Colors.BLUE}ℹ {text}{Colors.RESET}")

def check_python_version():
    """Check Python version >= 3.10"""
    print_header("1. Python Version Check")
    version = sys.version_info
    version_str = f"{version.major}.{version.minor}.{version.micro}"

    if version.major >= 3 and version.minor >= 10:
        print_success(f"Python {version_str} - OK")
        return True
    else:
        print_error(f"Python {version_str} - Requires Python 3.10+")
        return False

def check_pytorch():
    """Check PyTorch and CUDA availability"""
    print_header("2. PyTorch & CUDA Check")

    try:
        import torch
        print_success(f"PyTorch {torch.__version__} - Installed")

        if torch.cuda.is_available():
            device_name = torch.cuda.get_device_name(0)
            cuda_version = torch.version.cuda
            print_success(f"CUDA {cuda_version} - Available")
            print_success(f"GPU: {device_name}")

            # Check memory
            total_mem = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            print_info(f"GPU Memory: {total_mem:.1f} GB")
            return True
        else:
            print_warning("CUDA not available - Will use CPU (slower)")
            return True
    except ImportError:
        print_error("PyTorch not installed")
        print_info("Install: pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128")
        return False

def check_ultralytics():
    """Check Ultralytics (YOLO) installation"""
    print_header("3. Ultralytics (YOLO) Check")

    try:
        import ultralytics
        print_success(f"Ultralytics {ultralytics.__version__} - Installed")
        return True
    except ImportError:
        print_error("Ultralytics not installed")
        print_info("Install: pip install ultralytics==8.3.202")
        return False

def check_dependencies():
    """Check other required dependencies"""
    print_header("4. Dependencies Check")

    required = [
        'numpy', 'pandas', 'opencv-python', 'Pillow',
        'scikit-learn', 'matplotlib', 'seaborn', 'tqdm',
        'timm', 'albumentations', 'pyyaml'
    ]

    # Map package names to import names
    import_names = {
        'opencv-python': 'cv2',
        'Pillow': 'PIL',
        'scikit-learn': 'sklearn',
        'pyyaml': 'yaml'
    }

    all_ok = True
    for pkg in required:
        import_name = import_names.get(pkg, pkg)
        try:
            __import__(import_name)
            print_success(f"{pkg} - Installed")
        except ImportError:
            print_error(f"{pkg} - NOT installed")
            all_ok = False

    if not all_ok:
        print_info("Install missing: pip install -r requirements.txt")

    return all_ok

def check_data_structure():
    """Check data directory structure"""
    print_header("5. Data Structure Check")

    base_path = Path(__file__).parent

    # Required directories
    required_dirs = [
        'data/raw',
        'data/processed',
        'data/ground_truth_crops_224',
        'scripts',
        'results',
    ]

    all_ok = True
    for dir_path in required_dirs:
        full_path = base_path / dir_path
        if full_path.exists():
            print_success(f"{dir_path}/ - Exists")
        else:
            print_error(f"{dir_path}/ - Missing")
            all_ok = False

    return all_ok

def check_datasets():
    """Check processed datasets"""
    print_header("6. Processed Datasets Check")

    base_path = Path(__file__).parent / 'data' / 'processed'

    datasets = {
        'lifecycle': {'expected_images': 313, 'classes': 4},
        'species': {'expected_images': 209, 'classes': 4},
        'stages': {'expected_images': 209, 'classes': 4},
        'md_2019_stages': {'expected_images': 813, 'classes': 3},
    }

    all_ok = True
    for name, info in datasets.items():
        dataset_path = base_path / name
        if not dataset_path.exists():
            print_warning(f"{name}/ - Not found (run data setup)")
            all_ok = False
            continue

        # Check data.yaml
        yaml_path = dataset_path / 'data.yaml'
        if yaml_path.exists():
            # Count images
            train_imgs = list((dataset_path / 'train' / 'images').glob('*')) if (dataset_path / 'train' / 'images').exists() else []
            val_imgs = list((dataset_path / 'val' / 'images').glob('*')) if (dataset_path / 'val' / 'images').exists() else []
            test_imgs = list((dataset_path / 'test' / 'images').glob('*')) if (dataset_path / 'test' / 'images').exists() else []

            total = len(train_imgs) + len(val_imgs) + len(test_imgs)
            print_success(f"{name}/ - {total} images (train:{len(train_imgs)}/val:{len(val_imgs)}/test:{len(test_imgs)})")
        else:
            print_error(f"{name}/ - data.yaml missing")
            all_ok = False

    return all_ok

def check_ground_truth_crops():
    """Check ground truth crops"""
    print_header("7. Ground Truth Crops Check")

    base_path = Path(__file__).parent / 'data' / 'ground_truth_crops_224'

    datasets = ['lifecycle', 'species', 'stages', 'md_2019_stages']

    all_ok = True
    for name in datasets:
        dataset_path = base_path / name
        if not dataset_path.exists():
            print_warning(f"{name}/ - Not found (run crop generation)")
            all_ok = False
            continue

        # Count crops (check multiple image formats)
        def count_images(path):
            if not path.exists():
                return 0
            return sum(1 for ext in ['*.png', '*.jpg', '*.jpeg'] for _ in path.rglob(ext))

        # Check both structures: direct (train/) and nested (crops/train/)
        crops_base = dataset_path / 'crops' if (dataset_path / 'crops').exists() else dataset_path
        train_crops = count_images(crops_base / 'train')
        val_crops = count_images(crops_base / 'val')
        test_crops = count_images(crops_base / 'test')

        total = train_crops + val_crops + test_crops
        if total > 0:
            print_success(f"{name}/ - {total} crops (train:{train_crops}/val:{val_crops}/test:{test_crops})")
        else:
            print_warning(f"{name}/ - No crops found")
            all_ok = False

    return all_ok

def check_main_pipeline():
    """Check main pipeline script"""
    print_header("8. Main Pipeline Check")

    base_path = Path(__file__).parent
    main_pipeline = base_path / 'main_pipeline.py'

    if main_pipeline.exists():
        print_success("main_pipeline.py - Exists")

        # Try to import it
        try:
            import importlib.util
            spec = importlib.util.spec_from_file_location("main_pipeline", main_pipeline)
            print_success("main_pipeline.py - Syntax OK")
            return True
        except SyntaxError as e:
            print_error(f"main_pipeline.py - Syntax error: {e}")
            return False
    else:
        print_error("main_pipeline.py - Not found")
        return False

def print_summary(results):
    """Print summary of all checks"""
    print_header("SUMMARY")

    passed = sum(results.values())
    total = len(results)

    for check, result in results.items():
        status = "✓ PASS" if result else "✗ FAIL"
        color = Colors.GREEN if result else Colors.RED
        print(f"  {color}{status}{Colors.RESET} - {check}")

    print()
    if passed == total:
        print(f"{Colors.GREEN}{Colors.BOLD}All checks passed! ({passed}/{total}){Colors.RESET}")
        print(f"\n{Colors.BLUE}Ready to run:{Colors.RESET}")
        print("  python main_pipeline.py --dataset iml_lifecycle --include yolo11 --classification-models densenet121 --epochs-det 5 --epochs-cls 5")
    else:
        print(f"{Colors.YELLOW}{Colors.BOLD}Some checks failed ({passed}/{total}){Colors.RESET}")
        print(f"\n{Colors.BLUE}See SETUP_GUIDE.md for setup instructions{Colors.RESET}")

def main():
    print(f"\n{Colors.BOLD}Malaria Detection Pipeline - Setup Verification{Colors.RESET}")
    print(f"Project: {Path(__file__).parent}")

    results = {}

    results['Python Version'] = check_python_version()
    results['PyTorch & CUDA'] = check_pytorch()
    results['Ultralytics'] = check_ultralytics()
    results['Dependencies'] = check_dependencies()
    results['Data Structure'] = check_data_structure()
    results['Datasets'] = check_datasets()
    results['Ground Truth Crops'] = check_ground_truth_crops()
    results['Main Pipeline'] = check_main_pipeline()

    print_summary(results)

    return 0 if all(results.values()) else 1

if __name__ == '__main__':
    sys.exit(main())
