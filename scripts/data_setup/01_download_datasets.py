#!/usr/bin/env python3
"""
Script to download all malaria datasets from various sources
Author: Malaria Detection Team
Date: 2024
"""

import os
import sys
import json
import yaml
import zipfile
import tarfile
import shutil
import requests
import subprocess
from pathlib import Path
from typing import Dict, List, Optional
from tqdm import tqdm
import gdown
from kaggle.api.kaggle_api_extended import KaggleApi


class MalariaDatasetDownloader:
    """Download manager for multiple malaria datasets"""
    
    def __init__(self, config_path: str = "config/dataset_config.yaml"):
        """Initialize downloader with configuration"""
        self.config = self._load_config(config_path)
        self.base_dir = Path("data/raw")
        self.base_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize Kaggle API
        try:
            self.kaggle_api = KaggleApi()
            self.kaggle_api.authenticate()
        except:
            print("Warning: Kaggle API not configured. Some datasets may not download.")
            self.kaggle_api = None
    
    def _load_config(self, config_path: str) -> Dict:
        """Load dataset configuration from YAML"""
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
    
    def download_with_progress(self, url: str, destination: Path, chunk_size: int = 8192):
        """Download file with progress bar"""
        response = requests.get(url, stream=True)
        total_size = int(response.headers.get('content-length', 0))
        
        with open(destination, 'wb') as f:
            with tqdm(total=total_size, unit='B', unit_scale=True, desc=destination.name) as pbar:
                for chunk in response.iter_content(chunk_size=chunk_size):
                    if chunk:
                        f.write(chunk)
                        pbar.update(len(chunk))
    
    def extract_archive(self, archive_path: Path, extract_to: Path):
        """Extract zip or tar archives"""
        print(f"Extracting {archive_path.name}...")
        
        if archive_path.suffix == '.zip':
            with zipfile.ZipFile(archive_path, 'r') as zip_ref:
                zip_ref.extractall(extract_to)
        elif archive_path.suffix in ['.tar', '.gz', '.tgz']:
            with tarfile.open(archive_path, 'r:*') as tar_ref:
                tar_ref.extractall(extract_to)
        else:
            print(f"Unknown archive format: {archive_path.suffix}")
    
    def download_nih_cell_images(self):
        """Download NIH malaria cell images dataset"""
        print("\n" + "="*50)
        print("Downloading NIH Cell Images Dataset...")
        print("="*50)
        
        dataset_dir = self.base_dir / "nih_cell"
        dataset_dir.mkdir(exist_ok=True)
        
        url = "https://data.lhncbc.nlm.nih.gov/public/Malaria/cell_images.zip"
        zip_path = dataset_dir / "cell_images.zip"
        
        if not zip_path.exists():
            self.download_with_progress(url, zip_path)
            self.extract_archive(zip_path, dataset_dir)
            print(f"✓ NIH Cell Images downloaded to {dataset_dir}")
        else:
            print(f"✓ NIH Cell Images already exists at {dataset_dir}")
    
    def download_nih_thick_smears(self):
        """Download NIH thick blood smear datasets"""
        print("\n" + "="*50)
        print("Downloading NIH Thick Smear Datasets...")
        print("="*50)
        
        datasets = {
            "thick_pf": {
                "url": "https://data.lhncbc.nlm.nih.gov/public/Malaria/Thick_Smears_150/",
                "name": "P. falciparum thick smears"
            },
            "thick_pv": {
                "url": "https://data.lhncbc.nlm.nih.gov/public/Malaria/NIH-NLM-ThickBloodSmearsPV/NIH-NLM-ThickBloodSmearsPV.zip",
                "name": "P. vivax thick smears"
            },
            "thick_uninfected": {
                "url": "https://data.lhncbc.nlm.nih.gov/public/Malaria/NIH-NLM-ThickBloodSmearsU/NIH-NLM-ThickBloodSmearsU.zip",
                "name": "Uninfected thick smears"
            }
        }
        
        for key, info in datasets.items():
            dataset_dir = self.base_dir / f"nih_{key}"
            dataset_dir.mkdir(exist_ok=True)
            
            if info["url"].endswith(".zip"):
                zip_path = dataset_dir / f"{key}.zip"
                if not zip_path.exists():
                    print(f"Downloading {info['name']}...")
                    self.download_with_progress(info["url"], zip_path)
                    self.extract_archive(zip_path, dataset_dir)
                    print(f"✓ {info['name']} downloaded")
            else:
                print(f"ℹ {info['name']} requires manual download from: {info['url']}")
    
    def download_mp_idb(self):
        """Clone MP-IDB dataset from GitHub"""
        print("\n" + "="*50)
        print("Downloading MP-IDB Dataset...")
        print("="*50)
        
        dataset_dir = self.base_dir / "mp_idb"
        
        if not dataset_dir.exists():
            repo_url = "https://github.com/andrealoddo/MP-IDB-The-Malaria-Parasite-Image-Database-for-Image-Processing-and-Analysis.git"
            subprocess.run(["git", "clone", repo_url, str(dataset_dir)])
            print(f"✓ MP-IDB cloned to {dataset_dir}")
        else:
            print(f"✓ MP-IDB already exists at {dataset_dir}")
    
    def download_bbbc041(self):
        """Download BBBC041 dataset"""
        print("\n" + "="*50)
        print("Downloading BBBC041 Dataset...")
        print("="*50)
        
        dataset_dir = self.base_dir / "bbbc041"
        dataset_dir.mkdir(exist_ok=True)
        
        # Direct download URL for BBBC041
        base_url = "https://data.broadinstitute.org/bbbc/BBBC041/"
        
        print(f"ℹ BBBC041 requires manual download from: {base_url}")
        print("  Please download the following files:")
        print("  - malaria.zip (main dataset)")
        print("  - malaria_labels.csv (annotations)")
        print(f"  Place them in: {dataset_dir}")
    
    def download_kaggle_dataset(self):
        """Download dataset from Kaggle"""
        print("\n" + "="*50)
        print("Downloading Kaggle NIH Dataset...")
        print("="*50)
        
        if self.kaggle_api is None:
            print("⚠ Kaggle API not configured. Skipping Kaggle datasets.")
            return
        
        dataset_dir = self.base_dir / "kaggle_nih"
        dataset_dir.mkdir(exist_ok=True)
        
        dataset_name = "iarunava/cell-images-for-detecting-malaria"
        
        try:
            self.kaggle_api.dataset_download_files(
                dataset_name,
                path=str(dataset_dir),
                unzip=True
            )
            print(f"✓ Kaggle dataset downloaded to {dataset_dir}")
        except Exception as e:
            print(f"⚠ Failed to download Kaggle dataset: {e}")
    
    def download_plasmoID(self):
        """Information for PlasmoID dataset"""
        print("\n" + "="*50)
        print("PlasmoID Dataset Information...")
        print("="*50)
        
        dataset_dir = self.base_dir / "plasmoID"
        dataset_dir.mkdir(exist_ok=True)
        
        print("ℹ PlasmoID dataset requires contacting authors:")
        print("  Paper: https://arxiv.org/abs/2211.15105")
        print("  Authors: Hanung Adi Nugroho et al.")
        print("  Institution: Universitas Gadjah Mada, Indonesia")
        print(f"  Once obtained, place in: {dataset_dir}")
    
    def download_iml_dataset(self):
        """Information for IML dataset"""
        print("\n" + "="*50)
        print("IML Dataset Information...")
        print("="*50)
        
        dataset_dir = self.base_dir / "iml"
        dataset_dir.mkdir(exist_ok=True)
        
        print("ℹ IML dataset available at:")
        print("  http://im.itu.edu.pk/a-dataset-and-benchmark-for-malaria-life-cycle-classification-in-thin-blood-smear-images/")
        print("  345 images with P. vivax life cycle stages")
        print(f"  Download and place in: {dataset_dir}")
    
    def download_m5_dataset(self):
        """Information for M5 dataset"""
        print("\n" + "="*50)
        print("M5 Dataset Information...")
        print("="*50)
        
        dataset_dir = self.base_dir / "m5"
        dataset_dir.mkdir(exist_ok=True)
        
        print("ℹ M5 dataset available at:")
        print("  http://im.itu.edu.pk/m5-malaria-dataset/")
        print("  Multi-microscope, multi-magnification dataset")
        print(f"  Download and place in: {dataset_dir}")
    
    def create_dataset_info(self):
        """Create dataset information file"""
        info = {
            "datasets": {
                "nih_cell": {
                    "status": "downloaded" if (self.base_dir / "nih_cell").exists() else "pending",
                    "images": 27558,
                    "classes": 2
                },
                "mp_idb": {
                    "status": "downloaded" if (self.base_dir / "mp_idb").exists() else "pending",
                    "images": 210,
                    "classes": 4
                },
                "bbbc041": {
                    "status": "manual_download_required",
                    "images": 1364,
                    "classes": 6
                },
                "plasmoID": {
                    "status": "contact_authors",
                    "images": 559,
                    "classes": 4
                },
                "iml": {
                    "status": "manual_download_required",
                    "images": 345,
                    "classes": 4
                },
                "m5": {
                    "status": "manual_download_required",
                    "images": 1257,
                    "classes": 3
                }
            }
        }
        
        info_path = self.base_dir / "dataset_info.json"
        with open(info_path, 'w') as f:
            json.dump(info, f, indent=2)
        
        print(f"\n✓ Dataset information saved to {info_path}")
    
    def download_all(self):
        """Download all available datasets"""
        print("\n" + "="*60)
        print(" MALARIA DATASET DOWNLOAD MANAGER ")
        print("="*60)
        
        # Download datasets that can be automated
        self.download_nih_cell_images()
        self.download_nih_thick_smears()
        self.download_mp_idb()
        self.download_kaggle_dataset()
        
        # Provide information for manual downloads
        self.download_bbbc041()
        self.download_plasmoID()
        self.download_iml_dataset()
        self.download_m5_dataset()
        
        # Create summary
        self.create_dataset_info()
        
        print("\n" + "="*60)
        print(" DOWNLOAD SUMMARY ")
        print("="*60)
        print("✓ Automated downloads completed")
        print("ℹ Some datasets require manual download")
        print("ℹ Check dataset_info.json for status")
        print("="*60)


def main():
    """Main execution function with flexible dataset selection"""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Download malaria datasets for research",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Download only MP-IDB (recommended for two-step classification research)
  python scripts/01_download_datasets.py --dataset mp_idb
  
  # Download specific datasets
  python scripts/01_download_datasets.py --dataset nih_cell,mp_idb
  
  # Download all datasets (full research)
  python scripts/01_download_datasets.py --dataset all
  
  # List available datasets
  python scripts/01_download_datasets.py --list-datasets

Available datasets:
  mp_idb      - MP-IDB whole slide images (RECOMMENDED for detection research)
  nih_cell    - NIH segmented cell images
  nih_thick   - NIH thick smear images 
  kaggle_nih  - Kaggle NIH cell dataset
  bbbc041     - BBBC041 P. vivax stages
  plasmoID    - PlasmoID Indonesian dataset
  iml         - IML P. vivax lifecycle
  all         - Download all datasets
        """)
    
    parser.add_argument(
        "--config",
        type=str,
        default="config/dataset_config.yaml",
        help="Path to dataset configuration file"
    )
    
    parser.add_argument(
        "--dataset",
        type=str,
        default="mp_idb",
        help="Dataset(s) to download. Use comma-separated list for multiple datasets"
    )
    
    parser.add_argument(
        "--list-datasets",
        action="store_true",
        help="List available datasets and exit"
    )
    
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        default=True,
        help="Skip datasets that already exist (default: True)"
    )
    
    parser.add_argument(
        "--force-redownload",
        action="store_true",
        help="Force re-download even if dataset exists"
    )
    
    args = parser.parse_args()
    
    # List available datasets
    if args.list_datasets:
        print("Available datasets for malaria detection research:")
        print("=" * 60)
        print("🎯 RECOMMENDED for Two-Step Classification Research:")
        print("  mp_idb      - MP-IDB whole slide images (103 images, 1,242 parasites)")
        print("                Required for detection → classification pipeline")
        print()
        print("📊 Additional Research Datasets:")
        print("  nih_cell    - NIH segmented cell images (27k+ cells)")
        print("  nih_thick   - NIH thick smear images (P. falciparum & P. vivax)")
        print("  kaggle_nih  - Kaggle NIH dataset (alternative source)")
        print("  bbbc041     - BBBC041 P. vivax lifecycle stages")
        print("  plasmoID    - PlasmoID Indonesian dataset (4 species)")
        print("  iml         - IML P. vivax lifecycle images")
        print()
        print("💡 Usage Examples:")
        print("  # For main research pipeline (RECOMMENDED):")
        print("  python scripts/01_download_datasets.py --dataset mp_idb")
        print()
        print("  # For comprehensive research:")
        print("  python scripts/01_download_datasets.py --dataset all")
        print()
        print("  # Multiple specific datasets:")
        print("  python scripts/01_download_datasets.py --dataset mp_idb,nih_cell")
        return
    
    # Initialize downloader
    downloader = MalariaDatasetDownloader(args.config)
    
    # Override skip_existing if force redownload is requested
    if args.force_redownload:
        downloader.skip_existing = False
        print("🔄 Force re-download mode enabled")
    else:
        downloader.skip_existing = args.skip_existing
    
    # Parse dataset selection
    datasets_to_download = [ds.strip().lower() for ds in args.dataset.split(',')]
    
    # Mapping of dataset names to download methods (FIXED method names)
    dataset_methods = {
        'mp_idb': ('MP-IDB Whole Slide Images', downloader.download_mp_idb),
        'nih_cell': ('NIH Cell Images', downloader.download_nih_cell_images),
        'nih_thick': ('NIH Thick Smears', downloader.download_nih_thick_smears),
        'kaggle_nih': ('Kaggle NIH Dataset', downloader.download_kaggle_dataset),
        'bbbc041': ('BBBC041 P. vivax', downloader.download_bbbc041),
        'plasmoid': ('PlasmoID Indonesian', downloader.download_plasmoID),
        'iml': ('IML P. vivax Lifecycle', downloader.download_iml_dataset),
    }
    
    print("🔬 Malaria Dataset Downloader")
    print("=" * 50)
    
    if 'all' in datasets_to_download:
        print("📥 Downloading ALL datasets...")
        print("⏱️  Estimated time: 30-60 minutes")
        print("💾 Required space: ~6GB")
        print()
        downloader.download_all()
    else:
        # Validate dataset names
        invalid_datasets = []
        valid_datasets = []
        
        for dataset in datasets_to_download:
            if dataset in dataset_methods:
                valid_datasets.append(dataset)
            else:
                invalid_datasets.append(dataset)
        
        if invalid_datasets:
            print(f"❌ Invalid dataset(s): {', '.join(invalid_datasets)}")
            print(f"✅ Valid options: {', '.join(dataset_methods.keys())}, all")
            print("💡 Use --list-datasets to see all available options")
            return
        
        # Download selected datasets
        print(f"📥 Downloading {len(valid_datasets)} dataset(s)...")
        
        for dataset in valid_datasets:
            name, method = dataset_methods[dataset]
            print(f"\n🔄 Downloading: {name}")
            try:
                method()
                print(f"✅ {name} download completed")
            except Exception as e:
                print(f"❌ {name} download failed: {str(e)}")
                continue
    
    print("\n" + "=" * 50)
    print("✅ Download process completed!")
    
    # Show downloaded datasets summary
    import os
    data_dir = "data/raw"
    if os.path.exists(data_dir):
        downloaded_dirs = [d for d in os.listdir(data_dir) 
                          if os.path.isdir(os.path.join(data_dir, d)) and d != '.gitkeep']
        print(f"📊 Total datasets available: {len(downloaded_dirs)}")
        print(f"📁 Downloaded datasets: {', '.join(downloaded_dirs)}")
        
        # Special note for MP-IDB (main research dataset)
        if 'mp_idb' in downloaded_dirs:
            print("\n🎯 MP-IDB Ready for Two-Step Classification Pipeline:")
            print("   Next step: python scripts/08_parse_mpid_detection.py")
    
    print("\n💡 Next Steps:")
    if 'mp_idb' in [ds.strip().lower() for ds in args.dataset.split(',')]:
        print("   1. Parse detection dataset: python scripts/08_parse_mpid_detection.py")
        print("   2. Crop parasites: python scripts/09_crop_parasites_from_detection.py")
        print("   3. Train models: python pipeline.py train yolov8_detection --name first_run")
    else:
        print("   See README.md for complete pipeline instructions")


if __name__ == "__main__":
    main()
