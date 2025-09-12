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
    """Main execution function"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Download malaria datasets")
    parser.add_argument(
        "--config",
        type=str,
        default="config/dataset_config.yaml",
        help="Path to dataset configuration file"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        choices=["all", "nih", "mp_idb", "kaggle", "bbbc041"],
        default="all",
        help="Specific dataset to download"
    )
    
    args = parser.parse_args()
    
    downloader = MalariaDatasetDownloader(args.config)
    
    if args.dataset == "all":
        downloader.download_all()
    elif args.dataset == "nih":
        downloader.download_nih_cell_images()
        downloader.download_nih_thick_smears()
    elif args.dataset == "mp_idb":
        downloader.download_mp_idb()
    elif args.dataset == "kaggle":
        downloader.download_kaggle_dataset()
    elif args.dataset == "bbbc041":
        downloader.download_bbbc041()
    
    print("\n✅ Download process completed!")


if __name__ == "__main__":
    main()