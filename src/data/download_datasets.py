import os
import kaggle
import zipfile
import shutil
from pathlib import Path
import psutil
import sys

def setup_kaggle_credentials():
    """Setup Kaggle API credentials"""
    kaggle_dir = Path.home() / '.kaggle'
    if not (kaggle_dir / 'kaggle.json').exists():
        print("Please set up your Kaggle API credentials:")
        print("1. Go to https://www.kaggle.com/settings")
        print("2. Scroll to API section and click 'Create New API Token'")
        print("3. Save the downloaded kaggle.json to ~/.kaggle/kaggle.json")
        print("4. Run: chmod 600 ~/.kaggle/kaggle.json")
        return False
    return True

def check_disk_space(required_gb):
    """Check if there's enough disk space on D drive"""
    try:
        disk = psutil.disk_usage('D:/')
        available_gb = disk.free / (1024**3)  # Convert to GB
        return available_gb >= required_gb
    except Exception as e:
        print(f"Error checking disk D space: {str(e)}")
        return False

def download_dataset(dataset_name, dataset_path):
    """Download and extract a single dataset"""
    try:
        print(f"\nDownloading {dataset_name} dataset...")
        
        # Check disk space on D drive (add 1GB buffer)
        required_gb = 5  # Approximate size needed for download and extraction
        if not check_disk_space(required_gb):
            print(f"Error: Not enough disk space on D drive. Need at least {required_gb}GB free space.")
            return False
            
        # Create data directory on D drive
        data_dir = Path('D:/ml_data')
        data_dir.mkdir(exist_ok=True)
        
        # Download the dataset using Kaggle API
        kaggle.api.dataset_download_files(dataset_name, path=data_dir, unzip=True)
        print(f"Successfully downloaded and extracted {dataset_name} dataset!")
        return True
        
    except Exception as e:
        print(f"Error downloading {dataset_name}: {str(e)}")
        return False

def main():
    """Main function to download datasets"""
    print("Starting dataset download process...")
    
    # Check Kaggle credentials
    if not setup_kaggle_credentials():
        return
    
    # Define all datasets to download
    datasets = {
        'new_plant': ('vipoooool/new-plant-diseases-dataset', 'new_plant_diseases'),
        'plant_village': ('abdallahalidev/plantvillage-dataset', 'plant_village'),
        'crop_diseases': ('mexwell/crop-diseases-classification', 'crop_diseases'),
        'plant_village_updated': ('tushar5harma/plant-village-dataset-updated', 'plant_village_updated'),
        'pathogen': ('kanishk3813/pathogen-dataset', 'pathogen'),
        'plant_village_alt': ('adilmubashirchaudhry/plant-village-dataset', 'plant_village_alt'),
        'plant_622': ('daoliu/plant-622', 'plant_622')
    }
    
    # Download each dataset
    for dataset_id, (dataset_name, dataset_path) in datasets.items():
        print(f"\nProcessing {dataset_id} dataset...")
        success = download_dataset(dataset_name, dataset_path)
        if success:
            print(f"Successfully processed {dataset_id} dataset!")
        else:
            print(f"Failed to download {dataset_id} dataset!")

if __name__ == '__main__':
    main() 