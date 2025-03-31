import os
import shutil
from pathlib import Path

def move_all_datasets():
    """Move all datasets from kagglehub to their respective directories"""
    # Source base path
    source_base = Path('kagglehub/datasets')
    
    # Destination base path
    dest_base = Path('data')
    
    # Create destination base directory if it doesn't exist
    dest_base.mkdir(exist_ok=True)
    
    # Map of dataset owners to their respective directories
    dataset_mapping = {
        'vipoooool': 'new_plant_diseases',
        'abdallahalidev': 'plant_village',
        'mexwell': 'crop_diseases',
        'tushar5harma': 'plant_village_updated',
        'kanishk3813': 'pathogen',
        'adilmubashirchaudhry': 'plant_village_alt',
        'daoliu': 'plant_622'
    }
    
    # Move each dataset
    for owner, dest_dir in dataset_mapping.items():
        try:
            # Find the dataset directory
            owner_path = source_base / owner
            if not owner_path.exists():
                print(f"Source directory not found for {owner}")
                continue
                
            # Get the dataset directory (should be the first subdirectory)
            dataset_dirs = list(owner_path.iterdir())
            if not dataset_dirs:
                print(f"No dataset found for {owner}")
                continue
                
            dataset_dir = dataset_dirs[0]
            versions_dir = dataset_dir / 'versions'
            if not versions_dir.exists():
                print(f"No versions directory found for {owner}")
                continue
                
            # Get the latest version directory
            version_dirs = list(versions_dir.iterdir())
            if not version_dirs:
                print(f"No version found for {owner}")
                continue
                
            latest_version = version_dirs[0]
            
            # Find the actual dataset directory
            dataset_contents = list(latest_version.iterdir())
            if not dataset_contents:
                print(f"No contents found for {owner}")
                continue
                
            # The actual dataset should be in a subdirectory
            for content in dataset_contents:
                if content.is_dir():
                    source_path = content
                    break
            else:
                print(f"No dataset directory found for {owner}")
                continue
                
            # Create destination directory
            dest_path = dest_base / dest_dir
            dest_path.mkdir(exist_ok=True)
            
            # Move the dataset
            print(f"Moving {owner} dataset to {dest_dir}...")
            shutil.copytree(source_path, dest_path, dirs_exist_ok=True)
            print(f"Successfully moved {owner} dataset!")
            
        except Exception as e:
            print(f"Error moving {owner} dataset: {str(e)}")
            continue

if __name__ == '__main__':
    move_all_datasets() 