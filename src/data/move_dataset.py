import os
import shutil
from pathlib import Path

def move_dataset():
    """Move the Five Crop Diseases dataset to the correct location"""
    # Source path
    source_path = Path('kagglehub/datasets/shubham2703/five-crop-diseases-dataset/versions/1/Crop Diseases Dataset/Crop Diseases/Crop___Disease')
    
    # Destination path
    dest_path = Path('data/five_crop_diseases')
    
    # Create destination directory if it doesn't exist
    dest_path.mkdir(parents=True, exist_ok=True)
    
    # Move each crop directory
    for crop_dir in source_path.iterdir():
        if crop_dir.is_dir():
            print(f"Moving {crop_dir.name}...")
            shutil.copytree(crop_dir, dest_path / crop_dir.name, dirs_exist_ok=True)
    
    print("Dataset transfer complete!")

if __name__ == '__main__':
    move_dataset() 