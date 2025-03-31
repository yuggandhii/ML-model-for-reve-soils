import os
from pathlib import Path
from data.download_datasets import download_dataset
from training.train_unified import train_dataset
from data.five_crop_dataset import FiveCropDataset
from data.mendeley_dataset import MendeleyDataset
from data.plant_village_dataset import PlantVillageDataset
from data.crop_diseases_dataset import CropDiseasesDataset
from data.plant_622_dataset import Plant622Dataset
from data.pathogen_dataset import PathogenDataset

def main():
    # Create necessary directories
    Path('data').mkdir(exist_ok=True)
    Path('outputs').mkdir(exist_ok=True)
    
    # Define datasets to download and train
    datasets = {
        'five_crop': {
            'kaggle_id': 'shubham2703/five-crop-diseases-dataset',
            'path': 'data/five_crop_diseases',
            'class': FiveCropDataset,
            'num_classes': 20  # 5 crops * 4 states (healthy + 3 diseases)
        },
        'mendeley': {
            'kaggle_id': 'vipoooool/new-plant-diseases-dataset',
            'path': 'data/mendeley_dataset',
            'class': MendeleyDataset,
            'num_classes': 2  # healthy vs diseased
        },
        'plant_622': {
            'kaggle_id': 'daoliu/plant-622',
            'path': 'data/plant_622',
            'class': Plant622Dataset,
            'num_classes': 622  # 622 different plant species
        },
        'plant_village': {
            'kaggle_id': 'abdallahalidev/plantvillage-dataset',
            'path': 'data/plant_village',
            'class': PlantVillageDataset,
            'num_classes': 38  # 38 different plant diseases
        },
        'crop_diseases': {
            'kaggle_id': 'mexwell/crop-diseases-classification',
            'path': 'data/crop_diseases',
            'class': CropDiseasesDataset,
            'num_classes': 4  # 4 different crop diseases
        },
        'plant_village_updated': {
            'kaggle_id': 'tushar5harma/plant-village-dataset-updated',
            'path': 'data/plant_village_updated',
            'class': PlantVillageDataset,
            'num_classes': 38  # 38 different plant diseases
        },
        'pathogen': {
            'kaggle_id': 'kanishk3813/pathogen-dataset',
            'path': 'data/pathogen',
            'class': PathogenDataset,
            'num_classes': 3  # 3 different pathogen types
        },
        'plant_village_alt': {
            'kaggle_id': 'adilmubashirchaudhry/plant-village-dataset',
            'path': 'data/plant_village_alt',
            'class': PlantVillageDataset,
            'num_classes': 38  # 38 different plant diseases
        }
    }
    
    # Download and train on each dataset
    for name, config in datasets.items():
        print(f"\nProcessing {name} dataset...")
        
        # Download dataset if not already present
        if not os.path.exists(config['path']):
            print(f"Downloading {name} dataset...")
            try:
                download_dataset(config['kaggle_id'], config['path'])
                print(f"Successfully downloaded {name} dataset")
            except Exception as e:
                print(f"Error downloading {name} dataset: {str(e)}")
                continue
        else:
            print(f"{name} dataset already exists")
        
        # Train on the dataset
        try:
            train_dataset(name, config['class'], config['path'], config['num_classes'])
            print(f"Successfully trained on {name} dataset")
        except Exception as e:
            print(f"Error training on {name} dataset: {str(e)}")
            continue

if __name__ == '__main__':
    main() 