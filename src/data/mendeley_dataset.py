import os
from PIL import Image
import torch
from torch.utils.data import Dataset
import albumentations as A
from albumentations.pytorch import ToTensorV2
import numpy as np

class MendeleyDataset(Dataset):
    def __init__(self, root_dir, transform=None, is_train=True):
        """
        Initialize the Mendeley plant disease dataset
        Args:
            root_dir: Root directory containing the dataset
            transform: Optional transform to be applied on the data
            is_train: Whether this is training or validation data
        """
        self.root_dir = root_dir
        self.is_train = is_train
        
        # Define plant names and their corresponding class numbers
        self.plant_names = {
            'P0': 'Mango',
            'P1': 'Arjun',
            'P2': 'Alstonia Scholaris',
            'P3': 'Guava',
            'P4': 'Bael',
            'P5': 'Jamun',
            'P6': 'Jatropha',
            'P7': 'Pongamia Pinnata',
            'P8': 'Basil',
            'P9': 'Pomegranate',
            'P10': 'Lemon',
            'P11': 'Chinar'
        }
        
        # Get all image paths and labels
        self.images = []
        self.labels = []
        self.plant_types = []  # Store plant type for each image
        self.health_status = []  # Store health status for each image
        
        # Process all images in the dataset
        for img_name in os.listdir(root_dir):
            if img_name.lower().endswith(('.jpg', '.jpeg')):
                # Extract plant type and health status from filename
                # Format: P{plant_number}_{health_status}.jpg
                parts = img_name.split('_')
                if len(parts) >= 2:
                    plant_type = parts[0]
                    health_status = 'healthy' if int(parts[1].split('.')[0]) < 12 else 'diseased'
                    
                    self.images.append(os.path.join(root_dir, img_name))
                    self.labels.append(1 if health_status == 'diseased' else 0)
                    self.plant_types.append(plant_type)
                    self.health_status.append(health_status)
        
        # Define default transforms if none provided
        if transform is None:
            if is_train:
                self.transform = A.Compose([
                    A.Resize(224, 224),
                    A.RandomHorizontalFlip(p=0.5),
                    A.RandomRotation(limit=15),
                    A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
                    A.RandomBrightnessContrast(p=0.2),
                    A.Normalize(mean=[0.485, 0.456, 0.406], 
                              std=[0.229, 0.224, 0.225]),
                    ToTensorV2(),
                ])
            else:
                self.transform = A.Compose([
                    A.Resize(224, 224),
                    A.Normalize(mean=[0.485, 0.456, 0.406], 
                              std=[0.229, 0.224, 0.225]),
                    ToTensorV2(),
                ])
        else:
            self.transform = transform
            
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        img_path = self.images[idx]
        label = self.labels[idx]
        plant_type = self.plant_types[idx]
        health_status = self.health_status[idx]
        
        # Load and convert image
        image = Image.open(img_path).convert('RGB')
        
        # Apply transforms
        if self.transform:
            image = self.transform(image=np.array(image))['image']
            
        return {
            'image': image,
            'label': label,
            'plant_type': plant_type,
            'health_status': health_status
        }
    
    def get_class_distribution(self):
        """Get the distribution of classes in the dataset"""
        class_counts = {
            'healthy': 0,
            'diseased': 0
        }
        for status in self.health_status:
            class_counts[status] += 1
        return class_counts
    
    def get_plant_distribution(self):
        """Get the distribution of plant types in the dataset"""
        plant_counts = {}
        for plant in self.plant_types:
            plant_counts[plant] = plant_counts.get(plant, 0) + 1
        return plant_counts 