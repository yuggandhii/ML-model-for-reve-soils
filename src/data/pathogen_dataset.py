import os
from pathlib import Path
import torch
from torch.utils.data import Dataset
from PIL import Image
import albumentations as A
from albumentations.pytorch import ToTensorV2
import numpy as np
from collections import Counter

class PathogenDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = Path(root_dir)
        self.transform = transform or self._get_default_transforms()
        
        # Get all image paths and labels
        self.image_paths = []
        self.labels = []
        self.class_to_idx = {}
        
        # Walk through the directory structure
        for class_dir in sorted(self.root_dir.iterdir()):
            if class_dir.is_dir():
                class_name = class_dir.name
                self.class_to_idx[class_name] = len(self.class_to_idx)
                
                for img_path in class_dir.glob('*.jpg'):
                    self.image_paths.append(img_path)
                    self.labels.append(self.class_to_idx[class_name])
        
        # Create idx_to_class mapping
        self.idx_to_class = {v: k for k, v in self.class_to_idx.items()}
        
    def _get_default_transforms(self):
        return A.Compose([
            A.Resize(224, 224),
            A.HorizontalFlip(p=0.5),
            A.RandomRotate90(p=0.5),
            A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1, p=0.5),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2()
        ])
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        label = self.labels[idx]
        
        # Load and transform image
        image = Image.open(img_path).convert('RGB')
        image = np.array(image)
        
        if self.transform:
            transformed = self.transform(image=image)
            image = transformed['image']
        
        return {
            'image': image,
            'label': torch.tensor(label, dtype=torch.long)
        }
    
    def get_class_distribution(self):
        """Get the distribution of classes in the dataset"""
        return Counter(self.labels)
    
    def get_class_name(self, idx):
        """Get the class name for a given index"""
        return self.idx_to_class[idx]
    
    def get_num_classes(self):
        """Get the number of classes in the dataset"""
        return len(self.class_to_idx) 