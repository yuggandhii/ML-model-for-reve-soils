import os
from pathlib import Path
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms
import numpy as np

class PlantVillageDataset(Dataset):
    """Dataset class for Plant Village dataset"""
    
    def __init__(self, root_dir, transform=None, split='train', val_split=0.2):
        """
        Args:
            root_dir (str): Directory with all the images organized in class folders
            transform (callable, optional): Optional transform to be applied on a sample
            split (str): 'train' or 'valid' to specify the dataset split
            val_split (float): Fraction of data to use for validation
        """
        self.root_dir = Path(root_dir) / 'color'  # Use color images
        self.transform = transform
        self.split = split
        self.val_split = val_split
        
        # Get all class directories
        self.classes = sorted([d.name for d in self.root_dir.iterdir() if d.is_dir()])
        self.class_to_idx = {cls_name: i for i, cls_name in enumerate(self.classes)}
        
        # Get all image paths and labels
        self.image_paths = []
        self.labels = []
        
        for class_name in self.classes:
            class_dir = self.root_dir / class_name
            if not class_dir.is_dir():
                continue
                
            class_idx = self.class_to_idx[class_name]
            # Look for both .JPG and .jpg files
            image_files = []
            for ext in ['*.jpg', '*.JPG', '*.jpeg', '*.JPEG']:
                image_files.extend(list(class_dir.glob(ext)))
            
            # Sort for reproducibility
            image_files = sorted(image_files)
            
            # Split into train and validation
            num_images = len(image_files)
            num_val = int(num_images * val_split)
            
            if split == 'train':
                image_files = image_files[num_val:]
            else:  # validation
                image_files = image_files[:num_val]
            
            self.image_paths.extend(image_files)
            self.labels.extend([class_idx] * len(image_files))
        
        # Convert to numpy arrays for faster indexing
        self.image_paths = np.array(self.image_paths)
        self.labels = np.array(self.labels)
        
        print(f"Found {len(self.image_paths)} images in {split} split")
        print(f"Number of classes: {len(self.classes)}")
        
        # Print class distribution
        class_counts = {}
        for label in self.labels:
            class_name = self.get_class_name(label)
            class_counts[class_name] = class_counts.get(class_name, 0) + 1
        
        print("\nClass distribution:")
        for class_name, count in class_counts.items():
            print(f"{class_name}: {count} images")
    
    def __len__(self):
        """Returns the total number of samples"""
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        """Returns one sample of data"""
        if torch.is_tensor(idx):
            idx = idx.tolist()
        
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert('RGB')
        label = self.labels[idx]
        
        if self.transform:
            image = self.transform(image)
        
        return image, label
    
    def get_class_name(self, idx):
        """Returns the class name for a given class index"""
        for name, index in self.class_to_idx.items():
            if index == idx:
                return name
        return None 