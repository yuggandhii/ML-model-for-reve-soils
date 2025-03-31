import os
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms

class NewPlantDataset(Dataset):
    """Dataset class for New Plant Diseases dataset"""
    def __init__(self, root_dir, transform=None, split='train'):
        self.root_dir = root_dir
        self.transform = transform
        self.classes = []
        self.class_to_idx = {}
        self.images = []
        self.labels = []
        
        # Set the data directory based on split
        data_dir = os.path.join(root_dir, 'New Plant Diseases Dataset(Augmented)', split)
        
        # Get all class directories
        class_dirs = [d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))]
        
        # Process each class directory
        for class_idx, class_dir in enumerate(sorted(class_dirs)):
            class_path = os.path.join(data_dir, class_dir)
            if not os.path.isdir(class_path):
                continue
                
            # Get all images in the class directory
            for img_name in os.listdir(class_path):
                if img_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                    img_path = os.path.join(class_path, img_name)
                    self.images.append(img_path)
                    self.labels.append(class_idx)
                    
            # Add class name
            self.classes.append(class_dir)
            self.class_to_idx[class_dir] = class_idx
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        # Load image
        img_path = self.images[idx]
        image = Image.open(img_path).convert('RGB')
        
        # Apply transformations
        if self.transform:
            image = self.transform(image)
        
        # Get label
        label = self.labels[idx]
        
        return image, label
    
    def get_class_name(self, idx):
        """Get the class name for a given index"""
        for class_name, class_idx in self.class_to_idx.items():
            if class_idx == idx:
                return class_name
        return None 