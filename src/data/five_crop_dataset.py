import os
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms

class FiveCropDataset(Dataset):
    """Dataset class for Five Crop Diseases dataset"""
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.classes = []
        self.class_to_idx = {}
        self.images = []
        self.labels = []
        
        # Get all crop directories
        crop_dirs = [d for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d))]
        
        # Process each crop directory
        for crop_idx, crop_dir in enumerate(sorted(crop_dirs)):
            crop_path = os.path.join(root_dir, crop_dir)
            if not os.path.isdir(crop_path):
                continue
                
            # Get all disease directories
            disease_dirs = [d for d in os.listdir(crop_path) if os.path.isdir(os.path.join(crop_path, d))]
            
            # Process each disease directory
            for disease_idx, disease_dir in enumerate(sorted(disease_dirs)):
                disease_path = os.path.join(crop_path, disease_dir)
                if not os.path.isdir(disease_path):
                    continue
                    
                # Get all images in the disease directory
                for img_name in os.listdir(disease_path):
                    if img_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                        img_path = os.path.join(disease_path, img_name)
                        self.images.append(img_path)
                        self.labels.append(crop_idx * len(disease_dirs) + disease_idx)
                        
                # Add class name
                class_name = f"{crop_dir}_{disease_dir}"
                self.classes.append(class_name)
                self.class_to_idx[class_name] = crop_idx * len(disease_dirs) + disease_idx
    
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