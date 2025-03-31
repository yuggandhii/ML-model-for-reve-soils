import os
from PIL import Image
import torch
from torch.utils.data import Dataset
import albumentations as A
from albumentations.pytorch import ToTensorV2
import numpy as np

class CropDiseaseDataset(Dataset):
    def __init__(self, root_dir, transform=None, is_train=True):
        """
        Initialize the crop disease dataset
        Args:
            root_dir: Root directory containing disease subdirectories
            transform: Optional transform to be applied on the data
            is_train: Whether this is training or validation data
        """
        self.root_dir = root_dir
        self.is_train = is_train
        self.classes = sorted(os.listdir(root_dir))
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}
        
        # Get all image paths and labels
        self.images = []
        self.labels = []
        for cls in self.classes:
            cls_dir = os.path.join(root_dir, cls)
            for img_name in os.listdir(cls_dir):
                if img_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                    self.images.append(os.path.join(cls_dir, img_name))
                    self.labels.append(self.class_to_idx[cls])
        
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
        
        # Load and convert image
        image = Image.open(img_path).convert('RGB')
        
        # Apply transforms
        if self.transform:
            image = self.transform(image=np.array(image))['image']
            
        return image, label
    
    def get_class_distribution(self):
        """Get the distribution of classes in the dataset"""
        class_counts = {}
        for label in self.labels:
            class_name = self.classes[label]
            class_counts[class_name] = class_counts.get(class_name, 0) + 1
        return class_counts 