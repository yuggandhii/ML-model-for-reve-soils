import torch
import torch.nn as nn
import torchvision.models as models
import os

class CropDiseaseModel(nn.Module):
    def __init__(self, num_classes, pretrained=True):
        super(CropDiseaseModel, self).__init__()
        # Load pre-trained ResNet model
        self.model = models.resnet50(pretrained=pretrained)
        
        # Freeze the feature extraction layers
        for param in self.model.parameters():
            param.requires_grad = False
            
        # Modify the final layer for our number of classes
        num_features = self.model.fc.in_features
        self.model.fc = nn.Sequential(
            nn.Linear(num_features, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )
        
    def forward(self, x):
        return self.model(x)
    
    def unfreeze_layers(self, num_layers=0):
        """Unfreeze the last n layers for fine-tuning"""
        if num_layers == 0:
            return
        
        # Get all layers
        layers = list(self.model.children())
        
        # Unfreeze the last n layers
        for layer in layers[-num_layers:]:
            for param in layer.parameters():
                param.requires_grad = True 