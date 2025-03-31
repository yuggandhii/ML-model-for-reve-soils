import torch
import torch.nn as nn
from torchvision.models import resnet50, ResNet50_Weights

class CropDiseaseModel(nn.Module):
    def __init__(self, num_classes):
        """
        Initialize the crop disease classification model
        Args:
            num_classes: Number of disease classes
        """
        super(CropDiseaseModel, self).__init__()
        
        # Use ResNet50 instead of ResNet18 to match the saved weights
        self.resnet = resnet50(weights=ResNet50_Weights.DEFAULT)
        
        # Remove the original fc layer from resnet
        num_features = self.resnet.fc.in_features
        self.resnet.fc = nn.Identity()
        
        # Add separate classifier matching the saved weights structure
        self.classifier = nn.Sequential(
            nn.Linear(num_features, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )
        
    def forward(self, x):
        # Extract features from ResNet
        features = self.resnet(x)
        # Pass features through classifier
        output = self.classifier(features)
        return output
    
    def unfreeze_layers(self, num_layers=0):
        """
        Unfreeze the last n layers of the model for fine-tuning
        If num_layers=0, all layers will be unfrozen
        """
        # First freeze all layers
        for param in self.resnet.parameters():
            param.requires_grad = False
            
        if num_layers == 0:
            # Unfreeze all layers
            for param in self.resnet.parameters():
                param.requires_grad = True
        else:
            # Get all layers
            layers = list(self.resnet.children())
            
            # Unfreeze the last n layers
            for layer in layers[-num_layers:]:
                for param in layer.parameters():
                    param.requires_grad = True
        
        # Always unfreeze the classifier
        for param in self.classifier.parameters():
            param.requires_grad = True
    
    def predict(self, image):
        """
        Make prediction on a single image
        Args:
            image: Preprocessed image tensor
        Returns:
            Predicted class and confidence
        """
        self.eval()
        with torch.no_grad():
            outputs = self(image.unsqueeze(0))
            probabilities = torch.softmax(outputs, dim=1)
            confidence, predicted = torch.max(probabilities, 1)
            return predicted.item(), confidence.item()
    
    def predict_batch(self, images):
        """
        Make predictions on a batch of images
        Args:
            images: Batch of preprocessed image tensors
        Returns:
            Predicted classes and confidences
        """
        self.eval()
        with torch.no_grad():
            outputs = self(images)
            probabilities = torch.softmax(outputs, dim=1)
            confidences, predicted = torch.max(probabilities, 1)
            return predicted.cpu().numpy(), confidences.cpu().numpy() 