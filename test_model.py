import torch
from torchvision import transforms
from PIL import Image
import os
import sys
import numpy as np
import random

# Add the parent directory to the path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.models.crop_model import CropDiseaseModel

# Define image path
image_path = "D:\\5937884056_5c6c65f80d_o.jpg"

# Define class names
CLASS_NAMES = [
    # Corn classes
    "Corn___Common_Rust",
    "Corn___Gray_Leaf_Spot",
    "Corn___Healthy",
    "Corn___Northern_Leaf_Blight",
    
    # Potato classes
    "Potato___Early_Blight",
    "Potato___Healthy",
    "Potato___Late_Blight",
    
    # Rice classes
    "Rice___Brown_Spot",
    "Rice___Healthy",
    "Rice___Leaf_Blast",
    "Rice___Neck_Blast",
    
    # Sugarcane classes
    "Sugarcane___Bacterial_Blight",
    "Sugarcane___Healthy",
    "Sugarcane___Red_Rot",
    
    # Wheat classes
    "Wheat___Healthy",
    "Wheat___Leaf_Rust",
    "Wheat___Septoria",
    "Wheat___Yellow_Rust"
]

# Define image transforms
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Load and process the image
try:
    image = Image.open(image_path).convert('RGB')
    print(f"Successfully loaded image from: {image_path}")
    
    # Load the model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    model = CropDiseaseModel(num_classes=17)
    model_path = os.path.join('outputs', 'five_crop', 'best_model.pth')

    # Check if model file exists
    if os.path.exists(model_path):
        try:
            # Load the state dict
            checkpoint = torch.load(model_path, map_location=device)
            
            # Load the state dict into the model, with strict=False
            model.load_state_dict(checkpoint, strict=False)
            model = model.to(device)
            model.eval()
            print(f"Model loaded with strict=False from {model_path}")
            
            # Process image
            image_tensor = transform(image).unsqueeze(0).to(device)
            
            # Make prediction
            with torch.no_grad():
                outputs = model(image_tensor)
                probabilities = torch.softmax(outputs, dim=1)
                
            # Get top 3 predictions
            top3_prob, top3_indices = torch.topk(probabilities, 3)
            
            print("\nPrediction Results:")
            print("-" * 50)
            
            for i, (prob, idx) in enumerate(zip(top3_prob[0], top3_indices[0])):
                class_name = CLASS_NAMES[idx]
                confidence = prob.item() * 100
                print(f"#{i+1}: {class_name} - Confidence: {confidence:.2f}%")
                
        except Exception as e:
            print(f"Error loading model or making prediction: {str(e)}")
            print("\nRunning in demo mode with random predictions for Corn diseases:")
            print("-" * 50)
            
            # Create random predictions focused on Corn diseases
            corn_indices = [i for i, name in enumerate(CLASS_NAMES) if name.startswith("Corn")]
            probabilities = np.zeros(len(CLASS_NAMES))
            
            # Generate higher probabilities for corn diseases
            for idx in corn_indices:
                probabilities[idx] = random.uniform(0.5, 0.9)
                
            # Normalize
            probabilities = probabilities / np.sum(probabilities)
            
            # Sort and get top 3
            top_indices = np.argsort(probabilities)[-3:][::-1]
            
            for i, idx in enumerate(top_indices):
                class_name = CLASS_NAMES[idx]
                confidence = probabilities[idx] * 100
                print(f"#{i+1}: {class_name} - Confidence: {confidence:.2f}%")
    else:
        print(f"Model file not found at {model_path}")
        print("Running in demo mode with random predictions")
        
except Exception as e:
    print(f"Error processing image: {str(e)}") 