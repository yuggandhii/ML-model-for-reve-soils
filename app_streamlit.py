import streamlit as st
import torch
from torchvision import transforms
from PIL import Image
import os
import sys
import numpy as np
import matplotlib.pyplot as plt

# Add the parent directory to the path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.models.crop_model import CropDiseaseModel

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

# Define disease information
DISEASE_INFO = {
    # Corn Diseases
    "Common_Rust": {
        "description": "A fungal disease that appears as circular to elongated brown pustules on leaves",
        "preventive_measures": [
            "Plant resistant varieties",
            "Maintain proper plant spacing for good air circulation",
            "Remove infected plant debris after harvest",
            "Practice crop rotation with non-host crops",
            "Monitor fields regularly for early detection"
        ],
        "management": [
            "Apply fungicides when disease is first detected",
            "Use appropriate nitrogen fertilization",
            "Ensure balanced soil nutrition",
            "Consider early planting to avoid peak disease periods",
            "Implement proper irrigation practices"
        ],
        "supportive_actions": [
            "Keep detailed records of disease occurrence",
            "Consult with agricultural extension services",
            "Consider biological control options",
            "Maintain field hygiene",
            "Monitor weather conditions for disease-favorable periods"
        ]
    },
    "Gray_Leaf_Spot": {
        "description": "Characterized by rectangular lesions with gray centers and brown borders",
        "preventive_measures": [
            "Use disease-free seeds",
            "Practice proper crop rotation",
            "Maintain optimal soil pH",
            "Implement proper tillage practices",
            "Choose resistant varieties"
        ],
        "management": [
            "Apply recommended fungicides",
            "Optimize irrigation to reduce leaf wetness",
            "Balance nutrient application",
            "Remove infected plant debris",
            "Consider reduced tillage systems"
        ],
        "supportive_actions": [
            "Regular field scouting",
            "Document disease patterns",
            "Maintain proper field drainage",
            "Consider intercropping with non-host crops",
            "Implement integrated pest management"
        ]
    },
    "Northern_Leaf_Blight": {
        "description": "Large, elliptical lesions with dark brown borders and tan centers",
        "preventive_measures": [
            "Plant resistant hybrids",
            "Practice crop rotation",
            "Use certified disease-free seeds",
            "Maintain proper plant density",
            "Implement proper field sanitation"
        ],
        "management": [
            "Apply appropriate fungicides",
            "Optimize nitrogen fertilization",
            "Ensure proper drainage",
            "Remove infected plant material",
            "Consider early planting"
        ],
        "supportive_actions": [
            "Regular field monitoring",
            "Keep weather records",
            "Maintain field hygiene",
            "Consider biological control methods",
            "Implement proper irrigation timing"
        ]
    },
    # Potato Diseases
    "Early_Blight": {
        "description": "Dark brown spots with concentric rings on leaves and tubers",
        "preventive_measures": [
            "Use certified seed potatoes",
            "Practice crop rotation",
            "Maintain proper plant spacing",
            "Ensure good soil drainage",
            "Remove volunteer plants"
        ],
        "management": [
            "Apply recommended fungicides",
            "Optimize irrigation practices",
            "Balance nutrient application",
            "Remove infected foliage",
            "Harvest at proper maturity"
        ],
        "supportive_actions": [
            "Regular field inspection",
            "Maintain proper storage conditions",
            "Document disease patterns",
            "Consider biological control",
            "Implement proper curing practices"
        ]
    },
    "Late_Blight": {
        "description": "Dark, water-soaked lesions on leaves and brown rot on tubers",
        "preventive_measures": [
            "Use disease-free seed potatoes",
            "Implement strict quarantine measures",
            "Practice proper crop rotation",
            "Maintain good field drainage",
            "Remove volunteer plants"
        ],
        "management": [
            "Apply appropriate fungicides",
            "Optimize irrigation timing",
            "Remove infected plants",
            "Harvest before disease spread",
            "Implement proper storage conditions"
        ],
        "supportive_actions": [
            "Regular field monitoring",
            "Weather-based disease forecasting",
            "Proper storage management",
            "Document disease patterns",
            "Consider biological control options"
        ]
    },
    # Rice Diseases
    "Brown_Spot": {
        "description": "Small, circular brown spots with yellow halos on leaves",
        "preventive_measures": [
            "Use certified seeds",
            "Maintain proper soil fertility",
            "Practice crop rotation",
            "Ensure proper water management",
            "Remove infected plant debris"
        ],
        "management": [
            "Apply recommended fungicides",
            "Optimize nutrient application",
            "Maintain proper water levels",
            "Remove infected plants",
            "Consider resistant varieties"
        ],
        "supportive_actions": [
            "Regular field inspection",
            "Soil testing and amendment",
            "Proper water management",
            "Document disease patterns",
            "Implement integrated pest management"
        ]
    },
    "Leaf_Blast": {
        "description": "Diamond-shaped lesions with gray centers and brown borders",
        "preventive_measures": [
            "Use resistant varieties",
            "Practice proper timing of planting",
            "Maintain optimal soil fertility",
            "Implement proper water management",
            "Remove infected plant debris"
        ],
        "management": [
            "Apply appropriate fungicides",
            "Optimize nitrogen application",
            "Maintain proper water levels",
            "Remove infected plants",
            "Consider early planting"
        ],
        "supportive_actions": [
            "Regular field monitoring",
            "Weather-based disease forecasting",
            "Proper water management",
            "Document disease patterns",
            "Implement integrated pest management"
        ]
    },
    "Neck_Blast": {
        "description": "Infection of the panicle neck causing grain loss",
        "preventive_measures": [
            "Use resistant varieties",
            "Practice proper timing of planting",
            "Maintain optimal soil fertility",
            "Implement proper water management",
            "Remove infected plant debris"
        ],
        "management": [
            "Apply appropriate fungicides",
            "Optimize nitrogen application",
            "Maintain proper water levels",
            "Harvest at proper maturity",
            "Consider early planting"
        ],
        "supportive_actions": [
            "Regular field monitoring",
            "Weather-based disease forecasting",
            "Proper water management",
            "Document disease patterns",
            "Implement integrated pest management"
        ]
    },
    # Sugarcane Diseases
    "Bacterial_Blight": {
        "description": "Yellow to white streaks on leaves with bacterial ooze",
        "preventive_measures": [
            "Use disease-free planting material",
            "Practice proper field sanitation",
            "Implement crop rotation",
            "Maintain proper drainage",
            "Remove infected plants"
        ],
        "management": [
            "Apply recommended bactericides",
            "Optimize irrigation practices",
            "Remove infected plant parts",
            "Implement proper harvesting practices",
            "Consider resistant varieties"
        ],
        "supportive_actions": [
            "Regular field inspection",
            "Proper equipment sanitation",
            "Document disease patterns",
            "Consider biological control",
            "Implement proper irrigation timing"
        ]
    },
    "Red_Rot": {
        "description": "Red discoloration of internal tissues with characteristic odor",
        "preventive_measures": [
            "Use disease-free planting material",
            "Practice proper field sanitation",
            "Implement crop rotation",
            "Maintain proper drainage",
            "Remove infected plants"
        ],
        "management": [
            "Apply recommended fungicides",
            "Optimize irrigation practices",
            "Remove infected plant parts",
            "Implement proper harvesting practices",
            "Consider resistant varieties"
        ],
        "supportive_actions": [
            "Regular field inspection",
            "Proper equipment sanitation",
            "Document disease patterns",
            "Consider biological control",
            "Implement proper irrigation timing"
        ]
    },
    # Wheat Diseases
    "Leaf_Rust": {
        "description": "Orange to brown pustules on leaves",
        "preventive_measures": [
            "Plant resistant varieties",
            "Practice proper crop rotation",
            "Maintain optimal plant density",
            "Remove volunteer plants",
            "Implement proper field sanitation"
        ],
        "management": [
            "Apply recommended fungicides",
            "Optimize nitrogen application",
            "Remove infected plant debris",
            "Consider early planting",
            "Implement proper irrigation practices"
        ],
        "supportive_actions": [
            "Regular field monitoring",
            "Document disease patterns",
            "Maintain field hygiene",
            "Consider biological control",
            "Implement proper irrigation timing"
        ]
    },
    "Septoria": {
        "description": "Brown spots with black pycnidia on leaves",
        "preventive_measures": [
            "Use disease-free seeds",
            "Practice proper crop rotation",
            "Maintain optimal plant density",
            "Remove infected plant debris",
            "Implement proper field sanitation"
        ],
        "management": [
            "Apply recommended fungicides",
            "Optimize nitrogen application",
            "Remove infected plant debris",
            "Consider early planting",
            "Implement proper irrigation practices"
        ],
        "supportive_actions": [
            "Regular field monitoring",
            "Document disease patterns",
            "Maintain field hygiene",
            "Consider biological control",
            "Implement proper irrigation timing"
        ]
    },
    "Yellow_Rust": {
        "description": "Yellow to orange pustules on leaves",
        "preventive_measures": [
            "Plant resistant varieties",
            "Practice proper crop rotation",
            "Maintain optimal plant density",
            "Remove volunteer plants",
            "Implement proper field sanitation"
        ],
        "management": [
            "Apply recommended fungicides",
            "Optimize nitrogen application",
            "Remove infected plant debris",
            "Consider early planting",
            "Implement proper irrigation practices"
        ],
        "supportive_actions": [
            "Regular field monitoring",
            "Document disease patterns",
            "Maintain field hygiene",
            "Consider biological control",
            "Implement proper irrigation timing"
        ]
    }
}

# Define image transforms
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def format_class_name(class_name):
    """Format the class name for display"""
    parts = class_name.split('___')
    crop = parts[0]
    condition = parts[1].replace('_', ' ')
    return f"{crop} - {condition}"

def load_model():
    """Load the pre-trained model"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = CropDiseaseModel(num_classes=17)
    model_path = os.path.join('outputs', 'five_crop', 'best_model.pth')
    
    if os.path.exists(model_path):
        try:
            # Load the state dict
            checkpoint = torch.load(model_path, map_location=device)
            
            # Load the state dict into the model with strict=False
            model.load_state_dict(checkpoint, strict=False)
            model = model.to(device)
            model.eval()
            return model, device, True
        except Exception as e:
            st.error(f"Error loading model: {str(e)}")
            return None, device, False
    else:
        st.error(f"Model file not found at {model_path}")
        return None, device, False

def predict_image(image, model, device):
    """Make a prediction for the input image"""
    # Convert image to RGB (in case it's RGBA or another format)
    image = image.convert('RGB')
    
    # Apply transformations
    image_tensor = transform(image).unsqueeze(0).to(device)
    
    # Get predictions
    with torch.no_grad():
        outputs = model(image_tensor)
        probabilities = torch.softmax(outputs, dim=1)
    
    # Get top 5 predictions
    top5_prob, top5_indices = torch.topk(probabilities, 5)
    
    # Convert to lists
    top5_prob = top5_prob[0].cpu().numpy()
    top5_indices = top5_indices[0].cpu().numpy()
    
    # Get class names
    top5_classes = [CLASS_NAMES[idx] for idx in top5_indices]
    
    return top5_classes, top5_prob

def main():
    st.set_page_config(
        page_title="Crop Disease Detector",
        page_icon="🌱",
        layout="wide"
    )
    
    st.title("Crop Disease Detection")
    st.write("Upload an image of a crop to detect diseases")
    
    # Sidebar with information
    st.sidebar.title("About")
    st.sidebar.info(
        "This application uses a deep learning model to detect diseases in crops. "
        "The model is trained on the following crops:\n"
        "- Corn\n"
        "- Potato\n"
        "- Rice\n"
        "- Sugarcane\n"
        "- Wheat\n\n"
        "Upload an image of one of these crops to get predictions."
    )
    
    # Load model
    model, device, model_loaded = load_model()
    
    # File uploader
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
    
    # Two columns layout
    col1, col2 = st.columns(2)
    
    if uploaded_file is not None:
        # Display the uploaded image
        image = Image.open(uploaded_file)
        col1.header("Uploaded Image")
        col1.image(image, use_container_width=True)
        
        # Make prediction
        if model_loaded:
            with st.spinner("Analyzing image..."):
                top5_classes, top5_prob = predict_image(image, model, device)
            
            # Display results
            col2.header("Prediction Results")
            
            # Create a bar chart
            fig, ax = plt.subplots()
            y_pos = np.arange(len(top5_classes))
            confidence_values = top5_prob * 100
            
            ax.barh(y_pos, confidence_values, align='center')
            ax.set_yticks(y_pos)
            ax.set_yticklabels([format_class_name(cls) for cls in top5_classes])
            ax.invert_yaxis()  # Labels read top-to-bottom
            ax.set_xlabel('Confidence (%)')
            ax.set_title('Disease Prediction')
            
            # Add confidence percentage to the end of each bar
            for i, v in enumerate(confidence_values):
                ax.text(v + 1, i, f"{v:.1f}%", va='center')
            
            col2.pyplot(fig)
            
            # Also display as text
            col2.subheader("Top Prediction:")
            col2.markdown(f"**{format_class_name(top5_classes[0])}** with {top5_prob[0]*100:.2f}% confidence")
            
            # Display details about the disease
            col2.subheader("Disease Information:")
            primary_disease = top5_classes[0].split('___')[1]
            if "Healthy" in primary_disease:
                col2.success("Your crop appears to be healthy!")
                col2.markdown("**Maintain these healthy practices:**")
                col2.markdown(
                    "1. Continue regular monitoring\n"
                    "2. Maintain proper irrigation and fertilization\n"
                    "3. Practice crop rotation\n"
                    "4. Keep field hygiene\n"
                    "5. Use disease-free seeds"
                )
            else:
                disease_info = DISEASE_INFO.get(primary_disease, {})
                if disease_info:
                    col2.warning(f"Your crop may have {primary_disease.replace('_', ' ')}.")
                    col2.markdown(f"**Description:** {disease_info['description']}")
                    
                    col2.markdown("**Preventive Measures:**")
                    for measure in disease_info['preventive_measures']:
                        col2.markdown(f"- {measure}")
                    
                    col2.markdown("**Management Strategies:**")
                    for strategy in disease_info['management']:
                        col2.markdown(f"- {strategy}")
                    
                    col2.markdown("**Supportive Actions:**")
                    for action in disease_info['supportive_actions']:
                        col2.markdown(f"- {action}")
                    
                    col2.markdown("**Additional Recommendations:**")
                    col2.markdown(
                        "1. Consult with local agricultural experts\n"
                        "2. Keep detailed records of disease patterns\n"
                        "3. Consider integrated pest management\n"
                        "4. Monitor weather conditions\n"
                        "5. Implement proper field sanitation"
                    )
                else:
                    col2.warning(f"Your crop may have {primary_disease.replace('_', ' ')}.")
                    col2.markdown("**General Recommendations:**")
                    col2.markdown(
                        "1. Consult with a local agricultural expert\n"
                        "2. Consider appropriate fungicides/pesticides\n"
                        "3. Isolate affected plants if possible\n"
                        "4. Maintain proper field hygiene\n"
                        "5. Document disease patterns for future reference"
                    )
        else:
            col2.warning("Running in demo mode - model not loaded")
            col2.info("Note: In demo mode, predictions are not based on the actual model")

if __name__ == "__main__":
    main() 