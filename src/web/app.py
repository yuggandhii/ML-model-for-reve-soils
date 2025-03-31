from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import torch
from torchvision import transforms
from PIL import Image
import os
import sys
import random

# Add the parent directory to the path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.crop_model import CropDiseaseModel

app = Flask(__name__)
CORS(app)

# Configure upload folder
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'JPG', 'JPEG', 'PNG'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# Create upload folder if it doesn't exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Load the model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = CropDiseaseModel(num_classes=17)  # 17 classes for 5-crop dataset
model_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 'outputs', 'five_crop', 'best_model.pth')

# Check if model file exists
USE_MODEL = os.path.exists(model_path)
if USE_MODEL:
    try:
        # Load the state dict
        checkpoint = torch.load(model_path, map_location=device)
        
        # Load the state dict into the model, with strict=False to ignore missing keys
        model.load_state_dict(checkpoint, strict=False)
        model = model.to(device)
        model.eval()
        print(f"Model loaded successfully from {model_path}")
    except Exception as e:
        print(f"Error loading model: {str(e)}")
        USE_MODEL = False
else:
    print(f"Model file not found at {model_path}")
    print("Running in demo mode with random predictions")

# Define image transforms
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Define class names for the 5-crop dataset
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

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return jsonify({'error': 'No image provided'}), 400
    
    file = request.files['image']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    
    if file and allowed_file(file.filename):
        try:
            if USE_MODEL:
                # Use the trained model for predictions
                image = Image.open(file).convert('RGB')
                image_tensor = transform(image).unsqueeze(0).to(device)
                
                with torch.no_grad():
                    outputs = model(image_tensor)
                    probabilities = torch.softmax(outputs, dim=1)
                    confidence, predicted = torch.max(probabilities, 1)
                    
                class_name = CLASS_NAMES[predicted.item()]
                confidence = confidence.item() * 100
                
                top3_prob, top3_indices = torch.topk(probabilities, 3)
                top3_predictions = [
                    {
                        'class': CLASS_NAMES[idx],
                        'confidence': prob * 100
                    }
                    for prob, idx in zip(top3_prob[0], top3_indices[0])
                ]
            else:
                # Demo mode: Return random predictions
                primary_confidence = random.uniform(85, 98)
                second_confidence = random.uniform(60, 85)
                third_confidence = random.uniform(40, 60)
                
                selected_classes = random.sample(CLASS_NAMES, 3)
                class_name = selected_classes[0]
                confidence = primary_confidence
                
                top3_predictions = [
                    {
                        'class': selected_classes[0],
                        'confidence': primary_confidence
                    },
                    {
                        'class': selected_classes[1],
                        'confidence': second_confidence
                    },
                    {
                        'class': selected_classes[2],
                        'confidence': third_confidence
                    }
                ]
            
            return jsonify({
                'success': True,
                'prediction': class_name,
                'confidence': confidence,
                'top3_predictions': top3_predictions,
                'demo_mode': not USE_MODEL
            })
            
        except Exception as e:
            return jsonify({'error': str(e)}), 500
    
    return jsonify({'error': 'Invalid file type'}), 400

if __name__ == '__main__':
    print("Server is running at http://localhost:5000")
    print(f"Running in {'demo' if not USE_MODEL else 'model'} mode")
    app.run(debug=True, port=5000) 