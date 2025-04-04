# Crop Disease Detection System

A comprehensive machine learning system for detecting and classifying crop diseases from images. The system integrates multiple datasets, provides training scripts, and includes a web interface for making predictions.

## Features

- **Multiple Dataset Support**: Trained on various crop disease datasets including:
  - Five Crop Diseases Dataset (rice, wheat, maize, sugarcane, cotton)
  - Plant Village Dataset
  - Crop Diseases Classification Dataset
  - Plant 622 Dataset
  - Pathogen Dataset

- **Deep Learning Model**: Uses ResNet50 with transfer learning and custom classification layers.

- **Web Interface**: Upload images or provide image URLs to get disease predictions.

- **Visualization**: Generate confusion matrices, training curves, and classification reports.

## Installation

### Clone the Repository
```bash
git clone <repository-url>
cd crop-disease-detection
```

### Install Dependencies
```bash
pip install -r requirements.txt
```

### Set Up Kaggle API Credentials
1. Go to your Kaggle account settings.
2. Create a new API token (this will download `kaggle.json`).
3. Place the `kaggle.json` file in `~/.kaggle/` directory.
4. Make it readable only by you:
```bash
chmod 600 ~/.kaggle/kaggle.json
```

## Usage

### Download Datasets and Train Models
```bash
python src/train_all.py
```
This will:
1. Download all necessary datasets from Kaggle.
2. Train models on each dataset.
3. Generate visualizations and save the best models.

### Run the Web Application
```bash
python src/web/app.py
```
Then visit `http://localhost:5000` in your browser to use the web interface.

## Project Structure
```
├── src/
│   ├── data/          # Dataset classes and data loading utilities
│   ├── models/        # Model architecture definitions
│   ├── training/      # Training scripts for each dataset
│   ├── web/           # Web application for prediction
├── data/              # Downloaded datasets (created during setup)
├── outputs/           # Trained models and evaluation results
```

## Models

The system trains several models specialized for different crop types and diseases:

1. **Five Crop Model**: Detects diseases in rice, wheat, maize, sugarcane, and cotton.
2. **Plant Village Model**: Classifies among 38 different plant diseases.
3. **Crop Diseases Model**: Specialized for crop disease classification.
4. **Plant 622 Model**: Recognizes 622 different plant species.
5. **Pathogen Model**: Identifies 3 different types of pathogens.

## API Documentation

The web application provides a simple REST API:

### Predict Crop Disease
**Endpoint:** `POST /predict`

**Parameters:**
- `model`: Model name to use for prediction.
- `image`: Image file upload.
- `url`: URL to an image.

**Example Request:**
```bash
curl -X POST "http://localhost:5000/predict" \
     -F "model=Five Crop Model" \
     -F "image=@path/to/image.jpg"
```

This system aims to enhance early detection of crop diseases, helping farmers take timely action and improve yield quality.


 
