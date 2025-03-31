import torch
import torch.nn as nn
import numpy as np
from sklearn.preprocessing import StandardScaler

class SoilAnalysisModel(nn.Module):
    def __init__(self, input_dim, hidden_dims=[256, 128, 64], output_dim=7):
        """
        Initialize the soil analysis model
        Args:
            input_dim: Dimension of input features (spectrometer + sensor data)
            hidden_dims: List of hidden layer dimensions
            output_dim: Dimension of output (sensor measurements)
        """
        super(SoilAnalysisModel, self).__init__()
        
        # Build the network layers
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
                nn.BatchNorm1d(hidden_dim),
                nn.Dropout(0.3)
            ])
            prev_dim = hidden_dim
            
        # Output layer
        layers.append(nn.Linear(prev_dim, output_dim))
        
        self.network = nn.Sequential(*layers)
        self.scaler = StandardScaler()
        
    def forward(self, x):
        return self.network(x)
    
    def fit_scaler(self, features):
        """Fit the scaler on the input features"""
        self.scaler.fit(features)
        
    def transform_features(self, features):
        """Transform features using the fitted scaler"""
        return self.scaler.transform(features)
    
    def predict_sensor_values(self, spectra_data):
        """
        Predict sensor values from spectrometer data
        Args:
            spectra_data: Spectrometer measurements
        Returns:
            Predicted sensor values
        """
        self.eval()
        with torch.no_grad():
            # Scale the input data
            scaled_data = self.scaler.transform(spectra_data)
            # Convert to tensor
            tensor_data = torch.FloatTensor(scaled_data)
            # Make prediction
            predictions = self(tensor_data)
            return predictions.numpy()
    
    def get_feature_importance(self, spectra_data, sensor_data):
        """
        Calculate feature importance using permutation importance
        Args:
            spectra_data: Spectrometer measurements
            sensor_data: Actual sensor values
        Returns:
            Dictionary of feature importance scores
        """
        self.eval()
        base_score = self._calculate_score(spectra_data, sensor_data)
        
        importance_scores = {}
        n_features = spectra_data.shape[1]
        
        for i in range(n_features):
            # Create a copy of the data
            perturbed_data = spectra_data.copy()
            # Shuffle the i-th feature
            np.random.shuffle(perturbed_data[:, i])
            # Calculate new score
            new_score = self._calculate_score(perturbed_data, sensor_data)
            # Calculate importance
            importance = base_score - new_score
            importance_scores[f'Feature_{i}'] = importance
            
        return importance_scores
    
    def _calculate_score(self, spectra_data, sensor_data):
        """Calculate prediction score"""
        predictions = self.predict_sensor_values(spectra_data)
        return np.mean(np.square(predictions - sensor_data)) 