import pandas as pd
import numpy as np
from torch.utils.data import Dataset
import torch

class SoilDataset(Dataset):
    def __init__(self, excel_path, transform=None):
        """
        Initialize the soil dataset
        Args:
            excel_path: Path to the Excel file containing soil data
            transform: Optional transform to be applied on the data
        """
        self.transform = transform
        self.data = pd.read_excel(excel_path)
        
        # Group data by soil sample and moisture level
        self.samples = self._preprocess_data()
        
    def _preprocess_data(self):
        """
        Preprocess the data by:
        1. Grouping by soil sample and moisture level
        2. Calculating mean/median for repeated measurements
        3. Normalizing the data
        """
        # Group by soil sample and moisture level
        grouped = self.data.groupby(['Soil_Sample', 'Moisture_Level'])
        
        processed_samples = []
        for (sample, moisture), group in grouped:
            # Calculate mean for repeated measurements
            mean_spectra = group.filter(like='Wavelength').mean()
            mean_sensors = group[['Moisture', 'EC', 'Temperature', 'N', 'P', 'K', 'PH']].mean()
            
            # Combine features
            features = np.concatenate([mean_spectra.values, mean_sensors.values])
            
            processed_samples.append({
                'features': features,
                'soil_sample': sample,
                'moisture_level': moisture,
                'sensor_data': mean_sensors,
                'spectra_data': mean_spectra
            })
            
        return processed_samples
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        
        # Convert to tensor
        features = torch.FloatTensor(sample['features'])
        
        if self.transform:
            features = self.transform(features)
            
        return {
            'features': features,
            'soil_sample': sample['soil_sample'],
            'moisture_level': sample['moisture_level'],
            'sensor_data': sample['sensor_data'],
            'spectra_data': sample['spectra_data']
        }
    
    def get_correlation_matrix(self):
        """
        Calculate correlation matrix between spectrometer and sensor data
        """
        # Extract all features
        features = np.array([sample['features'] for sample in self.samples])
        
        # Create feature names
        n_spectra = len(self.samples[0]['spectra_data'])
        n_sensors = len(self.samples[0]['sensor_data'])
        
        feature_names = (
            [f'Wavelength_{i}' for i in range(n_spectra)] +
            list(self.samples[0]['sensor_data'].index)
        )
        
        # Calculate correlation matrix
        corr_matrix = pd.DataFrame(
            features,
            columns=feature_names
        ).corr()
        
        return corr_matrix 