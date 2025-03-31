import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import numpy as np
import pandas as pd

from ..data.soil_dataset import SoilDataset
from ..models.soil_model import SoilAnalysisModel

def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs, device):
    best_val_loss = float('inf')
    train_losses = []
    val_losses = []
    
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        for batch in tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs}'):
            features = batch['features'].to(device)
            sensor_data = batch['sensor_data'].values.astype(np.float32)
            sensor_data = torch.FloatTensor(sensor_data).to(device)
            
            optimizer.zero_grad()
            outputs = model(features)
            loss = criterion(outputs, sensor_data)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
        
        avg_train_loss = train_loss / len(train_loader)
        train_losses.append(avg_train_loss)
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in val_loader:
                features = batch['features'].to(device)
                sensor_data = batch['sensor_data'].values.astype(np.float32)
                sensor_data = torch.FloatTensor(sensor_data).to(device)
                
                outputs = model(features)
                loss = criterion(outputs, sensor_data)
                val_loss += loss.item()
        
        avg_val_loss = val_loss / len(val_loader)
        val_losses.append(avg_val_loss)
        
        print(f'Epoch {epoch+1}/{num_epochs}:')
        print(f'Training Loss: {avg_train_loss:.4f}')
        print(f'Validation Loss: {avg_val_loss:.4f}')
        
        # Save best model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), 'best_soil_model.pth')
            print(f'New best model saved with validation loss: {avg_val_loss:.4f}')
    
    return train_losses, val_losses

def plot_training_history(train_losses, val_losses):
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.title('Training History')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig('training_history.png')
    plt.close()

def plot_correlation_matrix(corr_matrix):
    plt.figure(figsize=(12, 10))
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0)
    plt.title('Feature Correlation Matrix')
    plt.tight_layout()
    plt.savefig('correlation_matrix.png')
    plt.close()

def main():
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    
    # Load dataset
    dataset = SoilDataset('soildataset.xlsx')
    
    # Split dataset
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    
    # Initialize model
    input_dim = len(dataset.samples[0]['features'])
    model = SoilAnalysisModel(input_dim=input_dim).to(device)
    
    # Define loss function and optimizer
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # Train the model
    train_losses, val_losses = train_model(
        model, train_loader, val_loader, criterion, optimizer, 
        num_epochs=50, device=device
    )
    
    # Plot training history
    plot_training_history(train_losses, val_losses)
    
    # Calculate and plot correlation matrix
    corr_matrix = dataset.get_correlation_matrix()
    plot_correlation_matrix(corr_matrix)
    
    # Calculate feature importance
    importance_scores = model.get_feature_importance(
        np.array([sample['features'] for sample in dataset.samples]),
        np.array([sample['sensor_data'].values for sample in dataset.samples])
    )
    
    # Save feature importance
    pd.DataFrame(importance_scores.items(), columns=['Feature', 'Importance']).to_csv(
        'feature_importance.csv', index=False
    )

if __name__ == '__main__':
    main() 