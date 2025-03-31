import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report
import pandas as pd
from pathlib import Path
import os

from ..data.five_crop_dataset import FiveCropDataset
from ..data.mendeley_dataset import MendeleyDataset
from ..models.crop_model import CropDiseaseModel

def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs, device, dataset_name):
    best_val_acc = 0.0
    train_losses = []
    val_losses = []
    train_accs = []
    val_accs = []
    
    # Create output directory
    output_dir = Path(f'outputs/{dataset_name}')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        train_preds = []
        train_labels = []
        
        for batch in tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs}'):
            images = batch['image'].to(device)
            labels = batch['label'].to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            train_preds.extend(predicted.cpu().numpy())
            train_labels.extend(labels.cpu().numpy())
        
        train_acc = np.mean(np.array(train_preds) == np.array(train_labels))
        avg_train_loss = train_loss / len(train_loader)
        train_losses.append(avg_train_loss)
        train_accs.append(train_acc)
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        val_preds = []
        val_labels = []
        
        with torch.no_grad():
            for batch in val_loader:
                images = batch['image'].to(device)
                labels = batch['label'].to(device)
                
                outputs = model(images)
                loss = criterion(outputs, labels)
                
                val_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                val_preds.extend(predicted.cpu().numpy())
                val_labels.extend(labels.cpu().numpy())
        
        val_acc = np.mean(np.array(val_preds) == np.array(val_labels))
        avg_val_loss = val_loss / len(val_loader)
        val_losses.append(avg_val_loss)
        val_accs.append(val_acc)
        
        print(f'Epoch {epoch+1}/{num_epochs}:')
        print(f'Training Loss: {avg_train_loss:.4f}, Training Accuracy: {train_acc:.4f}')
        print(f'Validation Loss: {avg_val_loss:.4f}, Validation Accuracy: {val_acc:.4f}')
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), output_dir / 'best_model.pth')
            print(f'New best model saved with validation accuracy: {val_acc:.4f}')
    
    return train_losses, val_losses, train_accs, val_accs, output_dir

def plot_training_history(train_losses, val_losses, train_accs, val_accs, output_dir):
    # Plot losses
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.title('Training History - Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    # Plot accuracies
    plt.subplot(1, 2, 2)
    plt.plot(train_accs, label='Training Accuracy')
    plt.plot(val_accs, label='Validation Accuracy')
    plt.title('Training History - Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(output_dir / 'training_history.png')
    plt.close()

def plot_confusion_matrix(y_true, y_pred, classes, output_dir):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(15, 12))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=classes, yticklabels=classes)
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(output_dir / 'confusion_matrix.png')
    plt.close()

def train_dataset(dataset_name, dataset_class, data_path, num_classes):
    """Train model on a specific dataset"""
    print(f"\nTraining on {dataset_name} dataset...")
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    
    # Load dataset
    dataset = dataset_class(data_path)
    
    # Print dataset statistics
    class_dist = dataset.get_class_distribution()
    print("\nClass Distribution:")
    for cls, count in class_dist.items():
        print(f"{cls}: {count}")
    
    # Split dataset
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=4)
    
    # Initialize model
    model = CropDiseaseModel(num_classes=num_classes).to(device)
    
    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # Train the model
    train_losses, val_losses, train_accs, val_accs, output_dir = train_model(
        model, train_loader, val_loader, criterion, optimizer, 
        num_epochs=50, device=device, dataset_name=dataset_name
    )
    
    # Plot training history
    plot_training_history(train_losses, val_losses, train_accs, val_accs, output_dir)
    
    # Evaluate on validation set
    model.eval()
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for batch in val_loader:
            images = batch['image'].to(device)
            labels = batch['label']
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.numpy())
    
    # Get class names for confusion matrix
    class_names = [dataset.get_class_name(i) for i in range(num_classes)]
    
    # Plot confusion matrix
    plot_confusion_matrix(all_labels, all_preds, class_names, output_dir)
    
    # Generate classification report
    report = classification_report(all_labels, all_preds, target_names=class_names)
    with open(output_dir / 'classification_report.txt', 'w') as f:
        f.write(report)

def main():
    """Main function to train on all datasets"""
    # Define datasets to train on
    datasets = {
        'five_crop': {
            'class': FiveCropDataset,
            'path': 'data/five_crop_diseases',
            'num_classes': 20  # 5 crops * 4 states (healthy + 3 diseases)
        },
        'mendeley': {
            'class': MendeleyDataset,
            'path': 'data/mendeley_dataset',
            'num_classes': 2  # healthy vs diseased
        }
    }
    
    # Train on each dataset
    for name, config in datasets.items():
        try:
            train_dataset(name, config['class'], config['path'], config['num_classes'])
        except Exception as e:
            print(f"Error training on {name} dataset: {str(e)}")

if __name__ == '__main__':
    main() 