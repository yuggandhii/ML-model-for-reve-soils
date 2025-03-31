import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
import numpy as np
from tqdm import tqdm

# Add the parent directory to the path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.crop_model import CropDiseaseModel
from data.five_crop_dataset import FiveCropDataset
from utils.visualization import plot_training_curves, plot_confusion_matrix

# Force CUDA initialization
if torch.cuda.is_available():
    torch.cuda.empty_cache()
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False

# Set random seed for reproducibility
torch.manual_seed(42)
np.random.seed(42)

# Define data transforms
train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

val_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs, device):
    """Train the model and return training history"""
    history = {
        'train_loss': [], 'train_acc': [],
        'val_loss': [], 'val_acc': []
    }
    
    best_val_acc = 0.0
    
    # Move model to GPU
    model = model.to(device)
    
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        # Create progress bar for training
        train_pbar = tqdm(train_loader, desc=f'Epoch [{epoch+1}/{num_epochs}] Training')
        for inputs, labels in train_pbar:
            # Move data to GPU
            inputs = inputs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            _, predicted = outputs.max(1)
            train_total += labels.size(0)
            train_correct += predicted.eq(labels).sum().item()
            
            # Update progress bar
            train_pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'acc': f'{100.*train_correct/train_total:.2f}%'
            })
        
        train_loss = train_loss / len(train_loader)
        train_acc = 100. * train_correct / train_total
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        # Create progress bar for validation
        val_pbar = tqdm(val_loader, desc='Validation')
        with torch.no_grad():
            for inputs, labels in val_pbar:
                # Move data to GPU
                inputs = inputs.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                
                val_loss += loss.item()
                _, predicted = outputs.max(1)
                val_total += labels.size(0)
                val_correct += predicted.eq(labels).sum().item()
                
                # Update progress bar
                val_pbar.set_postfix({
                    'loss': f'{loss.item():.4f}',
                    'acc': f'{100.*val_correct/val_total:.2f}%'
                })
        
        val_loss = val_loss / len(val_loader)
        val_acc = 100. * val_correct / val_total
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), 'outputs/five_crop/best_model.pth')
            print(f'\nNew best model saved with validation accuracy: {val_acc:.2f}%')
        
        # Update history
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        
        print(f'\nEpoch [{epoch+1}/{num_epochs}] Summary:')
        print(f'Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%')
        print(f'Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%')
        print('-' * 60)
    
    return history

def evaluate_model(model, test_loader, device):
    """Evaluate the model on test data"""
    model.eval()
    all_preds = []
    all_labels = []
    
    # Create progress bar for evaluation
    eval_pbar = tqdm(test_loader, desc='Evaluating')
    with torch.no_grad():
        for inputs, labels in eval_pbar:
            # Move data to GPU
            inputs = inputs.to(device, non_blocking=True)
            outputs = model(inputs)
            _, predicted = outputs.max(1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.numpy())
    
    return all_preds, all_labels

def main():
    # Set device and check CUDA availability
    if not torch.cuda.is_available():
        print("CUDA is not available. Please check your GPU installation.")
        return
    
    device = torch.device('cuda')
    print(f'Using device: {device}')
    print(f'GPU: {torch.cuda.get_device_name(0)}')
    print(f'CUDA Version: {torch.version.cuda}')
    print(f'GPU Memory Usage:')
    print(f'Allocated: {torch.cuda.memory_allocated(0)/1024**2:.2f} MB')
    print(f'Cached: {torch.cuda.memory_reserved(0)/1024**2:.2f} MB')
    
    # Create output directory
    os.makedirs('outputs/five_crop', exist_ok=True)
    
    # Load dataset
    print('Loading dataset...')
    dataset = FiveCropDataset('../data/five_crop_diseases', transform=train_transform)
    print(f'Dataset size: {len(dataset)}')
    
    # Print class distribution
    class_dist = {}
    for label in dataset.labels:
        class_name = dataset.get_class_name(label)
        class_dist[class_name] = class_dist.get(class_name, 0) + 1
    
    print("\nClass Distribution:")
    for cls, count in class_dist.items():
        print(f"{cls}: {count}")
    
    # Split dataset
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
    
    # Create data loaders with more workers for faster loading
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False, num_workers=4, pin_memory=True)
    
    # Initialize model
    print('Initializing model...')
    model = CropDiseaseModel(num_classes=len(dataset.classes))
    
    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # Train model
    print('Starting training...')
    history = train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs=50, device=device)
    
    # Plot training curves
    print('Generating training curves...')
    plot_training_curves(history, 'outputs/five_crop/training_curves.png')
    
    # Evaluate model
    print('Evaluating model...')
    all_preds, all_labels = evaluate_model(model, val_loader, device)
    
    # Plot confusion matrix
    print('Generating confusion matrix...')
    cm = confusion_matrix(all_labels, all_preds)
    plot_confusion_matrix(cm, 'outputs/five_crop/confusion_matrix.png')
    
    # Print classification report
    print('\nClassification Report:')
    print(classification_report(all_labels, all_preds, target_names=dataset.classes))

if __name__ == '__main__':
    main() 