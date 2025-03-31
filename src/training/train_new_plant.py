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
from torch.cuda.amp import autocast, GradScaler

# Add the parent directory to the path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.crop_model import CropDiseaseModel
from data.new_plant_dataset import NewPlantDataset
from utils.visualization import plot_training_curves, plot_confusion_matrix

# Force CUDA initialization
if torch.cuda.is_available():
    torch.cuda.empty_cache()
    # Enable cuDNN benchmarking for faster training
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False
    # Enable TF32 for better performance on Ampere GPUs
    torch.backends.cuda.matmul.allow_tf32 = True
    # Set CUDA device to use all available memory
    torch.cuda.set_per_process_memory_fraction(0.95)
    # Enable asynchronous GPU operations
    torch.cuda.set_device(0)
    # Set CUDA stream for better performance
    stream = torch.cuda.Stream()
    torch.cuda.set_stream(stream)

# Set random seed for reproducibility
torch.manual_seed(42)
np.random.seed(42)

# Define data transforms with optimized augmentation
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
    
    # Initialize gradient scaler for mixed precision training
    scaler = GradScaler()
    
    # Use OneCycleLR scheduler for faster convergence
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=0.01,
        epochs=num_epochs,
        steps_per_epoch=len(train_loader),
        pct_start=0.3,
        anneal_strategy='cos'  # Use cosine annealing for better convergence
    )
    
    # Create CUDA stream for training
    train_stream = torch.cuda.Stream()
    
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        # Create progress bar for training
        train_pbar = tqdm(train_loader, desc=f'Epoch [{epoch+1}/{num_epochs}] Training')
        
        # Use CUDA stream for training
        with torch.cuda.stream(train_stream):
            for inputs, labels in train_pbar:
                # Move data to GPU asynchronously
                inputs = inputs.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True)
                
                optimizer.zero_grad(set_to_none=True)  # More efficient than zero_grad()
                
                # Use mixed precision training
                with autocast():
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                
                # Scale loss and backpropagate
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                
                # Update learning rate
                scheduler.step()
                
                train_loss += loss.item()
                _, predicted = outputs.max(1)
                train_total += labels.size(0)
                train_correct += predicted.eq(labels).sum().item()
                
                # Update progress bar
                train_pbar.set_postfix({
                    'loss': f'{loss.item():.4f}',
                    'acc': f'{100.*train_correct/train_total:.2f}%',
                    'lr': f'{scheduler.get_last_lr()[0]:.6f}'
                })
        
        # Synchronize CUDA stream
        torch.cuda.synchronize()
        
        train_loss = train_loss / len(train_loader)
        train_acc = 100. * train_correct / train_total
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        # Create progress bar for validation
        val_pbar = tqdm(val_loader, desc='Validation')
        with torch.no_grad(), autocast():
            for inputs, labels in val_pbar:
                # Move data to GPU asynchronously
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
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'best_val_acc': best_val_acc,
            }, 'outputs/new_plant/best_model.pth')
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
    with torch.no_grad(), autocast():
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
    os.makedirs('outputs/new_plant', exist_ok=True)
    
    # Load datasets
    print('Loading datasets...')
    train_dataset = NewPlantDataset('data/new_plant_diseases', transform=train_transform, split='train')
    val_dataset = NewPlantDataset('data/new_plant_diseases', transform=val_transform, split='valid')
    
    print(f'Training dataset size: {len(train_dataset)}')
    print(f'Validation dataset size: {len(val_dataset)}')
    
    # Print class distribution
    class_dist = {}
    for label in train_dataset.labels:
        class_name = train_dataset.get_class_name(label)
        class_dist[class_name] = class_dist.get(class_name, 0) + 1
    
    print("\nClass Distribution:")
    for cls, count in class_dist.items():
        print(f"{cls}: {count}")
    
    # Create data loaders with optimized settings
    train_loader = DataLoader(
        train_dataset, 
        batch_size=256,  # Increased batch size for better GPU utilization
        shuffle=True, 
        num_workers=8,  # Increased number of workers
        pin_memory=True,
        persistent_workers=True,  # Keep workers alive between epochs
        prefetch_factor=2,  # Prefetch 2 batches per worker
        drop_last=True  # Drop incomplete batches for more stable training
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=256,  # Increased batch size for better GPU utilization
        shuffle=False, 
        num_workers=8,  # Increased number of workers
        pin_memory=True,
        persistent_workers=True,  # Keep workers alive between epochs
        prefetch_factor=2  # Prefetch 2 batches per worker
    )
    
    # Initialize model
    print('Initializing model...')
    model = CropDiseaseModel(num_classes=len(train_dataset.classes))
    
    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(  # Using AdamW optimizer
        model.parameters(),
        lr=0.001,
        weight_decay=0.01,  # L2 regularization
        betas=(0.9, 0.999),  # Optimized beta parameters
        eps=1e-8  # Optimized epsilon
    )
    
    # Train model
    print('Starting training...')
    history = train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs=50, device=device)
    
    # Plot training curves
    print('Generating training curves...')
    plot_training_curves(history, 'outputs/new_plant/training_curves.png')
    
    # Evaluate model
    print('Evaluating model...')
    all_preds, all_labels = evaluate_model(model, val_loader, device)
    
    # Plot confusion matrix
    print('Generating confusion matrix...')
    cm = confusion_matrix(all_labels, all_preds)
    plot_confusion_matrix(cm, 'outputs/new_plant/confusion_matrix.png')
    
    # Print classification report
    print('\nClassification Report:')
    print(classification_report(all_labels, all_preds, target_names=train_dataset.classes))

if __name__ == '__main__':
    main() 