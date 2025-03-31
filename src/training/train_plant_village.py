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
from torch.amp import autocast, GradScaler

# Add the parent directory to the path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.crop_model import CropDiseaseModel
from data.plant_village_dataset import PlantVillageDataset
from utils.visualization import plot_training_curves, plot_confusion_matrix

# Set memory efficient settings for maximum speed
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.deterministic = False
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.enabled = True

# Force CUDA initialization with maximum memory usage
if torch.cuda.is_available():
    torch.cuda.empty_cache()
    torch.cuda.set_per_process_memory_fraction(0.95)  # Use maximum available memory
    torch.cuda.set_device(0)
    stream = torch.cuda.Stream()
    torch.cuda.set_stream(stream)

# Set random seed for reproducibility
torch.manual_seed(42)
np.random.seed(42)

# Define data transforms with optimized augmentation
train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
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
    model = model.to(device)
    scaler = GradScaler()
    
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=0.01,
        epochs=num_epochs,
        steps_per_epoch=len(train_loader),
        pct_start=0.3,
        anneal_strategy='cos'
    )
    
    train_stream = torch.cuda.Stream()
    
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        train_pbar = tqdm(train_loader, desc=f'Epoch [{epoch+1}/{num_epochs}] Training')
        
        with torch.cuda.stream(train_stream):
            for inputs, labels in train_pbar:
                inputs = inputs.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True)
                
                optimizer.zero_grad(set_to_none=True)
                
                with autocast(device_type='cuda', dtype=torch.float16):
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                
                scheduler.step()
                
                train_loss += loss.item()
                _, predicted = outputs.max(1)
                train_total += labels.size(0)
                train_correct += predicted.eq(labels).sum().item()
                
                train_pbar.set_postfix({
                    'loss': f'{loss.item():.4f}',
                    'acc': f'{100.*train_correct/train_total:.2f}%',
                    'lr': f'{scheduler.get_last_lr()[0]:.6f}'
                })
                
                # Clear memory periodically
                if train_pbar.n % 50 == 0:
                    torch.cuda.empty_cache()
        
        torch.cuda.synchronize()
        
        train_loss = train_loss / len(train_loader)
        train_acc = 100. * train_correct / train_total
        
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        val_pbar = tqdm(val_loader, desc='Validation')
        with torch.no_grad(), autocast(device_type='cuda', dtype=torch.float16):
            for inputs, labels in val_pbar:
                inputs = inputs.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                
                val_loss += loss.item()
                _, predicted = outputs.max(1)
                val_total += labels.size(0)
                val_correct += predicted.eq(labels).sum().item()
                
                val_pbar.set_postfix({
                    'loss': f'{loss.item():.4f}',
                    'acc': f'{100.*val_correct/val_total:.2f}%'
                })
        
        val_loss = val_loss / len(val_loader)
        val_acc = 100. * val_correct / val_total
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'best_val_acc': best_val_acc,
            }, 'outputs/plant_village/best_model.pth')
            print(f'\nNew best model saved with validation accuracy: {val_acc:.2f}%')
        
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        
        print(f'\nEpoch [{epoch+1}/{num_epochs}] Summary:')
        print(f'Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%')
        print(f'Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%')
        print('-' * 60)
        
        # Clear memory after each epoch
        torch.cuda.empty_cache()
    
    return history

def evaluate_model(model, test_loader, device):
    """Evaluate the model on test data"""
    model.eval()
    all_preds = []
    all_labels = []
    
    eval_pbar = tqdm(test_loader, desc='Evaluating')
    with torch.no_grad(), autocast(device_type='cuda', dtype=torch.float16):
        for inputs, labels in eval_pbar:
            inputs = inputs.to(device, non_blocking=True)
            outputs = model(inputs)
            _, predicted = outputs.max(1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.numpy())
    
    return all_preds, all_labels

def main():
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
    
    os.makedirs('outputs/plant_village', exist_ok=True)
    
    print('Loading datasets...')
    train_dataset = PlantVillageDataset('data/plant_village', transform=train_transform, split='train')
    val_dataset = PlantVillageDataset('data/plant_village', transform=val_transform, split='valid')
    
    print(f'Training dataset size: {len(train_dataset)}')
    print(f'Validation dataset size: {len(val_dataset)}')
    
    class_dist = {}
    for label in train_dataset.labels:
        class_name = train_dataset.get_class_name(label)
        class_dist[class_name] = class_dist.get(class_name, 0) + 1
    
    print("\nClass Distribution:")
    for cls, count in class_dist.items():
        print(f"{cls}: {count}")
    
    # Optimized data loading for D drive
    train_loader = DataLoader(
        train_dataset, 
        batch_size=128,  # Balanced batch size for speed and memory
        shuffle=True, 
        num_workers=4,  # Balanced number of workers
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=2,
        drop_last=True
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=128,  # Balanced batch size for speed and memory
        shuffle=False, 
        num_workers=4,  # Balanced number of workers
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=2
    )
    
    print('Initializing model...')
    model = CropDiseaseModel(num_classes=len(train_dataset.classes))
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(
        model.parameters(),
        lr=0.001,
        weight_decay=0.01,
        betas=(0.9, 0.999),
        eps=1e-8
    )
    
    print('Starting training...')
    history = train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs=50, device=device)
    
    print('Generating training curves...')
    plot_training_curves(history, 'outputs/plant_village/training_curves.png')
    
    print('Evaluating model...')
    all_preds, all_labels = evaluate_model(model, val_loader, device)
    
    print('Generating confusion matrix...')
    cm = confusion_matrix(all_labels, all_preds)
    plot_confusion_matrix(cm, 'outputs/plant_village/confusion_matrix.png')
    
    print('\nClassification Report:')
    print(classification_report(all_labels, all_preds, target_names=train_dataset.classes))

if __name__ == '__main__':
    main() 