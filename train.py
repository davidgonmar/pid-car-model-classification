import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from torchvision import transforms, models
from torchvision.datasets import StanfordCars
from tqdm import tqdm
import matplotlib.pyplot as plt
import os
from kaggle.api.kaggle_api_extended import KaggleApi


def setup_kaggle_credentials():
    """Setup Kaggle API credentials"""
    api = KaggleApi()
    api.authenticate()
    return api


def download_dataset(api, dataset_path):
    """Download the Stanford Cars dataset from Kaggle if not already present"""
    # Check if dataset already exists
    if os.path.exists(os.path.join(dataset_path, 'cars_train')):
        print("Dataset already exists, skipping download...")
        return

    print("Downloading Stanford Cars dataset...")
    api.dataset_download_files(
        'rickyyyyyyy/torchvision-stanford-cars',
        path=dataset_path,
        unzip=True
    )
    print("Dataset downloaded successfully!")


def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs=10, device='cuda'):
    """
    Train the model.
    Args:
        model: The neural network model
        train_loader: DataLoader for training data
        val_loader: DataLoader for validation data
        criterion: Loss function
        optimizer: Optimizer
        num_epochs: Number of epochs to train
        device: Device to train on ('cuda' or 'cpu')
    """
    model = model.to(device)
    best_val_acc = 0.0
    
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        for inputs, labels in tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs}'):
            inputs, labels = inputs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
        
        train_loss = running_loss / len(train_loader)
        train_acc = 100. * correct / total
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                
                val_loss += loss.item()
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()
        
        val_loss = val_loss / len(val_loader)
        val_acc = 100. * correct / total
        
        print(f'Epoch {epoch+1}/{num_epochs}:')
        print(f'Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%')
        print(f'Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}%')
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), 'models/best_model.pth')


def main():
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Create necessary directories
    os.makedirs('models', exist_ok=True)
    dataset_path = "data"
    os.makedirs(dataset_path, exist_ok=True)

    # Setup Kaggle API and download dataset
    api = setup_kaggle_credentials()
    download_dataset(api, dataset_path)

    # Define transforms
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                           std=[0.229, 0.224, 0.225])
    ])

    # Load Stanford Cars dataset
    train_dataset = StanfordCars(
        root=dataset_path, split='train', transform=transform, download=False
    )
    test_dataset = StanfordCars(
        root=dataset_path, split='test', transform=transform, download=False
    )

    # Create subset of test data for validation
    subset_size = 5000
    indices = torch.randperm(len(test_dataset))[:subset_size]
    val_dataset = Subset(test_dataset, indices)
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

    # Initialize model
    model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
    model.fc = nn.Linear(model.fc.in_features, 196)  # 196 classes in Stanford Cars
    model = model.to(device)

    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Train the model
    train_model(model, train_loader, val_loader, criterion, optimizer, 
                num_epochs=10, device=device)


if __name__ == '__main__':
    main()
