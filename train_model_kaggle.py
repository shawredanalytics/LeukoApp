#!/usr/bin/env python3
"""
Blood Cancer Model Training Script using Kaggle Datasets
Supports multiple architectures: ViT-Large, GoogLeNet, ResNet, EfficientNet
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms, models
from transformers import ViTForImageClassification, ViTImageProcessor
import pandas as pd
import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import kaggle
import zipfile
import shutil

class BloodCancerDataset(Dataset):
    """Custom dataset for blood cancer images"""
    
    def __init__(self, image_paths, labels, transform=None, class_names=None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform
        self.class_names = class_names or ["Normal", "ALL", "AML", "CLL", "CML"]
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        image = Image.open(image_path).convert('RGB')
        label = self.labels[idx]
        
        if self.transform:
            image = self.transform(image)
        
        return image, label

class KaggleDatasetDownloader:
    """Download and prepare Kaggle datasets for training"""
    
    def __init__(self, data_dir="./kaggle_data"):
        self.data_dir = data_dir
        os.makedirs(data_dir, exist_ok=True)
    
    def download_blood_cells_dataset(self):
        """Download blood cells dataset from Kaggle"""
        try:
            print("📥 Downloading Blood Cells Dataset...")
            kaggle.api.dataset_download_files(
                'paultimothymooney/blood-cells',
                path=self.data_dir,
                unzip=True
            )
            print("✅ Blood Cells Dataset downloaded successfully!")
            return os.path.join(self.data_dir, "blood-cells")
        except Exception as e:
            print(f"❌ Error downloading dataset: {e}")
            return None
    
    def download_leukemia_dataset(self):
        """Download leukemia classification dataset"""
        try:
            print("📥 Downloading Leukemia Classification Dataset...")
            kaggle.api.dataset_download_files(
                'andrewmvd/leukemia-classification',
                path=self.data_dir,
                unzip=True
            )
            print("✅ Leukemia Dataset downloaded successfully!")
            return os.path.join(self.data_dir, "leukemia-classification")
        except Exception as e:
            print(f"❌ Error downloading dataset: {e}")
            return None
    
    def download_blood_cancer_dataset(self):
        """Download multi-class blood cancer dataset"""
        try:
            print("📥 Downloading Blood Cancer Dataset...")
            kaggle.api.dataset_download_files(
                'mohammadamireshraghi/blood-cancer-dataset',
                path=self.data_dir,
                unzip=True
            )
            print("✅ Blood Cancer Dataset downloaded successfully!")
            return os.path.join(self.data_dir, "blood-cancer-dataset")
        except Exception as e:
            print(f"❌ Error downloading dataset: {e}")
            return None

class BloodCancerTrainer:
    """Main training class for blood cancer detection models"""
    
    def __init__(self, model_type="googlenet", num_classes=5, device=None):
        self.model_type = model_type
        self.num_classes = num_classes
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None
        self.processor = None
        
        print(f"🔧 Using device: {self.device}")
        print(f"🎯 Training for {num_classes} classes")
    
    def create_model(self):
        """Create and initialize the model"""
        if self.model_type == "vit-large":
            print("🤖 Creating ViT-Large model...")
            self.model = ViTForImageClassification.from_pretrained(
                "google/vit-large-patch16-224",
                num_labels=self.num_classes,
                ignore_mismatched_sizes=True
            )
            self.processor = ViTImageProcessor.from_pretrained("google/vit-large-patch16-224")
            
        elif self.model_type == "googlenet":
            print("🤖 Creating GoogLeNet model...")
            self.model = models.googlenet(weights=models.GoogLeNet_Weights.DEFAULT)
            # Enhanced architecture
            self.model.fc = nn.Sequential(
                nn.Dropout(0.4),
                nn.Linear(1024, 512),
                nn.BatchNorm1d(512),
                nn.ReLU(inplace=True),
                nn.Dropout(0.2),
                nn.Linear(512, 256),
                nn.BatchNorm1d(256),
                nn.ReLU(inplace=True),
                nn.Dropout(0.1),
                nn.Linear(256, self.num_classes)
            )
            
        elif self.model_type == "resnet50":
            print("🤖 Creating ResNet-50 model...")
            self.model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
            self.model.fc = nn.Linear(2048, self.num_classes)
            
        elif self.model_type == "efficientnet":
            print("🤖 Creating EfficientNet-B0 model...")
            self.model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT)
            self.model.classifier[1] = nn.Linear(1280, self.num_classes)
        
        self.model.to(self.device)
        return self.model
    
    def get_transforms(self):
        """Get data augmentation transforms"""
        if self.model_type == "vit-large":
            # ViT uses its own processor
            train_transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.RandomHorizontalFlip(0.5),
                transforms.RandomRotation(10),
                transforms.ColorJitter(brightness=0.2, contrast=0.2),
                transforms.ToTensor(),
            ])
        else:
            # Standard transforms for CNN models
            train_transform = transforms.Compose([
                transforms.Resize(256),
                transforms.RandomCrop(224),
                transforms.RandomHorizontalFlip(0.5),
                transforms.RandomRotation(15),
                transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
        
        val_transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        return train_transform, val_transform
    
    def prepare_data(self, dataset_path, class_mapping=None):
        """Prepare training and validation data"""
        print("📊 Preparing dataset...")
        
        # Default class mapping for 5-class classification
        if class_mapping is None:
            class_mapping = {
                "normal": 0,
                "all": 1,  # Acute Lymphoblastic Leukemia
                "aml": 2,  # Acute Myeloid Leukemia
                "cll": 3,  # Chronic Lymphocytic Leukemia
                "cml": 4   # Chronic Myeloid Leukemia
            }
        
        image_paths = []
        labels = []
        
        # Scan dataset directory
        for class_name, class_idx in class_mapping.items():
            class_dir = os.path.join(dataset_path, class_name)
            if os.path.exists(class_dir):
                for img_file in os.listdir(class_dir):
                    if img_file.lower().endswith(('.png', '.jpg', '.jpeg')):
                        image_paths.append(os.path.join(class_dir, img_file))
                        labels.append(class_idx)
        
        print(f"📈 Found {len(image_paths)} images across {len(class_mapping)} classes")
        
        # Split data
        train_paths, val_paths, train_labels, val_labels = train_test_split(
            image_paths, labels, test_size=0.2, random_state=42, stratify=labels
        )
        
        # Get transforms
        train_transform, val_transform = self.get_transforms()
        
        # Create datasets
        train_dataset = BloodCancerDataset(train_paths, train_labels, train_transform)
        val_dataset = BloodCancerDataset(val_paths, val_labels, val_transform)
        
        # Create data loaders
        train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4)
        val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=4)
        
        return train_loader, val_loader
    
    def train_model(self, train_loader, val_loader, epochs=50, lr=0.001):
        """Train the model"""
        print(f"🚀 Starting training for {epochs} epochs...")
        
        # Loss function and optimizer
        criterion = nn.CrossEntropyLoss()
        
        if self.model_type == "vit-large":
            optimizer = optim.AdamW(self.model.parameters(), lr=lr, weight_decay=0.01)
        else:
            optimizer = optim.Adam(self.model.parameters(), lr=lr, weight_decay=0.0001)
        
        # Learning rate scheduler
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=5, verbose=True
        )
        
        best_val_acc = 0.0
        train_losses = []
        val_accuracies = []
        
        for epoch in range(epochs):
            # Training phase
            self.model.train()
            running_loss = 0.0
            
            train_pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} [Train]")
            for images, labels in train_pbar:
                images, labels = images.to(self.device), labels.to(self.device)
                
                optimizer.zero_grad()
                
                if self.model_type == "vit-large":
                    outputs = self.model(images)
                    loss = criterion(outputs.logits, labels)
                else:
                    outputs = self.model(images)
                    loss = criterion(outputs, labels)
                
                loss.backward()
                optimizer.step()
                
                running_loss += loss.item()
                train_pbar.set_postfix({'Loss': f'{loss.item():.4f}'})
            
            avg_train_loss = running_loss / len(train_loader)
            train_losses.append(avg_train_loss)
            
            # Validation phase
            self.model.eval()
            correct = 0
            total = 0
            val_loss = 0.0
            
            with torch.no_grad():
                val_pbar = tqdm(val_loader, desc=f"Epoch {epoch+1}/{epochs} [Val]")
                for images, labels in val_pbar:
                    images, labels = images.to(self.device), labels.to(self.device)
                    
                    if self.model_type == "vit-large":
                        outputs = self.model(images)
                        loss = criterion(outputs.logits, labels)
                        _, predicted = torch.max(outputs.logits.data, 1)
                    else:
                        outputs = self.model(images)
                        loss = criterion(outputs, labels)
                        _, predicted = torch.max(outputs.data, 1)
                    
                    val_loss += loss.item()
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()
            
            val_accuracy = 100 * correct / total
            avg_val_loss = val_loss / len(val_loader)
            val_accuracies.append(val_accuracy)
            
            print(f"Epoch {epoch+1}/{epochs}:")
            print(f"  Train Loss: {avg_train_loss:.4f}")
            print(f"  Val Loss: {avg_val_loss:.4f}")
            print(f"  Val Accuracy: {val_accuracy:.2f}%")
            
            # Save best model
            if val_accuracy > best_val_acc:
                best_val_acc = val_accuracy
                torch.save(self.model.state_dict(), 'best_blood_cancer_model.pth')
                print(f"  ✅ New best model saved! Accuracy: {best_val_acc:.2f}%")
            
            # Learning rate scheduling
            scheduler.step(avg_val_loss)
            
            print("-" * 50)
        
        print(f"🎉 Training completed! Best validation accuracy: {best_val_acc:.2f}%")
        return train_losses, val_accuracies

def main():
    """Main training pipeline"""
    print("🩸 Blood Cancer Detection Model Training")
    print("=" * 50)
    
    # Configuration
    MODEL_TYPE = "googlenet"  # Options: "vit-large", "googlenet", "resnet50", "efficientnet"
    NUM_CLASSES = 5
    EPOCHS = 50
    LEARNING_RATE = 0.001
    
    # Initialize Kaggle API (requires kaggle.json in ~/.kaggle/)
    try:
        kaggle.api.authenticate()
        print("✅ Kaggle API authenticated successfully!")
    except Exception as e:
        print(f"❌ Kaggle API authentication failed: {e}")
        print("Please ensure kaggle.json is in ~/.kaggle/ directory")
        return
    
    # Download datasets
    downloader = KaggleDatasetDownloader()
    
    # Choose which dataset to use
    print("\n📥 Available datasets:")
    print("1. Blood Cells Dataset (paultimothymooney/blood-cells)")
    print("2. Leukemia Classification (andrewmvd/leukemia-classification)")
    print("3. Blood Cancer Dataset (mohammadamireshraghi/blood-cancer-dataset)")
    
    choice = input("Enter dataset choice (1-3): ").strip()
    
    if choice == "1":
        dataset_path = downloader.download_blood_cells_dataset()
    elif choice == "2":
        dataset_path = downloader.download_leukemia_dataset()
    elif choice == "3":
        dataset_path = downloader.download_blood_cancer_dataset()
    else:
        print("❌ Invalid choice!")
        return
    
    if dataset_path is None:
        print("❌ Failed to download dataset!")
        return
    
    # Initialize trainer
    trainer = BloodCancerTrainer(
        model_type=MODEL_TYPE,
        num_classes=NUM_CLASSES
    )
    
    # Create model
    model = trainer.create_model()
    
    # Prepare data
    train_loader, val_loader = trainer.prepare_data(dataset_path)
    
    # Train model
    train_losses, val_accuracies = trainer.train_model(
        train_loader, val_loader, 
        epochs=EPOCHS, 
        lr=LEARNING_RATE
    )
    
    # Plot training history
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(train_losses)
    plt.title('Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    
    plt.subplot(1, 2, 2)
    plt.plot(val_accuracies)
    plt.title('Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    
    plt.tight_layout()
    plt.savefig('training_history.png')
    plt.show()
    
    print("🎉 Training pipeline completed successfully!")
    print("📊 Training history saved as 'training_history.png'")
    print("💾 Best model saved as 'best_blood_cancer_model.pth'")

if __name__ == "__main__":
    main()