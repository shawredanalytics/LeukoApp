#!/usr/bin/env python3
"""
Improved Blood Cancer Model Training Script
Addresses CLL bias issue with proper class balancing and regularization
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from torchvision import transforms, models
import pandas as pd
import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from sklearn.utils.class_weight import compute_class_weight
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from collections import Counter
import warnings
warnings.filterwarnings('ignore')

class BloodCancerDataset(Dataset):
    """Enhanced dataset with better data handling"""
    
    def __init__(self, image_paths, labels, transform=None, class_names=None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform
        self.class_names = class_names or ["ALL", "AML", "CLL", "CML"]
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        try:
            image_path = self.image_paths[idx]
            image = Image.open(image_path).convert('RGB')
            label = self.labels[idx]
            
            if self.transform:
                image = self.transform(image)
            
            return image, label
        except Exception as e:
            print(f"Error loading image {self.image_paths[idx]}: {e}")
            # Return a black image as fallback
            if self.transform:
                return self.transform(Image.new('RGB', (224, 224), (0, 0, 0))), self.labels[idx]
            return Image.new('RGB', (224, 224), (0, 0, 0)), self.labels[idx]

class ImprovedBloodCancerTrainer:
    """Improved trainer with class balancing and bias prevention"""
    
    def __init__(self, model_type="googlenet", num_classes=4, device=None):
        self.model_type = model_type
        self.num_classes = num_classes
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None
        self.class_names = ["ALL", "AML", "CLL", "CML"]
        
        print(f"🔧 Using device: {self.device}")
        print(f"🎯 Training for {num_classes} classes: {self.class_names}")
    
    def create_model(self):
        """Create model with proper architecture matching saved weights"""
        print("🤖 Creating GoogLeNet model with improved architecture...")
        
        self.model = models.googlenet(weights=models.GoogLeNet_Weights.DEFAULT)
        
        # Use the exact architecture that matches the saved weights
        # This prevents architecture mismatch issues
        self.model.fc = nn.Sequential(
            nn.Linear(1024, 512),      # fc.0
            nn.ReLU(inplace=True),     # fc.1
            nn.Dropout(0.3),           # fc.2 - increased dropout
            nn.Linear(512, 128),       # fc.3
            nn.ReLU(inplace=True),     # fc.4
            nn.Dropout(0.3),           # fc.5 - increased dropout
            nn.Linear(128, 64),        # fc.6
            nn.ReLU(inplace=True),     # fc.7
            nn.Dropout(0.2),           # fc.8
            nn.Linear(64, self.num_classes)  # fc.9
        )
        
        self.model.to(self.device)
        return self.model
    
    def get_transforms(self):
        """Enhanced data augmentation to prevent overfitting"""
        train_transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.RandomCrop(224),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.3),
            transforms.RandomRotation(degrees=15),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
            transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        val_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        return train_transform, val_transform
    
    def analyze_class_distribution(self, labels):
        """Analyze and print class distribution"""
        class_counts = Counter(labels)
        total_samples = len(labels)
        
        print("\n📊 Class Distribution Analysis:")
        print("-" * 40)
        for class_idx, class_name in enumerate(self.class_names):
            count = class_counts.get(class_idx, 0)
            percentage = (count / total_samples) * 100
            print(f"{class_name}: {count:,} samples ({percentage:.1f}%)")
        
        # Check for severe imbalance
        max_count = max(class_counts.values())
        min_count = min(class_counts.values())
        imbalance_ratio = max_count / min_count if min_count > 0 else float('inf')
        
        if imbalance_ratio > 5:
            print(f"⚠️  Severe class imbalance detected! Ratio: {imbalance_ratio:.1f}:1")
            print("   Applying class balancing techniques...")
        
        return class_counts
    
    def create_balanced_sampler(self, labels):
        """Create weighted sampler for balanced training"""
        class_counts = Counter(labels)
        total_samples = len(labels)
        
        # Calculate weights for each class (inverse frequency)
        class_weights = {}
        for class_idx in range(self.num_classes):
            count = class_counts.get(class_idx, 1)  # Avoid division by zero
            class_weights[class_idx] = total_samples / (self.num_classes * count)
        
        # Create sample weights
        sample_weights = [class_weights[label] for label in labels]
        
        print("\n⚖️  Class Weights for Balanced Sampling:")
        for class_idx, class_name in enumerate(self.class_names):
            print(f"{class_name}: {class_weights[class_idx]:.3f}")
        
        return WeightedRandomSampler(
            weights=sample_weights,
            num_samples=len(sample_weights),
            replacement=True
        )
    
    def prepare_data(self, dataset_path, class_mapping=None):
        """Prepare balanced training and validation data"""
        print("📊 Preparing dataset with class balancing...")
        
        # 4-class mapping (excluding Normal)
        if class_mapping is None:
            class_mapping = {
                "all": 0,   # Acute Lymphoblastic Leukemia
                "aml": 1,   # Acute Myeloid Leukemia
                "cll": 2,   # Chronic Lymphocytic Leukemia
                "cml": 3    # Chronic Myeloid Leukemia
            }
        
        image_paths = []
        labels = []
        
        # Scan dataset directory
        for class_name, class_idx in class_mapping.items():
            class_dir = os.path.join(dataset_path, class_name)
            if os.path.exists(class_dir):
                class_images = []
                for img_file in os.listdir(class_dir):
                    if img_file.lower().endswith(('.png', '.jpg', '.jpeg')):
                        class_images.append(os.path.join(class_dir, img_file))
                
                # Limit samples per class to prevent extreme imbalance
                max_samples_per_class = 2000  # Adjust based on your data
                if len(class_images) > max_samples_per_class:
                    class_images = np.random.choice(class_images, max_samples_per_class, replace=False)
                
                image_paths.extend(class_images)
                labels.extend([class_idx] * len(class_images))
                
                print(f"Found {len(class_images)} images for {class_name.upper()}")
        
        print(f"📈 Total: {len(image_paths)} images across {len(class_mapping)} classes")
        
        # Analyze class distribution
        class_counts = self.analyze_class_distribution(labels)
        
        # Split data with stratification
        train_paths, val_paths, train_labels, val_labels = train_test_split(
            image_paths, labels, test_size=0.2, random_state=42, stratify=labels
        )
        
        # Get transforms
        train_transform, val_transform = self.get_transforms()
        
        # Create datasets
        train_dataset = BloodCancerDataset(train_paths, train_labels, train_transform, self.class_names)
        val_dataset = BloodCancerDataset(val_paths, val_labels, val_transform, self.class_names)
        
        # Create balanced sampler for training
        train_sampler = self.create_balanced_sampler(train_labels)
        
        # Create data loaders
        train_loader = DataLoader(
            train_dataset, 
            batch_size=16,  # Reduced batch size for better gradient updates
            sampler=train_sampler,  # Use balanced sampler
            num_workers=2,
            pin_memory=True
        )
        
        val_loader = DataLoader(
            val_dataset, 
            batch_size=16, 
            shuffle=False, 
            num_workers=2,
            pin_memory=True
        )
        
        return train_loader, val_loader, class_counts
    
    def create_weighted_loss(self, class_counts):
        """Create weighted loss function to handle class imbalance"""
        # Calculate class weights
        total_samples = sum(class_counts.values())
        class_weights = []
        
        for class_idx in range(self.num_classes):
            count = class_counts.get(class_idx, 1)
            weight = total_samples / (self.num_classes * count)
            class_weights.append(weight)
        
        class_weights = torch.FloatTensor(class_weights).to(self.device)
        
        print("\n⚖️  Loss Function Class Weights:")
        for class_idx, class_name in enumerate(self.class_names):
            print(f"{class_name}: {class_weights[class_idx]:.3f}")
        
        return nn.CrossEntropyLoss(weight=class_weights)
    
    def train_model(self, train_loader, val_loader, class_counts, epochs=30, lr=0.0001):
        """Train model with improved methodology"""
        print(f"🚀 Starting improved training for {epochs} epochs...")
        
        # Create weighted loss function
        criterion = self.create_weighted_loss(class_counts)
        
        # Optimizer with weight decay for regularization
        optimizer = optim.AdamW(
            self.model.parameters(), 
            lr=lr, 
            weight_decay=0.01,  # Strong regularization
            betas=(0.9, 0.999)
        )
        
        # Learning rate scheduler
        scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer, T_0=10, T_mult=2, eta_min=1e-6
        )
        
        best_val_f1 = 0.0
        best_val_acc = 0.0
        patience = 10
        patience_counter = 0
        
        train_losses = []
        val_accuracies = []
        val_f1_scores = []
        
        for epoch in range(epochs):
            # Training phase
            self.model.train()
            running_loss = 0.0
            train_correct = 0
            train_total = 0
            
            train_pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} [Train]")
            for batch_idx, (images, labels) in enumerate(train_pbar):
                images, labels = images.to(self.device), labels.to(self.device)
                
                optimizer.zero_grad()
                
                outputs = self.model(images)
                loss = criterion(outputs, labels)
                
                loss.backward()
                
                # Gradient clipping to prevent exploding gradients
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                
                optimizer.step()
                
                running_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                train_total += labels.size(0)
                train_correct += (predicted == labels).sum().item()
                
                train_pbar.set_postfix({
                    'Loss': f'{loss.item():.4f}',
                    'Acc': f'{100.*train_correct/train_total:.1f}%'
                })
            
            avg_train_loss = running_loss / len(train_loader)
            train_accuracy = 100 * train_correct / train_total
            train_losses.append(avg_train_loss)
            
            # Validation phase
            self.model.eval()
            val_correct = 0
            val_total = 0
            val_loss = 0.0
            all_predictions = []
            all_labels = []
            
            with torch.no_grad():
                val_pbar = tqdm(val_loader, desc=f"Epoch {epoch+1}/{epochs} [Val]")
                for images, labels in val_pbar:
                    images, labels = images.to(self.device), labels.to(self.device)
                    
                    outputs = self.model(images)
                    loss = criterion(outputs, labels)
                    
                    val_loss += loss.item()
                    _, predicted = torch.max(outputs.data, 1)
                    val_total += labels.size(0)
                    val_correct += (predicted == labels).sum().item()
                    
                    all_predictions.extend(predicted.cpu().numpy())
                    all_labels.extend(labels.cpu().numpy())
            
            val_accuracy = 100 * val_correct / val_total
            avg_val_loss = val_loss / len(val_loader)
            val_f1 = f1_score(all_labels, all_predictions, average='weighted')
            
            val_accuracies.append(val_accuracy)
            val_f1_scores.append(val_f1)
            
            # Learning rate scheduling
            scheduler.step()
            
            print(f"\nEpoch {epoch+1}/{epochs}:")
            print(f"  Train Loss: {avg_train_loss:.4f} | Train Acc: {train_accuracy:.2f}%")
            print(f"  Val Loss: {avg_val_loss:.4f} | Val Acc: {val_accuracy:.2f}%")
            print(f"  Val F1-Score: {val_f1:.4f}")
            print(f"  Learning Rate: {optimizer.param_groups[0]['lr']:.6f}")
            
            # Print per-class accuracy
            if epoch % 5 == 0:  # Every 5 epochs
                print("\n  Per-class Performance:")
                for class_idx, class_name in enumerate(self.class_names):
                    class_mask = np.array(all_labels) == class_idx
                    if class_mask.sum() > 0:
                        class_acc = (np.array(all_predictions)[class_mask] == class_idx).mean() * 100
                        print(f"    {class_name}: {class_acc:.1f}%")
            
            # Save best model based on F1-score (better for imbalanced data)
            if val_f1 > best_val_f1:
                best_val_f1 = val_f1
                best_val_acc = val_accuracy
                torch.save(self.model.state_dict(), 'blood_cancer_model_improved.pth')
                print(f"  ✅ New best model saved! F1: {best_val_f1:.4f}, Acc: {best_val_acc:.2f}%")
                patience_counter = 0
            else:
                patience_counter += 1
            
            # Early stopping
            if patience_counter >= patience:
                print(f"\n⏹️  Early stopping triggered after {patience} epochs without improvement")
                break
            
            print("-" * 60)
        
        print(f"\n🎉 Training completed!")
        print(f"📊 Best F1-Score: {best_val_f1:.4f}")
        print(f"📊 Best Accuracy: {best_val_acc:.2f}%")
        
        return train_losses, val_accuracies, val_f1_scores

def main():
    """Main training function"""
    print("🩸 Improved Blood Cancer Model Training")
    print("=" * 50)
    
    # Configuration
    MODEL_TYPE = "googlenet"
    NUM_CLASSES = 4  # ALL, AML, CLL, CML (excluding Normal)
    EPOCHS = 30
    LEARNING_RATE = 0.0001
    
    # Dataset path (adjust as needed)
    dataset_path = "./kaggle_data/blood-cancer-dataset"
    
    if not os.path.exists(dataset_path):
        print(f"❌ Dataset not found at {dataset_path}")
        print("Please download the dataset first using the Kaggle API")
        return
    
    # Initialize trainer
    trainer = ImprovedBloodCancerTrainer(
        model_type=MODEL_TYPE,
        num_classes=NUM_CLASSES
    )
    
    # Create model
    model = trainer.create_model()
    
    # Prepare data with class balancing
    train_loader, val_loader, class_counts = trainer.prepare_data(dataset_path)
    
    # Train model with improved methodology
    train_losses, val_accuracies, val_f1_scores = trainer.train_model(
        train_loader, val_loader, class_counts,
        epochs=EPOCHS, 
        lr=LEARNING_RATE
    )
    
    # Plot training history
    plt.figure(figsize=(15, 5))
    
    plt.subplot(1, 3, 1)
    plt.plot(train_losses)
    plt.title('Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.grid(True)
    
    plt.subplot(1, 3, 2)
    plt.plot(val_accuracies)
    plt.title('Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.grid(True)
    
    plt.subplot(1, 3, 3)
    plt.plot(val_f1_scores)
    plt.title('Validation F1-Score')
    plt.xlabel('Epoch')
    plt.ylabel('F1-Score')
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig('improved_training_history.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("\n🎉 Improved training pipeline completed successfully!")
    print("📊 Training history saved as 'improved_training_history.png'")
    print("💾 Best model saved as 'blood_cancer_model_improved.pth'")
    print("\n🔄 To use the improved model, replace 'blood_cancer_model.pth' with 'blood_cancer_model_improved.pth'")

if __name__ == "__main__":
    main()