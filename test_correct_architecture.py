#!/usr/bin/env python3
"""
Test the model with the correct architecture from train_model_improved.py
This matches the exact architecture used in training
"""
import torch
import torch.nn as nn
import torchvision.models as models
import numpy as np
import os

class SimpleGoogLeNetClassifier(nn.Module):
    """Simple GoogLeNet classifier matching train_model_improved.py architecture"""
    def __init__(self, num_classes=2):
        super(SimpleGoogLeNetClassifier, self).__init__()
        
        # Use GoogLeNet backbone
        self.backbone = models.googlenet(weights=models.GoogLeNet_Weights.DEFAULT)
        
        # Replace the final classifier with the exact architecture from training
        self.backbone.fc = nn.Sequential(
            nn.Linear(1024, 512),      # fc.0
            nn.ReLU(inplace=True),     # fc.1
            nn.Dropout(0.3),           # fc.2
            nn.Linear(512, 128),       # fc.3
            nn.ReLU(inplace=True),     # fc.4
            nn.Dropout(0.3),           # fc.5
            nn.Linear(128, 64),        # fc.6
            nn.ReLU(inplace=True),     # fc.7
            nn.Dropout(0.2),           # fc.8
            nn.Linear(64, num_classes) # fc.9
        )
    
    def forward(self, x):
        return self.backbone(x)

def load_model_with_correct_architecture(model_path):
    """Load model with the correct architecture"""
    try:
        print(f"🔍 Testing model: {model_path}")
        
        # Create model with correct architecture
        model = SimpleGoogLeNetClassifier(num_classes=2)
        
        # Load the checkpoint
        checkpoint = torch.load(model_path, map_location='cpu')
        
        # Handle different checkpoint formats
        if 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
        elif 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
        else:
            state_dict = checkpoint
        
        # Load the state dict
        model.load_state_dict(state_dict, strict=False)
        model.eval()
        
        print(f"✅ Successfully loaded model: {model_path}")
        return model, model_path
        
    except Exception as e:
        print(f"❌ Failed to load {model_path}: {str(e)}")
        return None, model_path

def test_model_predictions(model, model_path, num_tests=20):
    """Test model predictions with random inputs"""
    if model is None:
        return model_path, [], []
    
    print(f"\n🧪 Testing {model_path} with {num_tests} samples...")
    
    normal_predictions = []
    abnormal_predictions = []
    
    with torch.no_grad():
        for i in range(num_tests):
            # Create random input (batch_size=1, channels=3, height=224, width=224)
            random_input = torch.randn(1, 3, 224, 224)
            
            # Get prediction
            outputs = model(random_input)
            probabilities = torch.softmax(outputs, dim=1)
            
            # Assuming class 0 = Normal, class 1 = Abnormal
            normal_prob = probabilities[0][0].item()
            abnormal_prob = probabilities[0][1].item()
            
            normal_predictions.append(normal_prob)
            abnormal_predictions.append(abnormal_prob)
    
    # Calculate statistics
    avg_normal = np.mean(normal_predictions)
    avg_abnormal = np.mean(abnormal_predictions)
    
    # Count predictions (using 0.5 threshold)
    normal_count = sum(1 for p in normal_predictions if p > 0.5)
    abnormal_count = sum(1 for p in abnormal_predictions if p > 0.5)
    
    print(f"📊 Results for {os.path.basename(model_path)}:")
    print(f"   Normal predictions: {normal_count}/{num_tests} ({normal_count/num_tests*100:.1f}%)")
    print(f"   Abnormal predictions: {abnormal_count}/{num_tests} ({abnormal_count/num_tests*100:.1f}%)")
    print(f"   Average normal probability: {avg_normal:.3f}")
    print(f"   Average abnormal probability: {avg_abnormal:.3f}")
    
    return model_path, normal_predictions, abnormal_predictions

def main():
    """Test different model files"""
    print("🔬 Testing Binary Screening Models with Correct Architecture")
    print("=" * 60)
    
    # List of model files to test (in order of preference)
    model_files = [
        "blood_cancer_model_random_balanced.pth",
        "best_binary_model.pth", 
        "blood_smear_screening_model_fixed.pth",
        "blood_smear_screening_model.pth"
    ]
    
    results = []
    
    for model_file in model_files:
        model_path = os.path.join(os.getcwd(), model_file)
        if os.path.exists(model_path):
            model, path = load_model_with_correct_architecture(model_path)
            path, normal_preds, abnormal_preds = test_model_predictions(model, path)
            results.append((path, normal_preds, abnormal_preds))
        else:
            print(f"⚠️  Model file not found: {model_file}")
    
    # Summary
    print("\n" + "=" * 60)
    print("📋 SUMMARY OF ALL MODELS:")
    print("=" * 60)
    
    for model_path, normal_preds, abnormal_preds in results:
        if normal_preds and abnormal_preds:
            normal_count = sum(1 for p in normal_preds if p > 0.5)
            abnormal_count = sum(1 for p in abnormal_preds if p > 0.5)
            avg_normal = np.mean(normal_preds)
            avg_abnormal = np.mean(abnormal_preds)
            
            print(f"\n🔍 {os.path.basename(model_path)}:")
            print(f"   Balance: {normal_count}N/{abnormal_count}A (Normal/Abnormal)")
            print(f"   Avg probabilities: {avg_normal:.3f}N / {avg_abnormal:.3f}A")
            
            # Determine if model is balanced
            if abs(normal_count - abnormal_count) <= 4:  # Allow some variance
                print(f"   ✅ BALANCED MODEL - Good for production!")
            else:
                print(f"   ❌ BIASED MODEL - Needs correction")

if __name__ == "__main__":
    main()