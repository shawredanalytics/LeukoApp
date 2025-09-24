#!/usr/bin/env python3
"""
Debug script to test the discriminative model predictions
"""

import torch
import torch.nn as nn
from torchvision import transforms
from torchvision.models import googlenet, GoogLeNet_Weights
import numpy as np
from PIL import Image
import os
import json

# Define the DiscriminativeClassifier class (same as in app.py)
class DiscriminativeClassifier(nn.Module):
    """Enhanced classifier with attention mechanism for better discrimination"""
    
    def __init__(self, input_features, num_classes):
        super(DiscriminativeClassifier, self).__init__()
        self.num_classes = num_classes
        
        # Multi-branch architecture for better feature learning
        self.feature_extractor = nn.Sequential(
            nn.Linear(input_features, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
        )
        
        # Attention mechanism for important feature selection
        self.attention = nn.Sequential(
            nn.Linear(512, 256),
            nn.Tanh(),
            nn.Linear(256, 512),
            nn.Sigmoid()
        )
        
        # Class-specific branches for better discrimination
        self.lymphoid_branch = nn.Sequential(  # For ALL vs CLL
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
        )
        
        self.myeloid_branch = nn.Sequential(   # For AML vs CML
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
        )
        
        # Final classifier combining all features
        self.classifier = nn.Sequential(
            nn.Linear(512 + 128 + 128, 256),  # Combined features
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(256, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, num_classes)
        )
    
    def forward(self, x):
        # Extract base features
        features = self.feature_extractor(x)
        
        # Apply attention mechanism
        attention_weights = self.attention(features)
        attended_features = features * attention_weights
        
        # Process through specialized branches
        lymphoid_features = self.lymphoid_branch(attended_features)
        myeloid_features = self.myeloid_branch(attended_features)
        
        # Combine all features
        combined_features = torch.cat([attended_features, lymphoid_features, myeloid_features], dim=1)
        
        # Final classification
        logits = self.classifier(combined_features)
        
        return logits

def load_model():
    """Load the discriminative model"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load base model
    model = googlenet(weights=GoogLeNet_Weights.DEFAULT)
    model.fc = DiscriminativeClassifier(1024, 4)
    
    # Load discriminative weights
    model_path = "blood_cancer_model_discriminative.pth"
    if os.path.exists(model_path):
        try:
            print(f"Loading discriminative model from {model_path}")
            weights = torch.load(model_path, map_location=device)
            model.load_state_dict(weights)
            print("✅ Model loaded successfully!")
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            return None, None
    else:
        print(f"❌ Model file not found: {model_path}")
        return None, None
    
    model.to(device)
    model.eval()
    return model, device

def test_synthetic_data():
    """Test with synthetic data to check model behavior"""
    model, device = load_model()
    if model is None:
        return
    
    print("\n=== Testing with synthetic data ===")
    
    # Create synthetic input (batch of 5 samples)
    batch_size = 5
    synthetic_input = torch.randn(batch_size, 3, 224, 224).to(device)
    
    with torch.no_grad():
        outputs = model(synthetic_input)
        probabilities = torch.softmax(outputs, dim=1)
        predictions = torch.argmax(outputs, dim=1)
    
    print(f"Raw outputs shape: {outputs.shape}")
    print(f"Raw outputs:\n{outputs}")
    print(f"Probabilities:\n{probabilities}")
    print(f"Predictions: {predictions}")
    
    # Check class distribution
    class_names = ['ALL', 'AML', 'CLL', 'CML']
    for i in range(batch_size):
        pred_class = predictions[i].item()
        confidence = probabilities[i][pred_class].item()
        print(f"Sample {i+1}: {class_names[pred_class]} (confidence: {confidence:.3f})")
    
    # Check if all predictions are the same
    unique_preds = torch.unique(predictions)
    print(f"\nUnique predictions: {unique_preds}")
    if len(unique_preds) == 1:
        print("⚠️  WARNING: All predictions are the same class!")
    else:
        print("✅ Model produces different predictions")

def analyze_model_weights():
    """Analyze model weights to check for issues"""
    model, device = load_model()
    if model is None:
        return
    
    print("\n=== Analyzing model weights ===")
    
    # Check final classifier weights
    final_layer = model.fc.classifier[-1]  # Last linear layer
    weights = final_layer.weight.data
    bias = final_layer.bias.data
    
    print(f"Final layer weights shape: {weights.shape}")
    print(f"Final layer weights:\n{weights}")
    print(f"Final layer bias: {bias}")
    
    # Check for potential issues
    weight_std = torch.std(weights, dim=1)
    print(f"Weight standard deviation per class: {weight_std}")
    
    if torch.allclose(weights[0], weights[1], atol=1e-3):
        print("⚠️  WARNING: Class 0 and 1 weights are very similar!")
    if torch.allclose(weights[2], weights[3], atol=1e-3):
        print("⚠️  WARNING: Class 2 and 3 weights are very similar!")

def main():
    print("🔍 Debugging discriminative model predictions...")
    
    # Test with synthetic data
    test_synthetic_data()
    
    # Analyze model weights
    analyze_model_weights()
    
    print("\n🔍 Debug analysis complete!")

if __name__ == "__main__":
    main()