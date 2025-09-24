#!/usr/bin/env python3
"""
Test script to check binary model bias issues
"""
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
import numpy as np
import os
from PIL import Image

class BinaryScreeningClassifier(nn.Module):
    """Binary classifier for blood smear screening (Normal vs WBC Cancerous Abnormalities)"""
    def __init__(self, input_features):
        super(BinaryScreeningClassifier, self).__init__()
        
        # Enhanced feature extraction for binary classification
        self.feature_extractor = nn.Sequential(
            nn.Linear(input_features, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
        )
        
        # Attention mechanism for important feature selection
        self.attention = nn.Sequential(
            nn.Linear(256, 128),
            nn.Tanh(),
            nn.Linear(128, 256),
            nn.Sigmoid()
        )
        
        # Specialized branches for different cell analysis
        self.morphology_branch = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(128),
            nn.Dropout(0.2),
        )
        
        self.pattern_branch = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(128),
            nn.Dropout(0.2),
        )
        
        # Confidence estimation branch
        self.confidence_branch = nn.Sequential(
            nn.Linear(256, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
        
        # Final binary classifier
        self.classifier = nn.Sequential(
            nn.Linear(256 + 128 + 128, 256),  # Combined features
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            
            nn.Linear(256, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 2)  # Binary: Normal (0) vs WBC Cancerous Abnormalities (1)
        )
        
        # Temperature for calibration
        self.temperature = nn.Parameter(torch.ones(1))
    
    def forward(self, x):
        # Extract features
        features = self.feature_extractor(x)
        
        # Apply attention
        attention_weights = self.attention(features)
        attended_features = features * attention_weights
        
        # Specialized analysis
        morphology_features = self.morphology_branch(attended_features)
        pattern_features = self.pattern_branch(attended_features)
        
        # Confidence prediction
        confidence_score = self.confidence_branch(attended_features)
        
        # Combine all features
        combined_features = torch.cat([attended_features, morphology_features, pattern_features], dim=1)
        
        # Binary classification
        logits = self.classifier(combined_features)
        
        # Temperature scaling
        calibrated_logits = logits / self.temperature
        
        return calibrated_logits, confidence_score

def load_model():
    """Load the binary screening model"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Test different model files
    model_files = [
        "blood_smear_screening_model_fixed.pth",  # Try fixed version first
        "best_binary_model.pth",                  # Then best binary model
        "blood_cancer_model_random_balanced.pth", # Then balanced model
        "blood_smear_screening_model.pth"         # Original as fallback
    ]
    
    for model_path in model_files:
        if os.path.exists(model_path):
            print(f"Loading model from: {model_path}")
            try:
                # Load binary screening model
                base_model = models.googlenet(weights=models.GoogLeNet_Weights.DEFAULT)
                base_model.fc = BinaryScreeningClassifier(1024)
                
                # Load trained weights
                state_dict = torch.load(model_path, map_location=device)
                base_model.load_state_dict(state_dict)
                
                base_model.to(device)
                base_model.eval()
                
                return base_model, device, False, model_path
            except Exception as e:
                print(f"Failed to load {model_path}: {e}")
                continue
    
    print("No compatible model found")
    return None, None, True, None

def test_model_predictions():
    """Test model with synthetic data to check for bias"""
    model, device, demo_mode, model_path = load_model()
    
    if demo_mode:
        print("❌ Model not found - cannot test bias")
        return
    
    print(f"🔍 Testing model for bias using: {model_path}")
    
    # Create random test tensors
    num_tests = 20
    results = []
    
    for i in range(num_tests):
        # Create random image tensor (simulating different inputs)
        test_tensor = torch.randn(1, 3, 224, 224).to(device)
        
        with torch.no_grad():
            logits, confidence = model(test_tensor)
            probabilities = torch.softmax(logits, dim=1)
            predicted_class = torch.argmax(probabilities, dim=1).item()
            
            normal_prob = probabilities[0, 0].item()
            abnormal_prob = probabilities[0, 1].item()
            
            results.append({
                'test_id': i + 1,
                'predicted_class': predicted_class,
                'normal_prob': normal_prob,
                'abnormal_prob': abnormal_prob,
                'confidence': confidence.item()
            })
            
            class_name = "Normal" if predicted_class == 0 else "WBC Cancerous Abnormalities"
            print(f"Test {i+1:2d}: {class_name:25s} | Normal: {normal_prob:.3f} | Abnormal: {abnormal_prob:.3f} | Conf: {confidence.item():.3f}")
    
    # Analyze results
    normal_predictions = sum(1 for r in results if r['predicted_class'] == 0)
    abnormal_predictions = sum(1 for r in results if r['predicted_class'] == 1)
    
    print(f"\n📊 Results Summary for {model_path}:")
    print(f"Normal predictions: {normal_predictions}/{num_tests} ({normal_predictions/num_tests*100:.1f}%)")
    print(f"Abnormal predictions: {abnormal_predictions}/{num_tests} ({abnormal_predictions/num_tests*100:.1f}%)")
    
    if abnormal_predictions > normal_predictions * 2:
        print("⚠️  BIAS DETECTED: Model heavily favors abnormal predictions")
    elif normal_predictions > abnormal_predictions * 2:
        print("⚠️  BIAS DETECTED: Model heavily favors normal predictions")
    else:
        print("✅ Model appears reasonably balanced")
    
    # Check average probabilities
    avg_normal_prob = np.mean([r['normal_prob'] for r in results])
    avg_abnormal_prob = np.mean([r['abnormal_prob'] for r in results])
    
    print(f"\nAverage probabilities:")
    print(f"Normal: {avg_normal_prob:.3f}")
    print(f"Abnormal: {avg_abnormal_prob:.3f}")
    
    return model_path, normal_predictions, abnormal_predictions

if __name__ == "__main__":
    test_model_predictions()