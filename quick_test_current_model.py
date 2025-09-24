#!/usr/bin/env python3
"""
Quick test script to verify current model behavior
"""
import torch
import torch.nn as nn
import torchvision.models as models
import numpy as np
import json

class DiscriminativeClassifier(nn.Module):
    def __init__(self, num_classes=4):
        super(DiscriminativeClassifier, self).__init__()
        self.num_classes = num_classes
        
        # Multi-branch architecture for better discrimination
        self.branch1 = nn.Sequential(
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 256)
        )
        
        self.branch2 = nn.Sequential(
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 256)
        )
        
        # Attention mechanism
        self.attention = nn.Sequential(
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )
        
        # Final classifier
        self.classifier = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(256, num_classes)
        )
    
    def forward(self, x):
        # Multi-branch processing
        branch1_out = self.branch1(x)
        branch2_out = self.branch2(x)
        
        # Combine branches
        combined = torch.cat([branch1_out, branch2_out], dim=1)
        
        # Apply attention
        attention_weights = self.attention(combined)
        attended_features = combined * attention_weights
        
        # Final classification
        output = self.classifier(attended_features)
        return output

def test_current_model():
    """Test the current discriminative model"""
    print("🔍 Testing current discriminative model...")
    
    # Load class mapping
    try:
        with open('model_metadata_discriminative.json', 'r') as f:
            metadata = json.load(f)
        class_names = metadata['class_names']
        print(f"📋 Class names: {class_names}")
    except Exception as e:
        print(f"❌ Error loading metadata: {e}")
        class_names = ["ALL", "AML", "CLL", "CML"]
    
    # Create model
    device = torch.device('cpu')
    model = models.googlenet(pretrained=True)
    model.fc = DiscriminativeClassifier(num_classes=4)
    
    # Load weights
    try:
        model_path = 'blood_cancer_model_discriminative.pth'
        checkpoint = torch.load(model_path, map_location=device)
        model.load_state_dict(checkpoint)
        model.eval()
        print(f"✅ Model loaded from {model_path}")
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return
    
    # Test with random inputs
    print("\n🧪 Testing with random inputs...")
    with torch.no_grad():
        for i in range(10):
            # Create random input (batch_size=1, channels=3, height=224, width=224)
            test_input = torch.randn(1, 3, 224, 224)
            
            # Get prediction
            outputs = model(test_input)
            probabilities = torch.softmax(outputs, dim=1)
            predicted_class = torch.argmax(probabilities, dim=1).item()
            confidence = probabilities[0][predicted_class].item()
            
            print(f"Test {i+1}: Predicted={class_names[predicted_class]} (confidence: {confidence:.3f})")
            print(f"  All probabilities: {[f'{p:.3f}' for p in probabilities[0].tolist()]}")
    
    # Analyze final layer weights
    print("\n📊 Analyzing final layer weights...")
    final_layer = model.fc.classifier[-1]  # Last linear layer
    weights = final_layer.weight.data
    biases = final_layer.bias.data
    
    print(f"Weight shape: {weights.shape}")
    print(f"Bias values: {biases.tolist()}")
    
    # Check weight statistics per class
    for i, class_name in enumerate(class_names):
        class_weights = weights[i]
        print(f"{class_name}: mean={class_weights.mean():.4f}, std={class_weights.std():.4f}")

if __name__ == "__main__":
    test_current_model()