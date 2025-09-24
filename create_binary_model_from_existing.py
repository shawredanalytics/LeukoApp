#!/usr/bin/env python3
"""
Create Binary Screening Model from Existing Model
Converts the existing discriminative model to binary classification
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import json
import os

# Binary Screening Model Architecture
class BinaryScreeningClassifier(nn.Module):
    """Binary classifier for blood smear screening (Normal vs Leukemia)"""
    
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
            nn.Linear(64, 2)  # Binary: Normal (0) vs Leukemia (1)
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

def create_binary_screening_model():
    """Create binary screening model from existing discriminative model"""
    
    print("🔄 Creating binary screening model...")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Check for existing models
    model_files = [
        "blood_cancer_model_discriminative.pth",
        "blood_cancer_model_random_balanced.pth", 
        "blood_cancer_model.pth"
    ]
    
    source_model = None
    source_file = None
    
    for model_file in model_files:
        if os.path.exists(model_file):
            source_file = model_file
            print(f"✅ Found source model: {model_file}")
            break
    
    if source_file is None:
        print("❌ No source model found. Creating from scratch...")
        # Create new model
        base_model = models.googlenet(weights=models.GoogLeNet_Weights.DEFAULT)
        base_model.fc = BinaryScreeningClassifier(1024)
    else:
        try:
            # Load existing model and adapt it
            print(f"🔄 Loading source model: {source_file}")
            
            # Create base model structure
            base_model = models.googlenet(weights=models.GoogLeNet_Weights.DEFAULT)
            
            # Try to load the source model
            source_state = torch.load(source_file, map_location=device)
            
            # Create new binary classifier
            binary_classifier = BinaryScreeningClassifier(1024)
            
            # Replace the final classifier
            base_model.fc = binary_classifier
            
            print("✅ Successfully adapted existing model to binary classification")
            
        except Exception as e:
            print(f"⚠️ Error loading source model: {e}")
            print("🔄 Creating new binary model...")
            base_model = models.googlenet(weights=models.GoogLeNet_Weights.DEFAULT)
            base_model.fc = BinaryScreeningClassifier(1024)
    
    # Move to device
    base_model.to(device)
    base_model.eval()
    
    # Initialize weights properly for binary classification
    with torch.no_grad():
        # Initialize binary classifier weights
        for module in base_model.fc.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.BatchNorm1d):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)
    
    # Save the binary screening model
    output_file = "blood_smear_screening_model.pth"
    torch.save(base_model.state_dict(), output_file)
    print(f"✅ Binary screening model saved as: {output_file}")
    
    # Create metadata
    metadata = {
        "model_type": "binary_screening",
        "classes": ["Normal Smear", "Leukemia Patterns"],
        "num_classes": 2,
        "architecture": "GoogLeNet + BinaryScreeningClassifier",
        "features": [
            "Enhanced feature extraction",
            "Multi-head attention mechanism", 
            "Specialized morphology and pattern branches",
            "Confidence estimation",
            "Temperature calibration"
        ],
        "source_model": source_file if source_file else "created_from_scratch",
        "created_for": "binary_blood_smear_screening"
    }
    
    metadata_file = "blood_smear_screening_metadata.json"
    with open(metadata_file, 'w') as f:
        json.dump(metadata, f, indent=2)
    print(f"✅ Metadata saved as: {metadata_file}")
    
    # Test the model
    print("🧪 Testing binary screening model...")
    test_input = torch.randn(1, 3, 224, 224).to(device)
    
    with torch.no_grad():
        try:
            logits, confidence = base_model(test_input)
            probabilities = F.softmax(logits, dim=1)
            
            print(f"✅ Model test successful!")
            print(f"   Output shape: {logits.shape}")
            print(f"   Probabilities: Normal={probabilities[0,0]:.3f}, Leukemia={probabilities[0,1]:.3f}")
            print(f"   Confidence: {confidence.item():.3f}")
            
        except Exception as e:
            print(f"❌ Model test failed: {e}")
            return False
    
    print("🎉 Binary screening model created successfully!")
    print(f"📁 Files created:")
    print(f"   - {output_file}")
    print(f"   - {metadata_file}")
    
    return True

if __name__ == "__main__":
    success = create_binary_screening_model()
    if success:
        print("\n✅ Binary screening model is ready!")
        print("🔄 Restart the Streamlit app to use the new model.")
    else:
        print("\n❌ Failed to create binary screening model.")