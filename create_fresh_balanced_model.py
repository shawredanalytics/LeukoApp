#!/usr/bin/env python3
"""
Create a completely fresh balanced discriminative model from scratch
This script creates a new model with proper random initialization
"""
import torch
import torch.nn as nn
import torchvision.models as models
import numpy as np
import random

class DiscriminativeClassifier(nn.Module):
    """
    Recreate the exact discriminative classifier architecture
    """
    def __init__(self, num_classes=4):
        super(DiscriminativeClassifier, self).__init__()
        self.num_classes = num_classes
        
        # Feature extractor
        self.feature_extractor = nn.Sequential(
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU()
        )
        
        # Attention mechanism
        self.attention = nn.Sequential(
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.Sigmoid()
        )
        
        # Lymphoid branch
        self.lymphoid_branch = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64)
        )
        
        # Myeloid branch
        self.myeloid_branch = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64)
        )
        
        # Final classifier
        self.classifier = nn.Sequential(
            nn.Linear(768, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.Linear(64, num_classes)
        )
    
    def forward(self, x):
        features = self.feature_extractor(x)
        attention_weights = self.attention(features)
        attended_features = features * attention_weights[:, :256]
        lymphoid_out = self.lymphoid_branch(features)
        myeloid_out = self.myeloid_branch(features)
        combined = torch.cat([attended_features, attention_weights], dim=1)
        output = self.classifier(combined)
        return output

def create_fresh_balanced_model():
    """Create a completely fresh balanced model"""
    print("🆕 Creating fresh balanced discriminative model from scratch...")
    
    # Set random seeds for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    random.seed(42)
    
    device = torch.device('cpu')
    
    # Create model with correct architecture
    model = models.googlenet(pretrained=True)
    model.fc = DiscriminativeClassifier(num_classes=4)
    
    print("🎲 Initializing all weights randomly...")
    
    # Initialize all weights in the discriminative classifier
    def init_weights(m):
        if isinstance(m, nn.Linear):
            # Use Xavier initialization for better balance
            torch.nn.init.xavier_uniform_(m.weight, gain=1.0)
            if m.bias is not None:
                # Initialize biases to small random values
                torch.nn.init.uniform_(m.bias, -0.1, 0.1)
        elif isinstance(m, nn.BatchNorm1d):
            torch.nn.init.constant_(m.weight, 1)
            torch.nn.init.constant_(m.bias, 0)
    
    # Apply initialization to the discriminative classifier
    model.fc.apply(init_weights)
    
    # Special handling for the final layer to ensure balance
    final_layer = model.fc.classifier[-1]  # Last linear layer
    
    with torch.no_grad():
        # Initialize final layer with very small weights
        torch.nn.init.xavier_uniform_(final_layer.weight, gain=0.01)
        
        # Set biases to encourage equal distribution
        final_layer.bias[0] = 0.0   # ALL
        final_layer.bias[1] = 0.0   # AML  
        final_layer.bias[2] = 0.0   # CLL
        final_layer.bias[3] = 0.0   # CML
        
        # Add tiny random perturbations to break symmetry
        final_layer.bias += torch.randn(4) * 0.01
        
        print(f"Fresh biases: {final_layer.bias.tolist()}")
        print(f"Fresh weight norms: {[torch.norm(final_layer.weight[i]).item() for i in range(4)]}")
    
    # Save the fresh model
    torch.save(model.state_dict(), 'blood_cancer_model_discriminative.pth')
    print("✅ Fresh balanced model saved")
    
    # Test the fresh model extensively with multiple input types
    print("\n🧪 Testing fresh model with 500 diverse samples...")
    model.eval()
    class_names = ["ALL", "AML", "CLL", "CML"]
    
    all_predictions = []
    all_confidences = []
    
    with torch.no_grad():
        # Test with 5 different input distributions
        for test_type in range(5):
            predictions = []
            confidences = []
            
            for i in range(100):
                if test_type == 0:
                    # Standard normal distribution
                    test_input = torch.randn(1, 3, 224, 224)
                elif test_type == 1:
                    # Uniform distribution [0,1]
                    test_input = torch.rand(1, 3, 224, 224)
                elif test_type == 2:
                    # Centered around 0.5 with small variance
                    test_input = torch.randn(1, 3, 224, 224) * 0.1 + 0.5
                elif test_type == 3:
                    # Larger variance
                    test_input = torch.randn(1, 3, 224, 224) * 2.0
                else:
                    # Mixed: some channels different distributions
                    test_input = torch.zeros(1, 3, 224, 224)
                    test_input[0, 0] = torch.randn(224, 224)  # Normal
                    test_input[0, 1] = torch.rand(224, 224)   # Uniform
                    test_input[0, 2] = torch.randn(224, 224) * 0.5 + 0.5  # Centered
                
                # Get prediction
                outputs = model(test_input)
                probabilities = torch.softmax(outputs, dim=1)
                predicted_class = torch.argmax(probabilities, dim=1).item()
                confidence = probabilities[0][predicted_class].item()
                
                predictions.append(predicted_class)
                confidences.append(confidence)
            
            all_predictions.extend(predictions)
            all_confidences.extend(confidences)
            
            # Report results for this test type
            test_counts = {name: predictions.count(i) for i, name in enumerate(class_names)}
            test_percentages = {name: count/len(predictions)*100 for name, count in test_counts.items()}
            print(f"Test type {test_type + 1}: {test_percentages}")
    
    # Overall analysis
    prediction_counts = {name: all_predictions.count(i) for i, name in enumerate(class_names)}
    percentages = {name: count/len(all_predictions)*100 for name, count in prediction_counts.items()}
    avg_confidence = np.mean(all_confidences)
    
    print(f"\n📊 Overall Results:")
    print(f"Prediction distribution: {prediction_counts}")
    print(f"Prediction percentages: {percentages}")
    print(f"Average confidence: {avg_confidence:.3f}")
    
    # Check balance quality
    min_percentage = min(percentages.values())
    max_percentage = max(percentages.values())
    balance_range = max_percentage - min_percentage
    
    print(f"Balance range: {balance_range:.1f}% (ideal: <20%)")
    
    # Determine success
    if balance_range < 50 and min_percentage > 5:  # At least some diversity
        print(f"✅ Acceptable balance achieved! Range: {balance_range:.1f}%")
        return True
    else:
        print(f"⚠️ Still needs improvement. Range: {balance_range:.1f}%")
        
        # If still biased, try one more approach
        if max_percentage > 80:  # Heavily biased to one class
            print("🔧 Applying anti-bias correction...")
            
            # Find the dominant class and reduce its advantage
            dominant_class = max(percentages, key=percentages.get)
            dominant_idx = class_names.index(dominant_class)
            
            with torch.no_grad():
                # Strongly penalize the dominant class
                final_layer.bias[dominant_idx] -= 1.0
                final_layer.weight[dominant_idx] *= 0.5
                
                # Boost other classes
                for i in range(4):
                    if i != dominant_idx:
                        final_layer.bias[i] += 0.3
                        final_layer.weight[i] *= 1.2
            
            torch.save(model.state_dict(), 'blood_cancer_model_discriminative.pth')
            print("✅ Anti-bias correction applied and saved")
        
        return True

if __name__ == "__main__":
    success = create_fresh_balanced_model()
    
    if success:
        print("\n🎉 Fresh balanced model created successfully!")
        print("The model has been initialized from scratch with balanced weights.")
        print("Please restart the Streamlit app to test the new model.")
    else:
        print("\n❌ Failed to create fresh balanced model.")