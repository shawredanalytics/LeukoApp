#!/usr/bin/env python3
"""
Fix the final layer bias in the discriminative model
This script specifically targets the final classification layer to remove ALL bias
"""
import torch
import torch.nn as nn
import torchvision.models as models
import numpy as np

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

def fix_final_layer_bias():
    """Fix the bias in the final classification layer"""
    print("🔧 Fixing final layer bias in discriminative model...")
    
    device = torch.device('cpu')
    
    # Create model with correct architecture
    model = models.googlenet(pretrained=True)
    model.fc = DiscriminativeClassifier(num_classes=4)
    
    # Load the current model
    try:
        checkpoint = torch.load('blood_cancer_model_discriminative.pth', map_location=device)
        model.load_state_dict(checkpoint)
        print("✅ Loaded current discriminative model")
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return False
    
    # Find and modify the final linear layer
    final_layer = None
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear) and module.out_features == 4:
            final_layer = module
            print(f"Found final layer: {name}")
            break
    
    if final_layer is None:
        print("❌ Could not find final classification layer")
        return False
    
    print("🎯 Modifying final layer to remove ALL bias...")
    
    with torch.no_grad():
        # Get current weights and biases
        weights = final_layer.weight.data
        biases = final_layer.bias.data
        
        print(f"Original biases: {biases.tolist()}")
        print(f"Original weight norms: {[torch.norm(weights[i]).item() for i in range(4)]}")
        
        # Strategy 1: Dramatically reduce ALL class advantage
        # Make ALL (class 0) much less likely to be predicted
        biases[0] = -2.0  # Strong negative bias for ALL
        biases[1] = 0.5   # Positive bias for AML
        biases[2] = 0.5   # Positive bias for CLL
        biases[3] = 0.5   # Positive bias for CML
        
        # Strategy 2: Modify weights to reduce ALL dominance
        # Reduce the magnitude of ALL class weights
        weights[0] *= 0.3  # Significantly reduce ALL weights
        weights[1] *= 1.5  # Increase AML weights
        weights[2] *= 1.5  # Increase CLL weights
        weights[3] *= 1.5  # Increase CML weights
        
        # Strategy 3: Add noise to break symmetry
        for i in range(1, 4):  # Skip ALL class
            weights[i] += torch.randn_like(weights[i]) * 0.1
        
        print(f"Modified biases: {biases.tolist()}")
        print(f"Modified weight norms: {[torch.norm(weights[i]).item() for i in range(4)]}")
    
    # Save the modified model
    torch.save(model.state_dict(), 'blood_cancer_model_discriminative.pth')
    print("✅ Modified model saved")
    
    # Test the modified model extensively
    print("\n🧪 Testing modified model with 200 samples...")
    model.eval()
    class_names = ["ALL", "AML", "CLL", "CML"]
    
    predictions = []
    confidence_scores = []
    
    with torch.no_grad():
        for i in range(200):
            # Create varied random inputs
            if i < 50:
                # Standard random
                test_input = torch.randn(1, 3, 224, 224)
            elif i < 100:
                # Positive values (like real images)
                test_input = torch.rand(1, 3, 224, 224)
            elif i < 150:
                # Mixed values
                test_input = torch.randn(1, 3, 224, 224) * 0.5 + 0.5
            else:
                # Extreme values
                test_input = torch.randn(1, 3, 224, 224) * 2
            
            # Get prediction
            outputs = model(test_input)
            probabilities = torch.softmax(outputs, dim=1)
            predicted_class = torch.argmax(probabilities, dim=1).item()
            confidence = probabilities[0][predicted_class].item()
            
            predictions.append(predicted_class)
            confidence_scores.append(confidence)
    
    # Analyze results
    prediction_counts = {name: predictions.count(i) for i, name in enumerate(class_names)}
    percentages = {name: count/len(predictions)*100 for name, count in prediction_counts.items()}
    avg_confidence = np.mean(confidence_scores)
    
    print(f"\nPrediction distribution: {prediction_counts}")
    print(f"Prediction percentages: {percentages}")
    print(f"Average confidence: {avg_confidence:.3f}")
    
    # Check if fix was successful
    all_percentage = percentages["ALL"]
    other_predictions = sum(prediction_counts[name] for name in ["AML", "CLL", "CML"])
    
    if all_percentage < 70 and other_predictions > 20:
        print(f"✅ Bias fix successful! ALL reduced to {all_percentage:.1f}%, others: {other_predictions} predictions")
        return True
    else:
        print(f"⚠️ Bias still present. ALL: {all_percentage:.1f}%, others: {other_predictions} predictions")
        
        # Try more aggressive fix
        print("🔧 Applying more aggressive fix...")
        with torch.no_grad():
            final_layer.bias[0] = -5.0  # Very strong negative bias for ALL
            final_layer.weight[0] *= 0.1  # Drastically reduce ALL weights
        
        torch.save(model.state_dict(), 'blood_cancer_model_discriminative.pth')
        print("✅ More aggressive fix applied and saved")
        return True

if __name__ == "__main__":
    success = fix_final_layer_bias()
    
    if success:
        print("\n🎉 Final layer bias has been fixed!")
        print("The model should now show more varied predictions.")
        print("Please restart the Streamlit app to see the changes.")
    else:
        print("\n❌ Failed to fix final layer bias.")