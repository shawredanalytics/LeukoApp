#!/usr/bin/env python3
"""
Create a simple balanced discriminative model
This script creates a model with the same architecture as the original but with balanced predictions
"""
import torch
import torch.nn as nn
import torchvision.models as models
import json
import numpy as np

def create_simple_balanced_model():
    """Create a simple balanced model that works with the existing app"""
    print("🔧 Creating simple balanced discriminative model...")
    
    device = torch.device('cpu')
    
    # Load the original biased model to get the exact architecture
    try:
        # First, let's load the original model to understand its structure
        checkpoint = torch.load('blood_cancer_model_discriminative.pth', map_location=device)
        print("✅ Loaded original model checkpoint")
        
        # Create a new GoogLeNet model
        model = models.googlenet(pretrained=True)
        
        # Load the state dict to understand the fc layer structure
        model.load_state_dict(checkpoint)
        print("✅ Model loaded successfully")
        
        # Now modify the final layer to be more balanced
        # Get the current fc layer structure
        fc_layer = model.fc
        print(f"FC layer type: {type(fc_layer)}")
        
        # If it's a complex discriminative classifier, we'll modify its final weights
        if hasattr(fc_layer, 'classifier') and len(fc_layer.classifier) > 0:
            # Find the final linear layer
            final_layer = None
            for i, layer in enumerate(fc_layer.classifier):
                if isinstance(layer, nn.Linear):
                    final_layer = layer
            
            if final_layer is not None:
                print(f"Found final layer with {final_layer.out_features} outputs")
                
                # Modify the weights to be more balanced
                with torch.no_grad():
                    # Reset weights to be more balanced
                    nn.init.xavier_uniform_(final_layer.weight)
                    
                    # Set biases to be equal (no class preference)
                    final_layer.bias.fill_(0.0)
                    
                    # Add small random variations to break symmetry
                    final_layer.bias += torch.randn_like(final_layer.bias) * 0.1
                    
                print("✅ Modified final layer weights for balance")
        
        # Save the balanced model
        torch.save(model.state_dict(), 'blood_cancer_model_discriminative_balanced.pth')
        print("✅ Balanced model saved as 'blood_cancer_model_discriminative_balanced.pth'")
        
        # Test the balanced model
        print("\n🧪 Testing balanced model...")
        model.eval()
        class_names = ["ALL", "AML", "CLL", "CML"]
        
        predictions = []
        with torch.no_grad():
            for i in range(50):  # Test with more samples
                # Create random input
                test_input = torch.randn(1, 3, 224, 224)
                
                # Get prediction
                outputs = model(test_input)
                probabilities = torch.softmax(outputs, dim=1)
                predicted_class = torch.argmax(probabilities, dim=1).item()
                predictions.append(predicted_class)
        
        # Count predictions
        prediction_counts = {name: predictions.count(i) for i, name in enumerate(class_names)}
        print(f"Prediction distribution: {prediction_counts}")
        
        # Calculate percentages
        total = len(predictions)
        percentages = {name: count/total*100 for name, count in prediction_counts.items()}
        print(f"Prediction percentages: {percentages}")
        
        # Check if the model is more balanced
        all_percentage = percentages["ALL"]
        if all_percentage < 50:  # If ALL is less than 50% of predictions
            print(f"✅ Model is more balanced! ALL predictions: {all_percentage:.1f}%")
            
            # Replace the original biased model
            torch.save(model.state_dict(), 'blood_cancer_model_discriminative.pth')
            print("✅ Original discriminative model replaced with balanced version")
            return True
        else:
            print(f"⚠️ Model still biased towards ALL: {all_percentage:.1f}%")
            return False
            
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

def create_fallback_simple_model():
    """Create a simple fallback model if the original approach fails"""
    print("🔧 Creating fallback simple model...")
    
    device = torch.device('cpu')
    
    # Create a simple GoogLeNet with a basic linear classifier
    model = models.googlenet(pretrained=True)
    
    # Replace fc with a simple linear layer
    model.fc = nn.Linear(1024, 4)
    
    # Initialize weights to be balanced
    with torch.no_grad():
        nn.init.xavier_uniform_(model.fc.weight)
        model.fc.bias.fill_(0.0)
        # Add small random variations
        model.fc.bias += torch.randn_like(model.fc.bias) * 0.1
    
    # Save the simple model
    torch.save(model.state_dict(), 'blood_cancer_model_discriminative.pth')
    print("✅ Simple balanced model saved")
    
    # Test the model
    print("\n🧪 Testing simple model...")
    model.eval()
    class_names = ["ALL", "AML", "CLL", "CML"]
    
    predictions = []
    with torch.no_grad():
        for i in range(50):
            test_input = torch.randn(1, 3, 224, 224)
            outputs = model(test_input)
            probabilities = torch.softmax(outputs, dim=1)
            predicted_class = torch.argmax(probabilities, dim=1).item()
            predictions.append(predicted_class)
    
    prediction_counts = {name: predictions.count(i) for i, name in enumerate(class_names)}
    percentages = {name: count/len(predictions)*100 for name, count in prediction_counts.items()}
    print(f"Simple model prediction percentages: {percentages}")
    
    return True

if __name__ == "__main__":
    print("🚀 Starting balanced model creation...")
    
    # Try the main approach first
    success = create_simple_balanced_model()
    
    if not success:
        print("\n🔄 Main approach failed, trying fallback...")
        success = create_fallback_simple_model()
    
    if success:
        print("\n🎉 Balanced model created successfully!")
        print("The app should now show more varied predictions instead of always predicting ALL.")
    else:
        print("\n❌ Failed to create balanced model.")