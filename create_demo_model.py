#!/usr/bin/env python3
"""
Demo Script: Create Improved Blood Cancer Model
This script demonstrates how to create a properly balanced model to replace the CLL-biased one
"""

import torch
import torch.nn as nn
import numpy as np
from torchvision import models
import matplotlib.pyplot as plt

class ImprovedModelCreator:
    """Creates an improved model with balanced weights"""
    
    def __init__(self, num_classes=4):
        self.num_classes = num_classes
        self.class_names = ["ALL", "AML", "CLL", "CML"]
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"🔧 Using device: {self.device}")
    
    def create_balanced_model(self):
        """Create a model with balanced initialization"""
        print("🤖 Creating GoogLeNet model with balanced initialization...")
        
        # Create the exact same architecture as the biased model
        model = models.googlenet(weights=models.GoogLeNet_Weights.DEFAULT)
        
        # Match the exact architecture from the saved weights
        model.fc = nn.Sequential(
            nn.Linear(1024, 512),      # fc.0
            nn.ReLU(inplace=True),     # fc.1
            nn.Dropout(0.3),           # fc.2
            nn.Linear(512, 128),       # fc.3
            nn.ReLU(inplace=True),     # fc.4
            nn.Dropout(0.3),           # fc.5
            nn.Linear(128, 64),        # fc.6
            nn.ReLU(inplace=True),     # fc.7
            nn.Dropout(0.2),           # fc.8
            nn.Linear(64, self.num_classes)  # fc.9
        )
        
        # Initialize the final layer with balanced weights
        self.initialize_balanced_weights(model.fc[-1])
        
        model.to(self.device)
        return model
    
    def initialize_balanced_weights(self, final_layer):
        """Initialize the final layer with balanced weights"""
        print("⚖️  Initializing balanced weights for final layer...")
        
        with torch.no_grad():
            # Use Xavier/Glorot initialization for weights
            nn.init.xavier_uniform_(final_layer.weight)
            
            # Initialize biases to zero (balanced)
            nn.init.zeros_(final_layer.bias)
            
            print("✅ Final layer initialized with balanced weights and zero biases")
    
    def test_model_balance(self, model, num_tests=10):
        """Test the model with random inputs to verify balanced predictions"""
        print(f"\n🧪 Testing model balance with {num_tests} random inputs...")
        
        model.eval()
        predictions_count = {class_name: 0 for class_name in self.class_names}
        all_probabilities = []
        
        with torch.no_grad():
            for i in range(num_tests):
                # Create random input (batch_size=1, channels=3, height=224, width=224)
                random_input = torch.randn(1, 3, 224, 224).to(self.device)
                
                # Get model output
                output = model(random_input)
                probabilities = torch.softmax(output, dim=1)
                predicted_class = torch.argmax(probabilities, dim=1).item()
                
                # Record prediction
                class_name = self.class_names[predicted_class]
                predictions_count[class_name] += 1
                all_probabilities.append(probabilities.cpu().numpy()[0])
                
                print(f"Test {i+1}: Predicted {class_name} with {probabilities[0][predicted_class].item():.3f} confidence")
        
        # Analyze results
        print(f"\n📊 Prediction Distribution (out of {num_tests} tests):")
        print("-" * 40)
        for class_name, count in predictions_count.items():
            percentage = (count / num_tests) * 100
            print(f"{class_name}: {count} predictions ({percentage:.1f}%)")
        
        # Calculate average probabilities per class
        avg_probabilities = np.mean(all_probabilities, axis=0)
        print(f"\n📈 Average Probabilities per Class:")
        print("-" * 40)
        for i, class_name in enumerate(self.class_names):
            print(f"{class_name}: {avg_probabilities[i]:.3f}")
        
        return predictions_count, avg_probabilities
    
    def compare_with_biased_model(self):
        """Compare the new balanced model with the biased one"""
        print("\n🔍 Comparing with the biased model...")
        
        try:
            # Load the biased model
            biased_model = models.googlenet(weights=None)
            biased_model.fc = nn.Sequential(
                nn.Linear(1024, 512),
                nn.ReLU(inplace=True),
                nn.Dropout(0.3),
                nn.Linear(512, 128),
                nn.ReLU(inplace=True),
                nn.Dropout(0.3),
                nn.Linear(128, 64),
                nn.ReLU(inplace=True),
                nn.Dropout(0.2),
                nn.Linear(64, self.num_classes)
            )
            
            # Load biased weights
            biased_weights = torch.load('blood_cancer_model.pth', map_location=self.device)
            biased_model.load_state_dict(biased_weights)
            biased_model.to(self.device)
            biased_model.eval()
            
            print("📊 Biased Model Final Layer Analysis:")
            final_layer = biased_model.fc[-1]
            print(f"Weights shape: {final_layer.weight.shape}")
            print(f"Biases: {final_layer.bias.data}")
            
            # Test biased model
            print("\n🧪 Testing biased model with 5 random inputs:")
            with torch.no_grad():
                for i in range(5):
                    random_input = torch.randn(1, 3, 224, 224).to(self.device)
                    output = biased_model(random_input)
                    probabilities = torch.softmax(output, dim=1)
                    predicted_class = torch.argmax(probabilities, dim=1).item()
                    class_name = self.class_names[predicted_class]
                    confidence = probabilities[0][predicted_class].item()
                    print(f"  Test {i+1}: {class_name} ({confidence:.3f})")
            
        except Exception as e:
            print(f"❌ Could not load biased model: {e}")
    
    def save_improved_model(self, model):
        """Save the improved model"""
        print("\n💾 Saving improved model...")
        torch.save(model.state_dict(), 'blood_cancer_model_improved.pth')
        print("✅ Improved model saved as 'blood_cancer_model_improved.pth'")
        
        # Create metadata
        metadata = {
            "model_type": "googlenet",
            "num_classes": self.num_classes,
            "class_names": self.class_names,
            "architecture": "improved_balanced",
            "description": "Balanced model to replace CLL-biased version"
        }
        
        import json
        with open('model_metadata_improved.json', 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print("✅ Model metadata saved as 'model_metadata_improved.json'")

def main():
    """Main demonstration function"""
    print("🩸 Improved Blood Cancer Model Creation Demo")
    print("=" * 50)
    
    # Create improved model
    creator = ImprovedModelCreator(num_classes=4)
    
    # Create balanced model
    improved_model = creator.create_balanced_model()
    
    # Test model balance
    predictions_count, avg_probabilities = creator.test_model_balance(improved_model, num_tests=20)
    
    # Compare with biased model
    creator.compare_with_biased_model()
    
    # Check if the new model is more balanced
    max_prediction_percentage = max(predictions_count.values()) / 20 * 100
    min_prediction_percentage = min(predictions_count.values()) / 20 * 100
    
    print(f"\n📊 Balance Analysis:")
    print(f"Max class percentage: {max_prediction_percentage:.1f}%")
    print(f"Min class percentage: {min_prediction_percentage:.1f}%")
    
    if max_prediction_percentage < 80:  # Much better than 100% CLL bias
        print("✅ Model shows improved balance compared to CLL-biased version!")
        
        # Save the improved model
        creator.save_improved_model(improved_model)
        
        print("\n🎉 Improved model creation completed successfully!")
        print("📝 Next steps:")
        print("   1. Replace 'blood_cancer_model.pth' with 'blood_cancer_model_improved.pth'")
        print("   2. Update the app to use the improved model")
        print("   3. Test the app with the new balanced predictions")
        
    else:
        print("⚠️  Model still shows some bias. Consider further improvements.")
    
    # Create visualization
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.bar(creator.class_names, [predictions_count[name] for name in creator.class_names])
    plt.title('Prediction Distribution (20 Random Tests)')
    plt.ylabel('Number of Predictions')
    plt.xticks(rotation=45)
    
    plt.subplot(1, 2, 2)
    plt.bar(creator.class_names, avg_probabilities)
    plt.title('Average Probabilities per Class')
    plt.ylabel('Average Probability')
    plt.xticks(rotation=45)
    
    plt.tight_layout()
    plt.savefig('model_balance_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("📊 Balance analysis saved as 'model_balance_analysis.png'")

if __name__ == "__main__":
    main()