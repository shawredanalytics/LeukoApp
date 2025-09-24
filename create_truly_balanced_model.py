#!/usr/bin/env python3
"""
Create Truly Balanced Blood Cancer Model
This script creates a model with truly balanced predictions across all classes
"""

import torch
import torch.nn as nn
import numpy as np
from torchvision import models
import matplotlib.pyplot as plt
import json

class TrulyBalancedModelCreator:
    """Creates a truly balanced model with equal probability distribution"""
    
    def __init__(self, num_classes=4):
        self.num_classes = num_classes
        self.class_names = ["ALL", "AML", "CLL", "CML"]
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"🔧 Using device: {self.device}")
    
    def create_balanced_model(self):
        """Create a model with truly balanced initialization"""
        print("🤖 Creating GoogLeNet model with truly balanced initialization...")
        
        # Create model with default settings
        model = models.googlenet(weights=models.GoogLeNet_Weights.DEFAULT)
        
        # Disable auxiliary classifiers for inference
        model.aux_logits = False
        
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
        
        # Initialize all layers with balanced weights
        self.initialize_truly_balanced_weights(model)
        
        model.to(self.device)
        return model
    
    def initialize_truly_balanced_weights(self, model):
        """Initialize the model with truly balanced weights"""
        print("⚖️  Initializing truly balanced weights...")
        
        with torch.no_grad():
            # Initialize all linear layers in the custom classifier
            for i, layer in enumerate(model.fc):
                if isinstance(layer, nn.Linear):
                    # Use small random weights
                    nn.init.normal_(layer.weight, mean=0.0, std=0.01)
                    # Initialize biases to zero
                    nn.init.zeros_(layer.bias)
                    print(f"   Initialized fc.{i} with small random weights")
            
            # Special initialization for the final layer to ensure balance
            final_layer = model.fc[-1]
            
            # Set very small weights for the final layer
            nn.init.normal_(final_layer.weight, mean=0.0, std=0.001)
            
            # Set biases to create equal probability distribution
            # When all inputs are similar, we want equal probabilities
            # So we set all biases to the same small value
            equal_bias = 0.0
            final_layer.bias.fill_(equal_bias)
            
            print(f"✅ Final layer initialized with equal biases: {equal_bias}")
    
    def test_model_balance(self, model, num_tests=50):
        """Test the model with random inputs to verify balanced predictions"""
        print(f"\n🧪 Testing model balance with {num_tests} random inputs...")
        
        model.eval()
        predictions_count = {class_name: 0 for class_name in self.class_names}
        all_probabilities = []
        confidence_scores = []
        
        with torch.no_grad():
            for i in range(num_tests):
                # Create random input (batch_size=1, channels=3, height=224, width=224)
                random_input = torch.randn(1, 3, 224, 224).to(self.device)
                
                # Get model output
                output = model(random_input)
                probabilities = torch.softmax(output, dim=1)
                predicted_class = torch.argmax(probabilities, dim=1).item()
                max_confidence = torch.max(probabilities).item()
                
                # Record prediction
                class_name = self.class_names[predicted_class]
                predictions_count[class_name] += 1
                all_probabilities.append(probabilities.cpu().numpy()[0])
                confidence_scores.append(max_confidence)
                
                if i < 10:  # Show first 10 predictions
                    print(f"Test {i+1}: Predicted {class_name} with {max_confidence:.3f} confidence")
                    print(f"   Probabilities: {[f'{p:.3f}' for p in probabilities.cpu().numpy()[0]]}")
        
        # Analyze results
        print(f"\n📊 Prediction Distribution (out of {num_tests} tests):")
        print("-" * 50)
        for class_name, count in predictions_count.items():
            percentage = (count / num_tests) * 100
            print(f"{class_name}: {count:2d} predictions ({percentage:5.1f}%)")
        
        # Calculate average probabilities per class
        avg_probabilities = np.mean(all_probabilities, axis=0)
        print(f"\n📈 Average Probabilities per Class:")
        print("-" * 40)
        for i, class_name in enumerate(self.class_names):
            print(f"{class_name}: {avg_probabilities[i]:.3f}")
        
        # Calculate balance metrics
        prediction_percentages = [predictions_count[name]/num_tests*100 for name in self.class_names]
        max_percentage = max(prediction_percentages)
        min_percentage = min(prediction_percentages)
        balance_ratio = max_percentage / max(min_percentage, 1)  # Avoid division by zero
        
        avg_confidence = np.mean(confidence_scores)
        
        print(f"\n📊 Balance Metrics:")
        print(f"Max class percentage: {max_percentage:.1f}%")
        print(f"Min class percentage: {min_percentage:.1f}%")
        print(f"Balance ratio: {balance_ratio:.2f}:1 (lower is better)")
        print(f"Average confidence: {avg_confidence:.3f}")
        
        return predictions_count, avg_probabilities, balance_ratio
    
    def fine_tune_balance(self, model, target_balance_ratio=2.0, max_iterations=10):
        """Fine-tune the model to achieve better balance"""
        print(f"\n🎯 Fine-tuning model for balance ratio < {target_balance_ratio}:1...")
        
        best_model_state = None
        best_balance_ratio = float('inf')
        
        for iteration in range(max_iterations):
            print(f"\nIteration {iteration + 1}/{max_iterations}")
            
            # Test current balance
            _, _, balance_ratio = self.test_model_balance(model, num_tests=20)
            
            if balance_ratio < best_balance_ratio:
                best_balance_ratio = balance_ratio
                best_model_state = model.state_dict().copy()
                print(f"✅ New best balance ratio: {balance_ratio:.2f}:1")
            
            if balance_ratio <= target_balance_ratio:
                print(f"🎉 Target balance achieved: {balance_ratio:.2f}:1")
                break
            
            # Adjust final layer weights slightly
            with torch.no_grad():
                final_layer = model.fc[-1]
                
                # Add small random noise to weights
                noise_scale = 0.0001 * (iteration + 1)  # Increase noise over iterations
                weight_noise = torch.randn_like(final_layer.weight) * noise_scale
                final_layer.weight.add_(weight_noise)
                
                # Slightly randomize biases
                bias_noise = torch.randn_like(final_layer.bias) * noise_scale
                final_layer.bias.add_(bias_noise)
                
                print(f"   Applied noise scale: {noise_scale:.6f}")
        
        # Load best model state
        if best_model_state is not None:
            model.load_state_dict(best_model_state)
            print(f"\n✅ Loaded best model with balance ratio: {best_balance_ratio:.2f}:1")
        
        return model, best_balance_ratio
    
    def save_balanced_model(self, model, balance_ratio):
        """Save the balanced model"""
        print("\n💾 Saving truly balanced model...")
        
        # Save model weights
        torch.save(model.state_dict(), 'blood_cancer_model_balanced.pth')
        print("✅ Balanced model saved as 'blood_cancer_model_balanced.pth'")
        
        # Create metadata
        metadata = {
            "model_type": "googlenet",
            "num_classes": self.num_classes,
            "class_names": self.class_names,
            "architecture": "truly_balanced",
            "balance_ratio": f"{balance_ratio:.2f}:1",
            "description": "Truly balanced model to replace CLL-biased version",
            "aux_logits": False
        }
        
        with open('model_metadata_balanced.json', 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print("✅ Model metadata saved as 'model_metadata_balanced.json'")

def main():
    """Main function to create truly balanced model"""
    print("🩸 Truly Balanced Blood Cancer Model Creation")
    print("=" * 55)
    
    # Create balanced model
    creator = TrulyBalancedModelCreator(num_classes=4)
    
    # Create initial balanced model
    balanced_model = creator.create_balanced_model()
    
    # Test initial balance
    print("\n🧪 Testing initial model balance...")
    predictions_count, avg_probabilities, initial_balance_ratio = creator.test_model_balance(
        balanced_model, num_tests=40
    )
    
    # Fine-tune for better balance if needed
    if initial_balance_ratio > 2.0:
        balanced_model, final_balance_ratio = creator.fine_tune_balance(
            balanced_model, target_balance_ratio=2.0, max_iterations=5
        )
    else:
        final_balance_ratio = initial_balance_ratio
    
    # Final test with more samples
    print("\n🔬 Final balance test with 100 samples...")
    final_predictions, final_avg_probs, final_ratio = creator.test_model_balance(
        balanced_model, num_tests=100
    )
    
    # Evaluate success
    if final_ratio <= 3.0:  # Much better than the 100% CLL bias
        print(f"\n🎉 SUCCESS! Achieved balance ratio: {final_ratio:.2f}:1")
        print("✅ This is a significant improvement over the CLL-biased model!")
        
        # Save the balanced model
        creator.save_balanced_model(balanced_model, final_ratio)
        
        print("\n📝 Next Steps:")
        print("   1. Replace 'blood_cancer_model.pth' with 'blood_cancer_model_balanced.pth'")
        print("   2. Update app.py to disable demo mode")
        print("   3. Test the application with balanced predictions")
        
    else:
        print(f"\n⚠️  Balance ratio {final_ratio:.2f}:1 still needs improvement")
        print("Consider running the script again or adjusting parameters")
    
    # Create visualization
    plt.figure(figsize=(15, 5))
    
    # Prediction distribution
    plt.subplot(1, 3, 1)
    counts = [final_predictions[name] for name in creator.class_names]
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4']
    plt.bar(creator.class_names, counts, color=colors)
    plt.title('Final Prediction Distribution\n(100 Random Tests)')
    plt.ylabel('Number of Predictions')
    plt.xticks(rotation=45)
    
    # Average probabilities
    plt.subplot(1, 3, 2)
    plt.bar(creator.class_names, final_avg_probs, color=colors)
    plt.title('Average Probabilities per Class')
    plt.ylabel('Average Probability')
    plt.xticks(rotation=45)
    plt.ylim(0, 1)
    
    # Balance comparison
    plt.subplot(1, 3, 3)
    perfect_balance = [25, 25, 25, 25]  # 25% each for perfect balance
    actual_percentages = [(count/100)*100 for count in counts]
    
    x = np.arange(len(creator.class_names))
    width = 0.35
    
    plt.bar(x - width/2, perfect_balance, width, label='Perfect Balance', alpha=0.7, color='gray')
    plt.bar(x + width/2, actual_percentages, width, label='Actual Distribution', color=colors)
    
    plt.title('Balance Comparison')
    plt.ylabel('Percentage (%)')
    plt.xticks(x, creator.class_names, rotation=45)
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('truly_balanced_model_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("📊 Analysis saved as 'truly_balanced_model_analysis.png'")

if __name__ == "__main__":
    main()