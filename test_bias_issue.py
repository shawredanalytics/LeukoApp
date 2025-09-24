import torch
import torch.nn as nn
import numpy as np
from PIL import Image
import json
import os

# Import the model architectures from the app
class BinaryScreeningClassifier(nn.Module):
    def __init__(self, num_classes=2):
        super(BinaryScreeningClassifier, self).__init__()
        
        # Enhanced Feature Extractor
        self.feature_extractor = nn.Sequential(
            # First conv block
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            
            # Second conv block
            nn.Conv2d(64, 128, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            # Third conv block
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            # Fourth conv block
            nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((7, 7))
        )
        
        # Attention mechanism
        self.attention = nn.Sequential(
            nn.Conv2d(512, 256, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 1, kernel_size=1),
            nn.Sigmoid()
        )
        
        # Specialized branches
        self.morphology_branch = nn.Sequential(
            nn.Linear(512 * 7 * 7, 1024),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(1024, 256)
        )
        
        self.pattern_branch = nn.Sequential(
            nn.Linear(512 * 7 * 7, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(512, 128)
        )
        
        # Confidence estimation branch
        self.confidence_branch = nn.Sequential(
            nn.Linear(256 + 128, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )
        
        # Final classifier
        self.classifier = nn.Sequential(
            nn.Linear(256 + 128, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )
        
        # Temperature scaling for calibration
        self.temperature = nn.Parameter(torch.ones(1))
    
    def forward(self, x):
        # Feature extraction
        features = self.feature_extractor(x)
        
        # Attention
        attention_weights = self.attention(features)
        attended_features = features * attention_weights
        
        # Flatten
        flattened = attended_features.view(attended_features.size(0), -1)
        
        # Specialized branches
        morphology_features = self.morphology_branch(flattened)
        pattern_features = self.pattern_branch(flattened)
        
        # Combine features
        combined_features = torch.cat([morphology_features, pattern_features], dim=1)
        
        # Confidence estimation
        confidence = self.confidence_branch(combined_features)
        
        # Classification
        logits = self.classifier(combined_features)
        
        # Temperature scaling
        scaled_logits = logits / self.temperature
        
        return scaled_logits, confidence

class RandomBalancedClassifier(nn.Module):
    def __init__(self, num_classes=4):
        super(RandomBalancedClassifier, self).__init__()
        self.classifier = nn.Sequential(
            nn.Linear(224 * 224 * 3, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )
    
    def forward(self, x):
        x = x.view(x.size(0), -1)
        return self.classifier(x)

def create_test_samples():
    """Create synthetic test samples representing normal and cancerous cells"""
    # Normal cell characteristics (lower intensity, more uniform)
    normal_samples = []
    for i in range(10):
        # Create a more uniform, lighter image (normal cells)
        img = np.random.normal(0.6, 0.1, (224, 224, 3))  # Higher mean, lower variance
        img = np.clip(img, 0, 1)
        normal_samples.append(torch.FloatTensor(img).permute(2, 0, 1).unsqueeze(0))
    
    # Cancerous cell characteristics (higher contrast, more irregular)
    cancerous_samples = []
    for i in range(10):
        # Create a more irregular, darker image (cancerous cells)
        img = np.random.normal(0.3, 0.3, (224, 224, 3))  # Lower mean, higher variance
        img = np.clip(img, 0, 1)
        cancerous_samples.append(torch.FloatTensor(img).permute(2, 0, 1).unsqueeze(0))
    
    return normal_samples, cancerous_samples

def test_model_bias(model_path):
    """Test a model for bias issues"""
    print(f"\n=== Testing Model: {model_path} ===")
    
    if not os.path.exists(model_path):
        print(f"Model file not found: {model_path}")
        return None
    
    try:
        # Load model
        checkpoint = torch.load(model_path, map_location='cpu')
        
        # Determine architecture
        if "random_balanced" in model_path.lower():
            model = RandomBalancedClassifier(num_classes=4)
            model.load_state_dict(checkpoint)
            model_type = "RandomBalanced"
        else:
            model = BinaryScreeningClassifier(num_classes=2)
            if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                model.load_state_dict(checkpoint['model_state_dict'])
            else:
                model.load_state_dict(checkpoint)
            model_type = "BinaryScreening"
        
        model.eval()
        
        # Create test samples
        normal_samples, cancerous_samples = create_test_samples()
        
        results = {
            'model_path': model_path,
            'model_type': model_type,
            'normal_predictions': [],
            'cancerous_predictions': [],
            'normal_confidences': [],
            'cancerous_confidences': []
        }
        
        # Test normal samples
        print("Testing normal samples...")
        with torch.no_grad():
            for i, sample in enumerate(normal_samples):
                if model_type == "RandomBalanced":
                    output = model(sample)
                    if len(output.shape) > 1 and output.shape[1] == 4:
                        # Convert 4-class to binary (classes 0,1 = normal, classes 2,3 = cancerous)
                        normal_prob = torch.softmax(output, dim=1)[:, :2].sum(dim=1)
                        cancerous_prob = torch.softmax(output, dim=1)[:, 2:].sum(dim=1)
                        binary_logits = torch.stack([normal_prob, cancerous_prob], dim=1)
                        prediction = torch.argmax(binary_logits, dim=1).item()
                        confidence = torch.max(torch.softmax(binary_logits, dim=1), dim=1)[0].item()
                    else:
                        prediction = torch.argmax(output, dim=1).item()
                        confidence = torch.max(torch.softmax(output, dim=1), dim=1)[0].item()
                else:
                    output, conf = model(sample)
                    prediction = torch.argmax(output, dim=1).item()
                    confidence = conf.item()
                
                results['normal_predictions'].append(prediction)
                results['normal_confidences'].append(confidence)
                print(f"  Normal sample {i+1}: Prediction={prediction} ({'Normal' if prediction == 0 else 'Cancerous'}), Confidence={confidence:.3f}")
        
        # Test cancerous samples
        print("Testing cancerous samples...")
        with torch.no_grad():
            for i, sample in enumerate(cancerous_samples):
                if model_type == "RandomBalanced":
                    output = model(sample)
                    if len(output.shape) > 1 and output.shape[1] == 4:
                        # Convert 4-class to binary
                        normal_prob = torch.softmax(output, dim=1)[:, :2].sum(dim=1)
                        cancerous_prob = torch.softmax(output, dim=1)[:, 2:].sum(dim=1)
                        binary_logits = torch.stack([normal_prob, cancerous_prob], dim=1)
                        prediction = torch.argmax(binary_logits, dim=1).item()
                        confidence = torch.max(torch.softmax(binary_logits, dim=1), dim=1)[0].item()
                    else:
                        prediction = torch.argmax(output, dim=1).item()
                        confidence = torch.max(torch.softmax(output, dim=1), dim=1)[0].item()
                else:
                    output, conf = model(sample)
                    prediction = torch.argmax(output, dim=1).item()
                    confidence = conf.item()
                
                results['cancerous_predictions'].append(prediction)
                results['cancerous_confidences'].append(confidence)
                print(f"  Cancerous sample {i+1}: Prediction={prediction} ({'Normal' if prediction == 0 else 'Cancerous'}), Confidence={confidence:.3f}")
        
        # Calculate bias metrics
        normal_correct = sum(1 for p in results['normal_predictions'] if p == 0)
        cancerous_correct = sum(1 for p in results['cancerous_predictions'] if p == 1)
        
        false_positive_rate = (10 - normal_correct) / 10  # Normal samples predicted as cancerous
        false_negative_rate = (10 - cancerous_correct) / 10  # Cancerous samples predicted as normal
        
        results['normal_accuracy'] = normal_correct / 10
        results['cancerous_accuracy'] = cancerous_correct / 10
        results['false_positive_rate'] = false_positive_rate
        results['false_negative_rate'] = false_negative_rate
        results['avg_normal_confidence'] = np.mean(results['normal_confidences'])
        results['avg_cancerous_confidence'] = np.mean(results['cancerous_confidences'])
        
        print(f"\n--- Bias Analysis ---")
        print(f"Normal samples correctly classified: {normal_correct}/10 ({results['normal_accuracy']:.1%})")
        print(f"Cancerous samples correctly classified: {cancerous_correct}/10 ({results['cancerous_accuracy']:.1%})")
        print(f"False Positive Rate (Normal→Cancerous): {false_positive_rate:.1%}")
        print(f"False Negative Rate (Cancerous→Normal): {false_negative_rate:.1%}")
        print(f"Average confidence on normal samples: {results['avg_normal_confidence']:.3f}")
        print(f"Average confidence on cancerous samples: {results['avg_cancerous_confidence']:.3f}")
        
        if false_positive_rate > 0.3:
            print("⚠️  HIGH BIAS DETECTED: Model is incorrectly classifying normal samples as cancerous!")
        
        return results
        
    except Exception as e:
        print(f"Error testing model {model_path}: {str(e)}")
        return None

def main():
    """Test all available models for bias"""
    models_to_test = [
        "blood_smear_screening_model.pth",
        "blood_cancer_model_random_balanced.pth", 
        "best_binary_model.pth",
        "blood_smear_screening_model_fixed.pth"
    ]
    
    all_results = []
    
    print("🔍 Testing models for bias issues...")
    print("=" * 60)
    
    for model_path in models_to_test:
        result = test_model_bias(model_path)
        if result:
            all_results.append(result)
    
    # Summary
    print("\n" + "=" * 60)
    print("📊 BIAS ANALYSIS SUMMARY")
    print("=" * 60)
    
    for result in all_results:
        print(f"\nModel: {result['model_path']}")
        print(f"  Type: {result['model_type']}")
        print(f"  False Positive Rate: {result['false_positive_rate']:.1%}")
        print(f"  Normal Accuracy: {result['normal_accuracy']:.1%}")
        print(f"  Cancerous Accuracy: {result['cancerous_accuracy']:.1%}")
        
        if result['false_positive_rate'] > 0.3:
            print(f"  ⚠️  BIAS ISSUE: High false positive rate!")
        else:
            print(f"  ✅ Acceptable bias levels")
    
    # Save results
    with open('bias_analysis_results.json', 'w') as f:
        json.dump(all_results, f, indent=2)
    
    print(f"\n📁 Results saved to: bias_analysis_results.json")

if __name__ == "__main__":
    main()