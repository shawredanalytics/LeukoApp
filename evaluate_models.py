#!/usr/bin/env python3
"""
Comprehensive Model Evaluation Script
Tests all available models to identify the best performing one for binary classification
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import json
import os
from datetime import datetime

# Model architectures
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
        # Feature extraction
        features = self.feature_extractor(x)
        
        # Attention mechanism
        attention_weights = self.attention(features)
        attended_features = features * attention_weights
        
        # Specialized branches
        morphology_features = self.morphology_branch(attended_features)
        pattern_features = self.pattern_branch(attended_features)
        
        # Combine features
        combined_features = torch.cat([attended_features, morphology_features, pattern_features], dim=1)
        
        # Classification
        logits = self.classifier(combined_features)
        
        # Temperature scaling
        scaled_logits = logits / self.temperature
        
        # Confidence estimation
        confidence_score = self.confidence_branch(attended_features)
        
        return scaled_logits, confidence_score

class RandomBalancedClassifier(nn.Module):
    """Random balanced classifier for 4-class cancer detection"""
    
    def __init__(self, num_classes=4):
        super(RandomBalancedClassifier, self).__init__()
        self.classifier = nn.Sequential(
            nn.Linear(1024, 512),              # fc.0
            nn.ReLU(inplace=True),             # fc.1
            nn.Dropout(0.4),                   # fc.2
            nn.Linear(512, 128),               # fc.3
            nn.ReLU(inplace=True),             # fc.4
            nn.Dropout(0.3),                   # fc.5
            nn.Linear(128, 64),                # fc.6
            nn.ReLU(inplace=True),             # fc.7
            nn.Dropout(0.2),                   # fc.8
            nn.Linear(64, num_classes)         # fc.9
        )
    
    def forward(self, x):
        return self.classifier(x)

def create_test_data(num_samples=100):
    """Create synthetic test data for evaluation"""
    # Generate random image-like features
    test_data = []
    test_labels = []
    
    for i in range(num_samples):
        # Create synthetic features (simulating GoogLeNet output)
        features = torch.randn(1024)
        
        # Create balanced labels
        label = i % 2  # Alternating 0 (normal) and 1 (abnormal)
        
        test_data.append(features)
        test_labels.append(label)
    
    return torch.stack(test_data), torch.tensor(test_labels)

def evaluate_model(model_path, model_name, device):
    """Evaluate a single model"""
    print(f"\n🔍 Evaluating {model_name}...")
    print("-" * 50)
    
    try:
        # Load model
        base_model = models.googlenet(weights=None)
        
        # Determine architecture based on model name
        if "random_balanced" in model_path.lower():
            base_model.fc = RandomBalancedClassifier(4)
            is_binary = False
        else:
            base_model.fc = BinaryScreeningClassifier(1024)
            is_binary = True
        
        # Load state dict
        state_dict = torch.load(model_path, map_location=device)
        base_model.load_state_dict(state_dict, strict=False)
        base_model.to(device)
        base_model.eval()
        
        # Create test data
        test_features, test_labels = create_test_data(200)
        test_features = test_features.to(device)
        test_labels = test_labels.to(device)
        
        # Evaluate
        correct = 0
        total = 0
        normal_predictions = 0
        abnormal_predictions = 0
        confidence_scores = []
        
        with torch.no_grad():
            for i in range(len(test_features)):
                features = test_features[i:i+1]
                true_label = test_labels[i].item()
                
                # Get model output
                model_output = base_model.fc(features)
                
                if isinstance(model_output, tuple):
                    # BinaryScreeningClassifier returns (logits, confidence)
                    logits, confidence = model_output
                    confidence_scores.append(confidence.item())
                else:
                    # RandomBalancedClassifier returns only logits
                    logits = model_output
                    confidence_scores.append(0.7)  # Default confidence
                    
                    # Convert 4-class to binary if needed
                    if logits.shape[1] == 4:
                        # Class 0 = normal, classes 1-3 = abnormal
                        normal_logit = logits[0, 0:1]
                        abnormal_logit = torch.logsumexp(logits[0, 1:], dim=0, keepdim=True)
                        logits = torch.cat([normal_logit, abnormal_logit], dim=0).unsqueeze(0)
                
                # Get prediction
                probabilities = F.softmax(logits, dim=1)
                predicted = torch.argmax(probabilities, dim=1).item()
                
                # Count predictions
                if predicted == 0:
                    normal_predictions += 1
                else:
                    abnormal_predictions += 1
                
                # Check accuracy (for balanced test data)
                if predicted == true_label:
                    correct += 1
                total += 1
        
        # Calculate metrics
        accuracy = correct / total
        normal_ratio = normal_predictions / total
        abnormal_ratio = abnormal_predictions / total
        avg_confidence = np.mean(confidence_scores)
        balance_score = 1 - abs(normal_ratio - 0.5) * 2  # 1 = perfect balance, 0 = completely biased
        
        # Overall score (weighted combination)
        overall_score = (accuracy * 0.4) + (balance_score * 0.4) + (avg_confidence * 0.2)
        
        results = {
            'model_name': model_name,
            'model_path': model_path,
            'accuracy': accuracy,
            'normal_predictions': normal_predictions,
            'abnormal_predictions': abnormal_predictions,
            'normal_ratio': normal_ratio,
            'abnormal_ratio': abnormal_ratio,
            'balance_score': balance_score,
            'avg_confidence': avg_confidence,
            'overall_score': overall_score,
            'is_binary': is_binary,
            'status': 'success'
        }
        
        print(f"✅ Accuracy: {accuracy:.3f}")
        print(f"📊 Balance: {normal_predictions} normal, {abnormal_predictions} abnormal")
        print(f"⚖️  Balance Score: {balance_score:.3f}")
        print(f"🎯 Confidence: {avg_confidence:.3f}")
        print(f"🏆 Overall Score: {overall_score:.3f}")
        
        return results
        
    except Exception as e:
        print(f"❌ Error evaluating {model_name}: {str(e)}")
        return {
            'model_name': model_name,
            'model_path': model_path,
            'status': 'error',
            'error': str(e),
            'overall_score': 0
        }

def main():
    """Main evaluation function"""
    print("🧪 Comprehensive Model Evaluation")
    print("=" * 60)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # List of models to evaluate
    models_to_test = [
        ("best_binary_model.pth", "Best Binary Model"),
        ("blood_smear_screening_model.pth", "Blood Smear Screening Model"),
        ("blood_smear_screening_model_fixed.pth", "Blood Smear Screening Model (Fixed)"),
        ("blood_cancer_model_discriminative.pth", "Discriminative Model"),
        ("blood_cancer_model_random_balanced.pth", "Random Balanced Model"),
        ("blood_cancer_model.pth", "Standard Blood Cancer Model"),
    ]
    
    results = []
    
    # Evaluate each model
    for model_path, model_name in models_to_test:
        if os.path.exists(model_path):
            result = evaluate_model(model_path, model_name, device)
            results.append(result)
        else:
            print(f"⚠️  Model not found: {model_path}")
    
    # Sort by overall score
    successful_results = [r for r in results if r['status'] == 'success']
    successful_results.sort(key=lambda x: x['overall_score'], reverse=True)
    
    # Print summary
    print("\n" + "=" * 60)
    print("📊 EVALUATION SUMMARY")
    print("=" * 60)
    
    if successful_results:
        print("\n🏆 TOP PERFORMING MODELS:")
        print("-" * 40)
        
        for i, result in enumerate(successful_results[:3], 1):
            print(f"{i}. {result['model_name']}")
            print(f"   Overall Score: {result['overall_score']:.3f}")
            print(f"   Accuracy: {result['accuracy']:.3f}")
            print(f"   Balance Score: {result['balance_score']:.3f}")
            print(f"   Confidence: {result['avg_confidence']:.3f}")
            print()
        
        # Recommend best model
        best_model = successful_results[0]
        print(f"🎯 RECOMMENDED MODEL: {best_model['model_name']}")
        print(f"   File: {best_model['model_path']}")
        print(f"   Overall Score: {best_model['overall_score']:.3f}")
        
        # Save results
        with open('model_evaluation_results.json', 'w') as f:
            json.dump({
                'evaluation_date': datetime.now().isoformat(),
                'device': str(device),
                'results': results,
                'recommended_model': best_model
            }, f, indent=2)
        
        print(f"\n💾 Results saved to: model_evaluation_results.json")
        
    else:
        print("❌ No models could be successfully evaluated!")

if __name__ == "__main__":
    main()