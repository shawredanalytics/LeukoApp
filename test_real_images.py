import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
import os
import json
import numpy as np
from torchvision import transforms

# Import model architectures
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

def preprocess_image(image_path):
    """Preprocess image for model input"""
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    image = Image.open(image_path)
    if image.mode != 'RGB':
        image = image.convert('RGB')
    
    return transform(image).unsqueeze(0)

def predict_with_bias_correction(model, image_tensor, model_type):
    """Make prediction with bias correction similar to the app"""
    with torch.no_grad():
        model_output = model(image_tensor)
        
        # Check if model returns tuple or single tensor
        if isinstance(model_output, tuple):
            # BinaryScreeningClassifier returns (logits, confidence_score)
            logits, confidence_score = model_output
        else:
            # RandomBalancedClassifier or other models return only logits
            logits = model_output
            
            # For RandomBalancedClassifier (4 classes), convert to binary
            if logits.shape[1] == 4:
                # Convert 4-class cancer prediction to binary (normal vs abnormal)
                normal_logit = logits[0, 0:1]  # Keep first class as normal
                abnormal_logit = torch.logsumexp(logits[0, 1:], dim=0, keepdim=True)  # Combine cancer classes
                logits = torch.cat([normal_logit, abnormal_logit], dim=0).unsqueeze(0)
            
            # Generate confidence score
            confidence_score = torch.rand(1, 1) * 0.3 + 0.4
        
        # Calculate probabilities
        probabilities = F.softmax(logits, dim=1)
        
        # BIAS CORRECTION: Adjust threshold to compensate for model bias
        ABNORMAL_THRESHOLD = 0.52
        
        normal_prob = probabilities[0, 0].item()
        abnormal_prob = probabilities[0, 1].item()
        
        # Apply bias correction
        if abnormal_prob >= ABNORMAL_THRESHOLD:
            predicted_class = 1  # Abnormal
            max_probability = abnormal_prob
        else:
            predicted_class = 0  # Normal
            max_probability = normal_prob
        
        return {
            'predicted_class': predicted_class,
            'predicted_label': "Normal Smear" if predicted_class == 0 else "WBC Cancerous Abnormalities",
            'confidence': max_probability,
            'normal_prob': normal_prob,
            'abnormal_prob': abnormal_prob,
            'raw_logits': logits.numpy(),
            'raw_probabilities': probabilities.numpy()
        }

def test_model_on_real_images(model_path):
    """Test model on real test images"""
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
        
        # Test on normal images
        normal_dir = "test_images/normal"
        leukemic_dir = "test_images/leukemic"
        
        results = {
            'model_path': model_path,
            'model_type': model_type,
            'normal_results': [],
            'leukemic_results': []
        }
        
        # Test normal images
        if os.path.exists(normal_dir):
            print("\n--- Testing Normal Images ---")
            for filename in os.listdir(normal_dir):
                if filename.endswith(('.svg', '.png', '.jpg', '.jpeg')):
                    image_path = os.path.join(normal_dir, filename)
                    try:
                        image_tensor = preprocess_image(image_path)
                        prediction = predict_with_bias_correction(model, image_tensor, model_type)
                        prediction['filename'] = filename
                        results['normal_results'].append(prediction)
                        
                        status = "✅ CORRECT" if prediction['predicted_class'] == 0 else "❌ WRONG (False Positive)"
                        print(f"  {filename}: {prediction['predicted_label']} (conf: {prediction['confidence']:.3f}) {status}")
                        
                    except Exception as e:
                        print(f"  Error processing {filename}: {str(e)}")
        
        # Test leukemic images
        if os.path.exists(leukemic_dir):
            print("\n--- Testing Leukemic Images ---")
            for filename in os.listdir(leukemic_dir):
                if filename.endswith(('.svg', '.png', '.jpg', '.jpeg')):
                    image_path = os.path.join(leukemic_dir, filename)
                    try:
                        image_tensor = preprocess_image(image_path)
                        prediction = predict_with_bias_correction(model, image_tensor, model_type)
                        prediction['filename'] = filename
                        results['leukemic_results'].append(prediction)
                        
                        status = "✅ CORRECT" if prediction['predicted_class'] == 1 else "❌ WRONG (False Negative)"
                        print(f"  {filename}: {prediction['predicted_label']} (conf: {prediction['confidence']:.3f}) {status}")
                        
                    except Exception as e:
                        print(f"  Error processing {filename}: {str(e)}")
        
        # Calculate metrics
        normal_correct = sum(1 for r in results['normal_results'] if r['predicted_class'] == 0)
        normal_total = len(results['normal_results'])
        leukemic_correct = sum(1 for r in results['leukemic_results'] if r['predicted_class'] == 1)
        leukemic_total = len(results['leukemic_results'])
        
        if normal_total > 0:
            false_positive_rate = (normal_total - normal_correct) / normal_total
            normal_accuracy = normal_correct / normal_total
        else:
            false_positive_rate = 0
            normal_accuracy = 0
        
        if leukemic_total > 0:
            false_negative_rate = (leukemic_total - leukemic_correct) / leukemic_total
            leukemic_accuracy = leukemic_correct / leukemic_total
        else:
            false_negative_rate = 0
            leukemic_accuracy = 0
        
        results['metrics'] = {
            'normal_accuracy': normal_accuracy,
            'leukemic_accuracy': leukemic_accuracy,
            'false_positive_rate': false_positive_rate,
            'false_negative_rate': false_negative_rate,
            'normal_total': normal_total,
            'leukemic_total': leukemic_total
        }
        
        print(f"\n--- Results Summary ---")
        print(f"Normal images: {normal_correct}/{normal_total} correct ({normal_accuracy:.1%})")
        print(f"Leukemic images: {leukemic_correct}/{leukemic_total} correct ({leukemic_accuracy:.1%})")
        print(f"False Positive Rate: {false_positive_rate:.1%}")
        print(f"False Negative Rate: {false_negative_rate:.1%}")
        
        if false_positive_rate > 0.3:
            print("⚠️  HIGH BIAS: Normal images being classified as cancerous!")
        
        return results
        
    except Exception as e:
        print(f"Error testing model {model_path}: {str(e)}")
        return None

def main():
    """Test models on real images"""
    models_to_test = [
        "blood_smear_screening_model.pth",
        "blood_cancer_model_random_balanced.pth", 
        "best_binary_model.pth",
        "blood_smear_screening_model_fixed.pth"
    ]
    
    all_results = []
    
    print("🔍 Testing models on real test images...")
    print("=" * 60)
    
    for model_path in models_to_test:
        result = test_model_on_real_images(model_path)
        if result:
            all_results.append(result)
    
    # Summary
    print("\n" + "=" * 60)
    print("📊 REAL IMAGE BIAS ANALYSIS SUMMARY")
    print("=" * 60)
    
    for result in all_results:
        metrics = result['metrics']
        print(f"\nModel: {result['model_path']}")
        print(f"  Type: {result['model_type']}")
        print(f"  False Positive Rate: {metrics['false_positive_rate']:.1%}")
        print(f"  Normal Accuracy: {metrics['normal_accuracy']:.1%}")
        print(f"  Leukemic Accuracy: {metrics['leukemic_accuracy']:.1%}")
        
        if metrics['false_positive_rate'] > 0.3:
            print(f"  ⚠️  BIAS ISSUE: High false positive rate!")
        else:
            print(f"  ✅ Acceptable bias levels")
    
    # Save results
    with open('real_image_bias_results.json', 'w') as f:
        json.dump(all_results, f, indent=2, default=str)
    
    print(f"\n📁 Results saved to: real_image_bias_results.json")

if __name__ == "__main__":
    main()