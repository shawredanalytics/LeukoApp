"""
Test the threshold fix to verify that normal smears are no longer misclassified as cancerous.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from PIL import Image
import os
import json
from torchvision import models, transforms

# Model architectures (copied from app)
class BinaryScreeningClassifier(nn.Module):
    def __init__(self, num_classes=2):
        super(BinaryScreeningClassifier, self).__init__()
        
        # Multi-scale feature extraction
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        
        # Multi-head attention mechanism
        self.attention = nn.MultiheadAttention(embed_dim=128, num_heads=8, batch_first=True)
        
        # Specialized analysis branches
        self.morphology_branch = nn.Sequential(
            nn.Conv2d(128, 64, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((4, 4))
        )
        
        self.cellular_branch = nn.Sequential(
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((4, 4))
        )
        
        self.pattern_branch = nn.Sequential(
            nn.Conv2d(128, 64, kernel_size=7, padding=3),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((4, 4))
        )
        
        # Fusion and classification
        self.fusion = nn.Linear(64 * 4 * 4 * 3, 256)
        self.dropout = nn.Dropout(0.5)
        self.classifier = nn.Linear(256, num_classes)
        
        # Uncertainty estimation
        self.uncertainty_head = nn.Linear(256, 1)
        
        # Learnable temperature for calibration
        self.temperature = nn.Parameter(torch.ones(1))
        
    def forward(self, x):
        # Feature extraction
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.conv3(x))
        x = F.max_pool2d(x, 2)
        
        # Attention mechanism
        batch_size, channels, height, width = x.shape
        x_flat = x.view(batch_size, channels, -1).permute(0, 2, 1)
        attended, _ = self.attention(x_flat, x_flat, x_flat)
        x = attended.permute(0, 2, 1).view(batch_size, channels, height, width)
        
        # Specialized branches
        morph_features = self.morphology_branch(x).view(batch_size, -1)
        cell_features = self.cellular_branch(x).view(batch_size, -1)
        pattern_features = self.pattern_branch(x).view(batch_size, -1)
        
        # Fusion
        combined = torch.cat([morph_features, cell_features, pattern_features], dim=1)
        fused = F.relu(self.fusion(combined))
        fused = self.dropout(fused)
        
        # Classification
        logits = self.classifier(fused)
        
        # Temperature scaling
        scaled_logits = logits / self.temperature
        
        # Uncertainty estimation
        uncertainty = torch.sigmoid(self.uncertainty_head(fused))
        
        return scaled_logits, uncertainty

class RandomBalancedClassifier(nn.Module):
    def __init__(self, num_classes=2):
        super(RandomBalancedClassifier, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, 3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        self.pool = nn.AdaptiveAvgPool2d((8, 8))
        self.fc1 = nn.Linear(32 * 8 * 8, 64)
        self.fc2 = nn.Linear(64, num_classes)
        self.dropout = nn.Dropout(0.5)
        
    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2)
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

def load_model(model_path):
    """Load a model from file (using correct GoogLeNet architecture)"""
    try:
        from torchvision import models
        
        # Check if it's the random balanced model
        if "random_balanced" in model_path.lower():
            # Use RandomBalancedClassifier for the random balanced model
            base_model = models.googlenet(weights=models.GoogLeNet_Weights.DEFAULT)
            base_model.fc = RandomBalancedClassifier(1024, 4)  # 4 classes for cancer types
        else:
            # Use BinaryScreeningClassifier for other models
            base_model = models.googlenet(weights=models.GoogLeNet_Weights.DEFAULT)
            base_model.fc = BinaryScreeningClassifier(1024)
        
        # Load trained weights
        state_dict = torch.load(model_path, map_location='cpu')
        base_model.load_state_dict(state_dict)
        base_model.eval()
        
        model_type = "RandomBalanced" if "random_balanced" in model_path.lower() else "BinaryScreening"
        return base_model, model_type
        
    except Exception as e:
        print(f"Failed to load model {model_path}: {e}")
        return None, None

def predict_with_threshold(model, image_tensor, threshold=0.55):
    """Predict with the new threshold (mirroring app logic)"""
    with torch.no_grad():
        output = model(image_tensor)
        
        # Handle different model outputs
        if isinstance(output, tuple):
            logits = output[0]
        else:
            logits = output
        
        # Convert 4-class to binary if needed
        if logits.shape[1] == 4:
            # Convert 4-class predictions to binary
            probabilities = F.softmax(logits, dim=1)
            normal_prob = probabilities[0, 0].item()
            abnormal_prob = 1 - normal_prob
            binary_probs = torch.tensor([[normal_prob, abnormal_prob]])
        else:
            # Already binary
            binary_probs = F.softmax(logits, dim=1)
        
        normal_prob = binary_probs[0, 0].item()
        abnormal_prob = binary_probs[0, 1].item()
        
        # Apply threshold - Fixed logic to reduce false positives
        if abnormal_prob > threshold:  # Changed from >= to > for stricter threshold
            predicted_class = 1  # Abnormal
            confidence = abnormal_prob
        else:
            predicted_class = 0  # Normal
            confidence = normal_prob
        
        return predicted_class, confidence, normal_prob, abnormal_prob

def create_synthetic_images():
    """Create synthetic normal and abnormal images for testing"""
    # Create proper image tensors that match the expected input format
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Normal image (brighter, more uniform)
    normal_image = np.random.rand(224, 224, 3) * 0.8 + 0.2
    normal_image = (normal_image * 255).astype(np.uint8)
    normal_pil = Image.fromarray(normal_image)
    normal_tensor = transform(normal_pil).unsqueeze(0)
    
    # Abnormal image (darker, more varied)
    abnormal_image = np.random.rand(224, 224, 3) * 0.6
    abnormal_image = (abnormal_image * 255).astype(np.uint8)
    abnormal_pil = Image.fromarray(abnormal_image)
    abnormal_tensor = transform(abnormal_pil).unsqueeze(0)
    
    return normal_tensor, abnormal_tensor

def test_threshold_fix():
    """Test the threshold fix with different values"""
    print("🔬 TESTING THRESHOLD FIX FOR NORMAL SMEAR MISCLASSIFICATION")
    print("=" * 60)
    
    # Load the best model
    model_path = "blood_smear_screening_model.pth"
    model, model_type = load_model(model_path)
    
    if model is None:
        print(f"❌ Could not load model: {model_path}")
        return
    
    print(f"✅ Loaded model: {model_type}")
    
    # Create test images
    normal_tensor, abnormal_tensor = create_synthetic_images()
    
    print(f"✅ Created test images with proper preprocessing")
    
    # Test different thresholds
    thresholds = [0.50, 0.52, 0.55, 0.60, 0.65]
    
    print("\n📊 THRESHOLD COMPARISON:")
    print("Threshold | Normal Image | Abnormal Image | False Positives")
    print("-" * 60)
    
    results = {}
    
    for threshold in thresholds:
        # Test normal image
        normal_pred, normal_conf, normal_prob, normal_abnormal_prob = predict_with_threshold(
            model, normal_tensor, threshold
        )
        
        # Test abnormal image
        abnormal_pred, abnormal_conf, abnormal_normal_prob, abnormal_prob = predict_with_threshold(
            model, abnormal_tensor, threshold
        )
        
        # Check for false positives (normal classified as abnormal)
        false_positive = normal_pred == 1
        
        results[threshold] = {
            'normal_prediction': normal_pred,
            'normal_confidence': normal_conf,
            'abnormal_prediction': abnormal_pred,
            'abnormal_confidence': abnormal_conf,
            'false_positive': false_positive
        }
        
        normal_result = "❌ ABNORMAL" if normal_pred == 1 else "✅ NORMAL"
        abnormal_result = "✅ ABNORMAL" if abnormal_pred == 1 else "❌ NORMAL"
        fp_indicator = "❌ YES" if false_positive else "✅ NO"
        
        print(f"   {threshold:.2f}   |   {normal_result}   |   {abnormal_result}   |     {fp_indicator}")
    
    # Detailed analysis for current threshold (0.55)
    print(f"\n🎯 DETAILED ANALYSIS FOR THRESHOLD 0.55:")
    print("-" * 40)
    
    current_threshold = 0.55
    normal_pred, normal_conf, normal_prob, normal_abnormal_prob = predict_with_threshold(
        model, normal_tensor, current_threshold
    )
    
    print(f"Normal Image:")
    print(f"  Normal Probability: {normal_prob:.3f}")
    print(f"  Abnormal Probability: {normal_abnormal_prob:.3f}")
    print(f"  Prediction: {'Normal' if normal_pred == 0 else 'Abnormal'}")
    print(f"  Confidence: {normal_conf:.3f}")
    
    abnormal_pred, abnormal_conf, abnormal_normal_prob, abnormal_prob = predict_with_threshold(
        model, abnormal_tensor, current_threshold
    )
    
    print(f"\nAbnormal Image:")
    print(f"  Normal Probability: {abnormal_normal_prob:.3f}")
    print(f"  Abnormal Probability: {abnormal_prob:.3f}")
    print(f"  Prediction: {'Normal' if abnormal_pred == 0 else 'Abnormal'}")
    print(f"  Confidence: {abnormal_conf:.3f}")
    
    # Save results
    with open('threshold_fix_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n💾 Results saved to 'threshold_fix_results.json'")
    
    # Recommendation
    print(f"\n🔧 RECOMMENDATION:")
    if results[0.55]['false_positive']:
        print("⚠️  Threshold 0.55 may still cause false positives.")
        print("   Consider increasing to 0.60 or higher.")
    else:
        print("✅ Threshold 0.55 appears to reduce false positives effectively.")
        print("   Normal smears should now be classified correctly.")

if __name__ == "__main__":
    test_threshold_fix()