import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import json
import os

# Binary Screening Model Architecture (same as in app)
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

def load_model():
    """Load the binary screening model (same as app)"""
    try:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Check if binary screening model exists
        screening_model_path = "blood_smear_screening_model.pth"
        
        if os.path.exists(screening_model_path):
            # Load binary screening model
            base_model = models.googlenet(weights=models.GoogLeNet_Weights.DEFAULT)
            base_model.fc = BinaryScreeningClassifier(1024)
            
            # Load trained weights
            state_dict = torch.load(screening_model_path, map_location=device)
            base_model.load_state_dict(state_dict)
            
            base_model.to(device)
            base_model.eval()
            
            return base_model, device, False  # Not demo mode
        else:
            print("❌ Binary screening model not found!")
            return None, None, True
            
    except Exception as e:
        print(f"❌ Error loading binary screening model: {str(e)}")
        return None, None, True

def create_test_image(image_type="normal"):
    """Create a synthetic test image for validation"""
    # Create a 224x224x3 test image
    if image_type == "normal":
        # Normal cells: more uniform, circular shapes
        image = np.random.rand(224, 224, 3) * 0.3 + 0.4  # Light background
        # Add some circular patterns (normal cells)
        for i in range(10):
            center_x, center_y = np.random.randint(20, 204, 2)
            radius = np.random.randint(8, 15)
            y, x = np.ogrid[:224, :224]
            mask = (x - center_x)**2 + (y - center_y)**2 <= radius**2
            image[mask] = [0.6, 0.4, 0.7]  # Purple-ish normal cells
    else:
        # Cancerous cells: more irregular, varied shapes and sizes
        image = np.random.rand(224, 224, 3) * 0.2 + 0.3  # Darker background
        # Add irregular patterns (abnormal cells)
        for i in range(15):
            center_x, center_y = np.random.randint(15, 209, 2)
            # Irregular shapes
            for j in range(np.random.randint(20, 40)):
                offset_x = np.random.randint(-12, 12)
                offset_y = np.random.randint(-12, 12)
                x_pos = max(0, min(223, center_x + offset_x))
                y_pos = max(0, min(223, center_y + offset_y))
                image[y_pos, x_pos] = [0.8, 0.2, 0.2]  # Red-ish abnormal cells
    
    return (image * 255).astype(np.uint8)

def predict_binary_screening(model, image_tensor, device, demo_mode=False):
    """Perform binary screening prediction (same as app)"""
    try:
        with torch.no_grad():
            image_tensor = image_tensor.to(device)
            
            if demo_mode:
                # Demo mode: simulate realistic binary predictions
                logits = torch.randn(1, 2) * 2.0
                confidence_score = torch.rand(1, 1) * 0.4 + 0.5  # 0.5-0.9 range
                
                # Add some bias towards normal for demo
                logits[0, 0] += 0.5  # Slight bias towards normal
            else:
                # Real model prediction
                if hasattr(model.fc, 'forward') and len(list(model.fc.parameters())) > 0:
                    # Model has binary screening architecture
                    # Get features from base model
                    features = model.conv1(image_tensor)
                    features = model.maxpool1(features)
                    features = model.conv2(features)
                    features = model.conv3(features)
                    features = model.maxpool2(features)
                    features = model.inception3a(features)
                    features = model.inception3b(features)
                    features = model.maxpool3(features)
                    features = model.inception4a(features)
                    features = model.inception4b(features)
                    features = model.inception4c(features)
                    features = model.inception4d(features)
                    features = model.inception4e(features)
                    features = model.maxpool4(features)
                    features = model.inception5a(features)
                    features = model.inception5b(features)
                    features = model.avgpool(features)
                    features = torch.flatten(features, 1)
                    
                    # Pass through binary screening classifier
                    logits, confidence_score = model.fc(features)
                else:
                    # Fallback to standard prediction
                    logits = model(image_tensor)
                    confidence_score = torch.rand(1, 1) * 0.3 + 0.4
            
            # Convert to probabilities
            probabilities = F.softmax(logits, dim=1)
            
            return probabilities, confidence_score, logits
            
    except Exception as e:
        print(f"❌ Prediction error: {str(e)}")
        return None, None, None

def test_model_validation():
    """Test the actual model used by the app"""
    print("🔍 Testing Actual Binary Screening Model")
    print("=" * 50)
    
    # Load model
    model, device, demo_mode = load_model()
    if model is None:
        print("❌ Failed to load model!")
        return
    
    if demo_mode:
        print("⚠️  Running in DEMO MODE - model not found")
    else:
        print("✅ Real model loaded successfully")
    
    print(f"🖥️  Device: {device}")
    print()
    
    # Define transforms
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Test with multiple samples
    test_cases = [
        ("Normal Sample 1", "normal"),
        ("Normal Sample 2", "normal"),
        ("Normal Sample 3", "normal"),
        ("Cancerous Sample 1", "cancerous"),
        ("Cancerous Sample 2", "cancerous"),
        ("Cancerous Sample 3", "cancerous"),
    ]
    
    results = []
    
    print("🧪 Testing Model Predictions:")
    print("-" * 30)
    
    for sample_name, sample_type in test_cases:
        # Create test image
        test_image = create_test_image(sample_type)
        
        # Transform image
        input_tensor = transform(test_image).unsqueeze(0)
        
        # Get prediction
        probabilities, confidence_score, logits = predict_binary_screening(
            model, input_tensor, device, demo_mode
        )
        
        if probabilities is None:
            print(f"❌ Failed to get prediction for {sample_name}")
            continue
        
        normal_prob = probabilities[0][0].item()
        leukemia_prob = probabilities[0][1].item()
        confidence = confidence_score[0].item()
        
        predicted_class = "Normal" if normal_prob > leukemia_prob else "Leukemia"
        expected_class = "Normal" if sample_type == "normal" else "Leukemia"
        
        is_correct = predicted_class == expected_class
        
        results.append({
            'sample': sample_name,
            'expected': expected_class,
            'predicted': predicted_class,
            'correct': is_correct,
            'normal_prob': normal_prob,
            'leukemia_prob': leukemia_prob,
            'confidence': confidence
        })
        
        status = "✅" if is_correct else "❌"
        print(f"{status} {sample_name}:")
        print(f"   Expected: {expected_class}")
        print(f"   Predicted: {predicted_class}")
        print(f"   Normal: {normal_prob:.3f}, Leukemia: {leukemia_prob:.3f}")
        print(f"   Confidence: {confidence:.3f}")
        print(f"   Raw logits: [{logits[0][0].item():.3f}, {logits[0][1].item():.3f}]")
        print()
    
    # Analysis
    print("📈 ANALYSIS RESULTS:")
    print("=" * 30)
    
    correct_predictions = sum(1 for r in results if r['correct'])
    total_predictions = len(results)
    accuracy = correct_predictions / total_predictions
    
    print(f"Overall Accuracy: {accuracy:.2%} ({correct_predictions}/{total_predictions})")
    
    # Check for bias towards normal classification
    normal_samples = [r for r in results if r['expected'] == 'Normal']
    cancerous_samples = [r for r in results if r['expected'] == 'Leukemia']
    
    if normal_samples:
        normal_accuracy = sum(1 for r in normal_samples if r['correct']) / len(normal_samples)
        print(f"Normal Sample Accuracy: {normal_accuracy:.2%}")
    
    if cancerous_samples:
        cancerous_accuracy = sum(1 for r in cancerous_samples if r['correct']) / len(cancerous_samples)
        print(f"Cancerous Sample Accuracy: {cancerous_accuracy:.2%}")
    
    # Check for bias
    if normal_samples and cancerous_samples:
        avg_normal_prob_for_normal = np.mean([r['normal_prob'] for r in normal_samples])
        avg_normal_prob_for_cancerous = np.mean([r['normal_prob'] for r in cancerous_samples])
        
        print(f"\n🎯 Bias Analysis:")
        print(f"Average 'Normal' probability for normal samples: {avg_normal_prob_for_normal:.3f}")
        print(f"Average 'Normal' probability for cancerous samples: {avg_normal_prob_for_cancerous:.3f}")
        
        if avg_normal_prob_for_cancerous > 0.5:
            print("⚠️  WARNING: Model shows bias towards classifying cancerous samples as normal!")
            print("   This explains why cancerous smears are being identified as normal.")
        
        if cancerous_accuracy < 0.5:
            print("⚠️  CRITICAL: Model performs poorly on cancerous samples!")
            print("   This confirms the reported issue.")
        
        if demo_mode:
            print("\n🎭 DEMO MODE DETECTED:")
            print("   The model is running in demo mode with simulated predictions.")
            print("   Demo mode has a built-in bias towards 'Normal' classification.")
            print("   This explains the misclassification issue!")
    
    return results

if __name__ == "__main__":
    test_model_validation()