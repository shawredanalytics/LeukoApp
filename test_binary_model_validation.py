import torch
import torch.nn as nn
import torch.nn.functional as F
import json
import numpy as np
from PIL import Image
import torchvision.transforms as transforms
import os

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super(MultiHeadAttention, self).__init__()
        self.num_heads = num_heads
        self.d_model = d_model
        self.depth = d_model // num_heads
        
        self.wq = nn.Linear(d_model, d_model)
        self.wk = nn.Linear(d_model, d_model)
        self.wv = nn.Linear(d_model, d_model)
        self.dense = nn.Linear(d_model, d_model)
        
    def forward(self, x):
        batch_size = x.size(0)
        
        q = self.wq(x).view(batch_size, -1, self.num_heads, self.depth).transpose(1, 2)
        k = self.wk(x).view(batch_size, -1, self.num_heads, self.depth).transpose(1, 2)
        v = self.wv(x).view(batch_size, -1, self.num_heads, self.depth).transpose(1, 2)
        
        attention_weights = torch.matmul(q, k.transpose(-2, -1)) / np.sqrt(self.depth)
        attention_weights = F.softmax(attention_weights, dim=-1)
        
        attention_output = torch.matmul(attention_weights, v)
        attention_output = attention_output.transpose(1, 2).contiguous().view(
            batch_size, -1, self.d_model)
        
        return self.dense(attention_output)

class BinaryScreeningClassifier(nn.Module):
    def __init__(self, num_classes=2):
        super(BinaryScreeningClassifier, self).__init__()
        
        # Enhanced feature extraction
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            
            # Block 1
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            # Block 2
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            # Block 3
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        
        # Global average pooling
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        
        # Multi-head attention
        self.attention = MultiHeadAttention(512, 8)
        
        # Specialized analysis branches
        self.normal_branch = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
        )
        
        self.leukemia_branch = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
        )
        
        # Final classification
        self.classifier = nn.Linear(256, num_classes)
        
        # Confidence estimation
        self.confidence_head = nn.Sequential(
            nn.Linear(256, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        # Feature extraction
        features = self.features(x)
        features = self.global_pool(features)
        features = features.view(features.size(0), -1)
        
        # Apply attention
        features_reshaped = features.unsqueeze(1)
        attended_features = self.attention(features_reshaped)
        attended_features = attended_features.squeeze(1)
        
        # Specialized branches
        normal_features = self.normal_branch(attended_features)
        leukemia_features = self.leukemia_branch(attended_features)
        
        # Combine features
        combined_features = torch.cat([normal_features, leukemia_features], dim=1)
        
        # Classification
        logits = self.classifier(combined_features)
        
        # Confidence estimation
        confidence = self.confidence_head(combined_features)
        
        return logits, confidence

def load_model():
    """Load the binary screening model"""
    try:
        model = BinaryScreeningClassifier(num_classes=2)
        model.load_state_dict(torch.load('blood_smear_screening_model.pth', map_location='cpu'))
        model.eval()
        
        # Load metadata
        with open('blood_smear_screening_metadata.json', 'r') as f:
            metadata = json.load(f)
        
        return model, metadata
    except Exception as e:
        print(f"Error loading model: {e}")
        return None, None

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

def test_model_predictions():
    """Test the model with synthetic samples"""
    print("🔍 Testing Binary Screening Model Validation")
    print("=" * 50)
    
    # Load model
    model, metadata = load_model()
    if model is None:
        print("❌ Failed to load model!")
        return
    
    print(f"✅ Model loaded successfully")
    print(f"📊 Model metadata: {metadata}")
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
    
    with torch.no_grad():
        for sample_name, sample_type in test_cases:
            # Create test image
            test_image = create_test_image(sample_type)
            
            # Transform image
            input_tensor = transform(test_image).unsqueeze(0)
            
            # Get prediction
            logits, confidence = model(input_tensor)
            probabilities = F.softmax(logits, dim=1)
            
            normal_prob = probabilities[0][0].item()
            leukemia_prob = probabilities[0][1].item()
            confidence_score = confidence[0].item()
            
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
                'confidence': confidence_score
            })
            
            status = "✅" if is_correct else "❌"
            print(f"{status} {sample_name}:")
            print(f"   Expected: {expected_class}")
            print(f"   Predicted: {predicted_class}")
            print(f"   Normal: {normal_prob:.3f}, Leukemia: {leukemia_prob:.3f}")
            print(f"   Confidence: {confidence_score:.3f}")
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
    
    normal_accuracy = sum(1 for r in normal_samples if r['correct']) / len(normal_samples)
    cancerous_accuracy = sum(1 for r in cancerous_samples if r['correct']) / len(cancerous_samples)
    
    print(f"Normal Sample Accuracy: {normal_accuracy:.2%}")
    print(f"Cancerous Sample Accuracy: {cancerous_accuracy:.2%}")
    
    # Check for bias
    avg_normal_prob_for_normal = np.mean([r['normal_prob'] for r in normal_samples])
    avg_normal_prob_for_cancerous = np.mean([r['normal_prob'] for r in cancerous_samples])
    
    print(f"\nBias Analysis:")
    print(f"Average 'Normal' probability for normal samples: {avg_normal_prob_for_normal:.3f}")
    print(f"Average 'Normal' probability for cancerous samples: {avg_normal_prob_for_cancerous:.3f}")
    
    if avg_normal_prob_for_cancerous > 0.5:
        print("⚠️  WARNING: Model shows bias towards classifying cancerous samples as normal!")
        print("   This could explain why cancerous smears are being identified as normal.")
    
    if cancerous_accuracy < 0.5:
        print("⚠️  CRITICAL: Model performs poorly on cancerous samples!")
        print("   This confirms the reported issue.")
    
    return results

if __name__ == "__main__":
    test_model_predictions()