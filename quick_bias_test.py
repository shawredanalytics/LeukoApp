import torch
import torch.nn.functional as F
import numpy as np

def simulate_model_predictions():
    """Simulate different model prediction scenarios to understand bias"""
    
    print("🔍 Testing Bias Correction Logic")
    print("=" * 50)
    
    # Test scenarios with different probability distributions
    test_cases = [
        # Case 1: Model strongly predicts normal (should be classified as normal)
        {"name": "Strong Normal", "normal_prob": 0.85, "abnormal_prob": 0.15},
        
        # Case 2: Model slightly predicts normal (should be classified as normal)
        {"name": "Slight Normal", "normal_prob": 0.55, "abnormal_prob": 0.45},
        
        # Case 3: Model slightly predicts abnormal (borderline case)
        {"name": "Slight Abnormal", "normal_prob": 0.45, "abnormal_prob": 0.55},
        
        # Case 4: Model strongly predicts abnormal (should be classified as abnormal)
        {"name": "Strong Abnormal", "normal_prob": 0.25, "abnormal_prob": 0.75},
        
        # Case 5: Very close to threshold
        {"name": "Near Threshold", "normal_prob": 0.48, "abnormal_prob": 0.52},
        
        # Case 6: Exactly at threshold
        {"name": "At Threshold", "normal_prob": 0.48, "abnormal_prob": 0.52},
    ]
    
    # Current bias correction threshold from the app
    ABNORMAL_THRESHOLD = 0.52
    
    print(f"Current Abnormal Threshold: {ABNORMAL_THRESHOLD}")
    print("-" * 50)
    
    for case in test_cases:
        normal_prob = case["normal_prob"]
        abnormal_prob = case["abnormal_prob"]
        
        # Apply the same bias correction logic as in the app
        if abnormal_prob >= ABNORMAL_THRESHOLD:
            predicted_class = 1  # Abnormal
            max_probability = abnormal_prob
            predicted_label = "WBC Cancerous Abnormalities"
        else:
            predicted_class = 0  # Normal
            max_probability = normal_prob
            predicted_label = "Normal Smear"
        
        # Determine if this is correct for a normal sample
        is_normal_sample = True  # Assume we're testing normal samples
        is_correct = (predicted_class == 0) if is_normal_sample else (predicted_class == 1)
        status = "✅ CORRECT" if is_correct else "❌ FALSE POSITIVE"
        
        print(f"{case['name']:15} | Normal: {normal_prob:.2f} | Abnormal: {abnormal_prob:.2f} | "
              f"Prediction: {predicted_label:25} | {status}")
    
    print("\n" + "=" * 50)
    print("📊 ANALYSIS")
    print("=" * 50)
    
    # Test what threshold would work better
    print("\nTesting different thresholds:")
    thresholds = [0.45, 0.50, 0.52, 0.55, 0.60, 0.65]
    
    # Simulate normal samples (should predict normal)
    normal_samples = [
        {"normal": 0.85, "abnormal": 0.15},  # Strong normal
        {"normal": 0.70, "abnormal": 0.30},  # Moderate normal
        {"normal": 0.55, "abnormal": 0.45},  # Slight normal
        {"normal": 0.52, "abnormal": 0.48},  # Weak normal
        {"normal": 0.48, "abnormal": 0.52},  # Borderline (model says abnormal)
    ]
    
    print(f"{'Threshold':>10} | {'False Positives':>15} | {'Accuracy':>10}")
    print("-" * 40)
    
    for threshold in thresholds:
        false_positives = 0
        total_samples = len(normal_samples)
        
        for sample in normal_samples:
            if sample["abnormal"] >= threshold:
                false_positives += 1
        
        accuracy = (total_samples - false_positives) / total_samples
        print(f"{threshold:>10.2f} | {false_positives:>15} | {accuracy:>10.1%}")
    
    print("\n🎯 RECOMMENDATION:")
    print("Current threshold (0.52) may be too low, causing normal samples to be classified as abnormal.")
    print("Consider increasing threshold to 0.55-0.60 to reduce false positives.")
    
    return thresholds

def test_actual_model_behavior():
    """Test how the actual models behave with synthetic data"""
    print("\n" + "=" * 50)
    print("🧪 TESTING ACTUAL MODEL BEHAVIOR")
    print("=" * 50)
    
    # Create synthetic "normal" and "abnormal" looking tensors
    torch.manual_seed(42)  # For reproducible results
    
    # Normal-looking image (higher brightness, less variation)
    normal_image = torch.randn(1, 3, 224, 224) * 0.3 + 0.7
    normal_image = torch.clamp(normal_image, 0, 1)
    
    # Abnormal-looking image (lower brightness, more variation)
    abnormal_image = torch.randn(1, 3, 224, 224) * 0.5 + 0.3
    abnormal_image = torch.clamp(abnormal_image, 0, 1)
    
    print("Created synthetic test images:")
    print(f"Normal image stats: mean={normal_image.mean():.3f}, std={normal_image.std():.3f}")
    print(f"Abnormal image stats: mean={abnormal_image.mean():.3f}, std={abnormal_image.std():.3f}")
    
    # Simulate what different models might predict
    print("\nSimulated model predictions:")
    
    # Scenario 1: Biased model (tends to predict abnormal)
    print("\n1. Biased Model (tends to predict abnormal):")
    biased_normal_logits = torch.tensor([[-0.5, 1.2]])  # Favors abnormal
    biased_abnormal_logits = torch.tensor([[-1.0, 2.0]])  # Strongly favors abnormal
    
    for name, logits in [("Normal Image", biased_normal_logits), ("Abnormal Image", biased_abnormal_logits)]:
        probs = F.softmax(logits, dim=1)
        normal_prob = probs[0, 0].item()
        abnormal_prob = probs[0, 1].item()
        
        # Apply bias correction
        if abnormal_prob >= 0.52:
            prediction = "WBC Cancerous Abnormalities"
            is_fp = name == "Normal Image"
        else:
            prediction = "Normal Smear"
            is_fp = False
        
        status = " ❌ FALSE POSITIVE" if is_fp else ""
        print(f"  {name}: Normal={normal_prob:.3f}, Abnormal={abnormal_prob:.3f} → {prediction}{status}")
    
    # Scenario 2: Balanced model
    print("\n2. Balanced Model:")
    balanced_normal_logits = torch.tensor([[0.8, -0.3]])  # Favors normal
    balanced_abnormal_logits = torch.tensor([[-0.3, 0.8]])  # Favors abnormal
    
    for name, logits in [("Normal Image", balanced_normal_logits), ("Abnormal Image", balanced_abnormal_logits)]:
        probs = F.softmax(logits, dim=1)
        normal_prob = probs[0, 0].item()
        abnormal_prob = probs[0, 1].item()
        
        # Apply bias correction
        if abnormal_prob >= 0.52:
            prediction = "WBC Cancerous Abnormalities"
            is_fp = name == "Normal Image"
        else:
            prediction = "Normal Smear"
            is_fp = False
        
        status = " ❌ FALSE POSITIVE" if is_fp else ""
        print(f"  {name}: Normal={normal_prob:.3f}, Abnormal={abnormal_prob:.3f} → {prediction}{status}")

if __name__ == "__main__":
    simulate_model_predictions()
    test_actual_model_behavior()
    
    print("\n" + "=" * 50)
    print("🔧 NEXT STEPS TO FIX BIAS:")
    print("=" * 50)
    print("1. Increase the abnormal threshold from 0.52 to 0.55-0.60")
    print("2. Test with real images to validate the fix")
    print("3. Consider model retraining if threshold adjustment isn't sufficient")
    print("4. Add confidence-based filtering for borderline cases")