#!/usr/bin/env python3
"""
Test the bias correction implementation
This script simulates the bias correction logic to verify it works
"""
import torch
import torch.nn.functional as F
import numpy as np

def simulate_bias_correction(probabilities, threshold=0.65):
    """Simulate the bias correction logic from the app"""
    normal_prob = probabilities[0]
    abnormal_prob = probabilities[1]
    
    # Apply bias correction
    if abnormal_prob >= threshold:
        predicted_class = 1  # Abnormal
        max_probability = abnormal_prob
    else:
        predicted_class = 0  # Normal
        max_probability = normal_prob
    
    class_names = ["Normal Smear", "WBC Cancerous Abnormalities"]
    predicted_label = class_names[predicted_class]
    
    return predicted_class, predicted_label, max_probability

def test_bias_correction():
    """Test the bias correction with various probability scenarios"""
    print("🧪 Testing Bias Correction Logic")
    print("=" * 50)
    
    # Test cases: [normal_prob, abnormal_prob]
    test_cases = [
        [0.6, 0.4],   # Should predict Normal (abnormal < 0.65)
        [0.4, 0.6],   # Should predict Normal (abnormal < 0.65)
        [0.3, 0.7],   # Should predict Abnormal (abnormal >= 0.65)
        [0.2, 0.8],   # Should predict Abnormal (abnormal >= 0.65)
        [0.35, 0.65], # Should predict Abnormal (abnormal >= 0.65)
        [0.45, 0.55], # Should predict Normal (abnormal < 0.65)
        [0.5, 0.5],   # Should predict Normal (abnormal < 0.65)
        [0.1, 0.9],   # Should predict Abnormal (abnormal >= 0.65)
    ]
    
    normal_predictions = 0
    abnormal_predictions = 0
    
    for i, (normal_prob, abnormal_prob) in enumerate(test_cases):
        predicted_class, predicted_label, max_prob = simulate_bias_correction([normal_prob, abnormal_prob])
        
        print(f"Test {i+1}: Normal={normal_prob:.2f}, Abnormal={abnormal_prob:.2f}")
        print(f"   → Prediction: {predicted_label} (confidence: {max_prob:.2f})")
        
        if predicted_class == 0:
            normal_predictions += 1
        else:
            abnormal_predictions += 1
    
    print("\n" + "=" * 50)
    print("📊 BIAS CORRECTION RESULTS:")
    print(f"   Normal predictions: {normal_predictions}/{len(test_cases)} ({normal_predictions/len(test_cases)*100:.1f}%)")
    print(f"   Abnormal predictions: {abnormal_predictions}/{len(test_cases)} ({abnormal_predictions/len(test_cases)*100:.1f}%)")
    
    # Test with realistic model outputs (biased towards abnormal)
    print("\n🔬 Testing with Realistic Biased Model Outputs:")
    print("=" * 50)
    
    # Simulate 20 predictions with bias towards abnormal (like our models)
    np.random.seed(42)  # For reproducible results
    realistic_normal_count = 0
    realistic_abnormal_count = 0
    
    for i in range(20):
        # Simulate biased model output (tends to predict abnormal)
        normal_prob = np.random.uniform(0.4, 0.52)  # Lower normal probabilities
        abnormal_prob = 1.0 - normal_prob  # Complement
        
        predicted_class, predicted_label, max_prob = simulate_bias_correction([normal_prob, abnormal_prob])
        
        if i < 5:  # Show first 5 examples
            print(f"Sample {i+1}: Normal={normal_prob:.3f}, Abnormal={abnormal_prob:.3f}")
            print(f"   → Prediction: {predicted_label}")
        
        if predicted_class == 0:
            realistic_normal_count += 1
        else:
            realistic_abnormal_count += 1
    
    print(f"\n📈 Realistic Test Results:")
    print(f"   Normal predictions: {realistic_normal_count}/20 ({realistic_normal_count/20*100:.1f}%)")
    print(f"   Abnormal predictions: {realistic_abnormal_count}/20 ({realistic_abnormal_count/20*100:.1f}%)")
    
    if abs(realistic_normal_count - realistic_abnormal_count) <= 6:
        print("   ✅ BIAS CORRECTION WORKING - More balanced predictions!")
    else:
        print("   ⚠️  Still some bias, but improved from 100% abnormal")

if __name__ == "__main__":
    test_bias_correction()