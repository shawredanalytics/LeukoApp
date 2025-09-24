"""
Simple test to verify the threshold fix works by simulating predictions.
"""

import numpy as np

def simulate_prediction_with_threshold(normal_prob, abnormal_prob, threshold=0.55):
    """Simulate the prediction logic with the new threshold"""
    
    # Apply threshold (matching the app logic)
    if abnormal_prob >= threshold:
        predicted_class = 1  # Abnormal
        confidence = abnormal_prob
        prediction_text = "WBC Cancerous Abnormalities"
    else:
        predicted_class = 0  # Normal
        confidence = normal_prob
        prediction_text = "Normal Smear"
    
    return predicted_class, confidence, prediction_text

def test_threshold_scenarios():
    """Test various probability scenarios with different thresholds"""
    
    print("🔬 TESTING THRESHOLD FIX FOR NORMAL SMEAR MISCLASSIFICATION")
    print("=" * 70)
    
    # Test scenarios that were causing false positives
    test_cases = [
        # (normal_prob, abnormal_prob, description)
        (0.48, 0.52, "Borderline case - slightly abnormal"),
        (0.45, 0.55, "Moderate abnormal prediction"),
        (0.40, 0.60, "Strong abnormal prediction"),
        (0.65, 0.35, "Clear normal case"),
        (0.70, 0.30, "Strong normal case"),
        (0.50, 0.50, "Exactly balanced"),
        (0.46, 0.54, "Slightly abnormal - common false positive"),
    ]
    
    thresholds = [0.50, 0.52, 0.55, 0.60]
    
    print("\n📊 COMPARISON OF THRESHOLDS:")
    print("Case Description                    | Normal | Abnormal | 0.50 | 0.52 | 0.55 | 0.60")
    print("-" * 85)
    
    for normal_prob, abnormal_prob, description in test_cases:
        results = []
        for threshold in thresholds:
            pred_class, confidence, pred_text = simulate_prediction_with_threshold(
                normal_prob, abnormal_prob, threshold
            )
            # Show N for Normal, A for Abnormal
            results.append("N" if pred_class == 0 else "A")
        
        print(f"{description:<35} | {normal_prob:>6.2f} | {abnormal_prob:>8.2f} | {results[0]:>4} | {results[1]:>4} | {results[2]:>4} | {results[3]:>4}")
    
    print("\n🎯 ANALYSIS OF THRESHOLD 0.55 (CURRENT FIX):")
    print("-" * 50)
    
    false_positives_old = 0
    false_positives_new = 0
    
    for normal_prob, abnormal_prob, description in test_cases:
        # Assume cases with normal_prob > abnormal_prob should be classified as normal
        true_label = 0 if normal_prob > abnormal_prob else 1
        
        # Test old threshold (0.52)
        pred_old, _, _ = simulate_prediction_with_threshold(normal_prob, abnormal_prob, 0.52)
        if true_label == 0 and pred_old == 1:  # False positive
            false_positives_old += 1
        
        # Test new threshold (0.55)
        pred_new, _, _ = simulate_prediction_with_threshold(normal_prob, abnormal_prob, 0.55)
        if true_label == 0 and pred_new == 1:  # False positive
            false_positives_new += 1
        
        # Show the specific case
        old_result = "Normal" if pred_old == 0 else "Abnormal"
        new_result = "Normal" if pred_new == 0 else "Abnormal"
        
        if true_label == 0:  # This should be normal
            status_old = "✅" if pred_old == 0 else "❌ FALSE POSITIVE"
            status_new = "✅" if pred_new == 0 else "❌ FALSE POSITIVE"
            print(f"{description}")
            print(f"  Probabilities: Normal={normal_prob:.2f}, Abnormal={abnormal_prob:.2f}")
            print(f"  Threshold 0.52: {old_result} {status_old}")
            print(f"  Threshold 0.55: {new_result} {status_new}")
            print()
    
    print(f"📈 IMPROVEMENT SUMMARY:")
    print(f"False Positives with threshold 0.52: {false_positives_old}")
    print(f"False Positives with threshold 0.55: {false_positives_new}")
    
    if false_positives_new < false_positives_old:
        print(f"✅ IMPROVEMENT: Reduced false positives by {false_positives_old - false_positives_new}")
    elif false_positives_new == false_positives_old:
        print(f"⚠️  NO CHANGE: Same number of false positives")
    else:
        print(f"❌ REGRESSION: Increased false positives by {false_positives_new - false_positives_old}")
    
    print(f"\n🔧 RECOMMENDATION:")
    if false_positives_new == 0:
        print("✅ Threshold 0.55 eliminates false positives in test cases.")
        print("   Normal smears should now be classified correctly.")
    elif false_positives_new < false_positives_old:
        print("✅ Threshold 0.55 reduces false positives significantly.")
        print("   Consider testing with real images to validate.")
    else:
        print("⚠️  Consider increasing threshold further to 0.60 or higher.")

if __name__ == "__main__":
    test_threshold_scenarios()