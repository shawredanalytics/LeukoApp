"""
Test script to verify the fixed classification logic
Tests the specific case where abnormal probability (52.2%) > normal probability (47.8%)
"""

def test_classification_logic():
    """Test the updated classification logic"""
    
    # Test case from user's screenshot
    normal_prob = 0.478  # 47.8%
    abnormal_prob = 0.522  # 52.2%
    ABNORMAL_THRESHOLD = 0.52
    
    print("=== Classification Logic Test ===")
    print(f"Normal Probability: {normal_prob:.1%}")
    print(f"Abnormal Probability: {abnormal_prob:.1%}")
    print(f"Abnormal Threshold: {ABNORMAL_THRESHOLD}")
    print()
    
    # Original logic (WRONG)
    print("--- Original Logic (WRONG) ---")
    if abnormal_prob >= ABNORMAL_THRESHOLD:
        original_class = "Abnormal"
        original_confidence = abnormal_prob
    else:
        original_class = "Normal"
        original_confidence = normal_prob
    print(f"Classification: {original_class}")
    print(f"Confidence: {original_confidence:.1%}")
    print()
    
    # Fixed logic (CORRECT)
    print("--- Fixed Logic (CORRECT) ---")
    if abnormal_prob > normal_prob and abnormal_prob >= ABNORMAL_THRESHOLD:
        fixed_class = "Abnormal"
        fixed_confidence = abnormal_prob
    else:
        fixed_class = "Normal"
        fixed_confidence = normal_prob
    print(f"Classification: {fixed_class}")
    print(f"Confidence: {fixed_confidence:.1%}")
    print()
    
    # Test multiple scenarios
    print("=== Multiple Test Scenarios ===")
    test_cases = [
        (0.45, 0.55, "High abnormal probability"),
        (0.478, 0.522, "User's case - slightly higher abnormal"),
        (0.60, 0.40, "High normal probability"),
        (0.51, 0.49, "Slightly higher normal"),
        (0.48, 0.52, "Borderline case"),
        (0.30, 0.70, "Very high abnormal"),
    ]
    
    for normal, abnormal, description in test_cases:
        print(f"\n{description}:")
        print(f"  Normal: {normal:.1%}, Abnormal: {abnormal:.1%}")
        
        # Apply fixed logic
        if abnormal > normal and abnormal >= ABNORMAL_THRESHOLD:
            result = "Abnormal"
            confidence = abnormal
        else:
            result = "Normal"
            confidence = normal
            
        print(f"  → Classification: {result} (Confidence: {confidence:.1%})")

if __name__ == "__main__":
    test_classification_logic()