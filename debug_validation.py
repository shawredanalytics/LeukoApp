#!/usr/bin/env python3
"""
Debug script to test image validation function
"""

import cv2
import numpy as np
from PIL import Image
import requests
from io import BytesIO

def validate_medical_image(image):
    """Validate if the uploaded image appears to be a medical/microscopy image"""
    
    # Convert PIL to numpy array
    img_array = np.array(image)
    
    # Convert to grayscale for analysis
    if len(img_array.shape) == 3:
        gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
    else:
        gray = img_array
    
    # Check image characteristics typical of blood smears
    # 1. Check for high contrast (blood cells should have distinct boundaries)
    contrast = gray.std()
    
    # 2. Check for circular/cellular structures using edge detection
    edges = cv2.Canny(gray, 50, 150)
    edge_density = np.sum(edges > 0) / edges.size
    
    # 3. Check color distribution (blood smears typically have specific color ranges)
    medical_color_score = 0.0
    if len(img_array.shape) == 3:
        # Convert to HSV for better color analysis
        hsv = cv2.cvtColor(img_array, cv2.COLOR_RGB2HSV)
        
        # Check for medical staining colors more precisely
        # Wright-Giemsa staining produces specific purple/pink hues (280-320° in HSV)
        # and blue hues for nuclei (200-260° in HSV)
        
        # Purple/magenta range (typical for cytoplasm in stained blood smears)
        purple_mask = (hsv[:,:,0] >= 140) & (hsv[:,:,0] <= 170) & (hsv[:,:,1] >= 50)
        purple_ratio = np.sum(purple_mask) / purple_mask.size
        
        # Blue range (typical for nuclei in stained blood smears)  
        blue_mask = (hsv[:,:,0] >= 100) & (hsv[:,:,0] <= 130) & (hsv[:,:,1] >= 50)
        blue_ratio = np.sum(blue_mask) / blue_mask.size
        
        # Pink range (typical for red blood cells)
        pink_mask = (hsv[:,:,0] >= 170) & (hsv[:,:,0] <= 180) & (hsv[:,:,1] >= 30)
        pink_ratio = np.sum(pink_mask) / pink_mask.size
        
        # Check for overall color saturation (medical images are usually less saturated)
        avg_saturation = np.mean(hsv[:,:,1]) / 255.0
        
        # Medical images typically have moderate saturation (not too bright/vivid)
        saturation_penalty = 0 if avg_saturation < 0.6 else (avg_saturation - 0.6) * 2
        
        medical_color_score = (purple_ratio + blue_ratio + pink_ratio) - saturation_penalty
        medical_color_score = max(0, medical_color_score)  # Don't go negative
    
    # 4. Check image size and resolution (medical images are usually high-res)
    width, height = image.size
    resolution_score = min(1.0, (width * height) / (500 * 500))  # Normalize to 500x500 baseline
    
    # 5. Check for uniform background (medical images often have consistent backgrounds)
    background_uniformity = 1.0 - (contrast / 100.0)  # Higher contrast = less uniform background
    background_uniformity = max(0, min(1.0, background_uniformity))
    
    # Scoring system with improved weights
    scores = {
        'contrast': min(1.0, contrast / 80.0),  # Medical images have moderate contrast
        'edge_density': min(1.0, edge_density * 15),  # Cellular structures
        'medical_colors': medical_color_score * 3,  # Boost medical color importance
        'resolution': resolution_score,
        'background_uniformity': background_uniformity
    }
    
    # Calculate overall medical image probability with stricter criteria
    weights = {
        'contrast': 0.2, 
        'edge_density': 0.25, 
        'medical_colors': 0.4,  # Most important factor
        'resolution': 0.05,
        'background_uniformity': 0.1
    }
    medical_score = sum(scores[key] * weights[key] for key in weights)
    
    # More stringent threshold - require higher confidence for medical classification
    is_medical = medical_score > 0.4 and medical_color_score > 0.1
    
    return {
        'is_likely_medical': is_medical,
        'confidence': medical_score,
        'scores': scores,
        'warnings': []
    }

def test_validation():
    """Test the validation function with different image types"""
    
    # Test with a simple colored image (simulating the bird)
    # Create a test image with bright colors like the kingfisher
    test_image = Image.new('RGB', (400, 300), color=(0, 150, 200))  # Blue background
    
    print("=== Testing Bird-like Image ===")
    result = validate_medical_image(test_image)
    print(f"Is likely medical: {result['is_likely_medical']}")
    print(f"Confidence: {result['confidence']:.3f}")
    print(f"Scores: {result['scores']}")
    print()
    
    # Test with a more complex colorful image
    # Create an image with multiple bright colors
    complex_image = Image.new('RGB', (400, 300))
    pixels = []
    for y in range(300):
        for x in range(400):
            # Create a gradient with bright colors
            r = int(255 * (x / 400))
            g = int(255 * (y / 300))
            b = 150
            pixels.append((r, g, b))
    complex_image.putdata(pixels)
    
    print("=== Testing Complex Colorful Image ===")
    result = validate_medical_image(complex_image)
    print(f"Is likely medical: {result['is_likely_medical']}")
    print(f"Confidence: {result['confidence']:.3f}")
    print(f"Scores: {result['scores']}")
    print()
    
    # Test with a medical-like image (purple/pink colors)
    medical_image = Image.new('RGB', (400, 300), color=(150, 100, 150))  # Purple-ish
    
    print("=== Testing Medical-like Image ===")
    result = validate_medical_image(medical_image)
    print(f"Is likely medical: {result['is_likely_medical']}")
    print(f"Confidence: {result['confidence']:.3f}")
    print(f"Scores: {result['scores']}")

if __name__ == "__main__":
    test_validation()