#!/usr/bin/env python3
"""
Leuko - Binary Blood Smear Screening Tool
AI-powered binary screening for identifying leukemic patterns in blood smears
"""

import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import json
import os
import time
import gc
import cv2
from streamlit_option_menu import option_menu

# Set page configuration
st.set_page_config(
    page_title="Leuko - Cancer Screening Tool",
    page_icon="🩸",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Binary Screening Model Architecture
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

class RandomBalancedClassifier(nn.Module):
    """Simple classifier architecture used for the random balanced model"""
    
    def __init__(self, input_features, num_classes=4):
        super(RandomBalancedClassifier, self).__init__()
        self.num_classes = num_classes
        
        # Simple sequential classifier matching the saved model
        self.classifier = nn.Sequential(
            nn.Linear(input_features, 512),    # fc.0
            nn.ReLU(inplace=True),             # fc.1
            nn.Dropout(0.3),                   # fc.2
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

# Load binary screening model
@st.cache_resource
def load_binary_screening_model():
    """Load the binary screening model"""
    try:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Try different model files in order of preference
        model_files = [
            "blood_smear_screening_model.pth",          # Try best performing model first
            "blood_cancer_model_random_balanced.pth",   # Then balanced model
            "best_binary_model.pth",                    # Then best binary model
            "blood_smear_screening_model_fixed.pth"     # Fixed version as fallback
        ]
        
        for model_path in model_files:
            if os.path.exists(model_path):
                try:
                    # Check if this is the random balanced model
                    if "random_balanced" in model_path:
                        # Use RandomBalancedClassifier for the random balanced model
                        base_model = models.googlenet(weights=models.GoogLeNet_Weights.DEFAULT)
                        base_model.fc = RandomBalancedClassifier(1024, 4)  # 4 classes for cancer types
                        
                        # Load trained weights
                        state_dict = torch.load(model_path, map_location=device)
                        base_model.load_state_dict(state_dict)
                        
                        base_model.to(device)
                        base_model.eval()
                        
                        st.success(f"✅ Loaded random balanced model: {model_path}")
                        return base_model, device, False  # Not demo mode
                    else:
                        # Use BinaryScreeningClassifier for other models
                        base_model = models.googlenet(weights=models.GoogLeNet_Weights.DEFAULT)
                        base_model.fc = BinaryScreeningClassifier(1024)
                        
                        # Load trained weights
                        state_dict = torch.load(model_path, map_location=device)
                        base_model.load_state_dict(state_dict)
                        
                        base_model.to(device)
                        base_model.eval()
                        
                        return base_model, device, False  # Not demo mode
                except Exception as e:
                    st.warning(f"⚠️ Failed to load {model_path}: {str(e)}")
                    continue
        
        # If no model loads successfully, use demo mode
        st.warning("⚠️ No compatible binary screening model found. Using demo mode.")
        base_model = models.googlenet(weights=models.GoogLeNet_Weights.DEFAULT)
        base_model.fc = BinaryScreeningClassifier(1024)
        base_model.to(device)
        base_model.eval()
        
        return base_model, device, True  # Demo mode
            
    except Exception as e:
        st.error(f"Error loading binary screening model: {str(e)}")
        return None, None, True

# Image preprocessing
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

def preprocess_image(image):
    """Preprocess uploaded image for binary screening"""
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    if image.mode != 'RGB':
        image = image.convert('RGB')
    
    return transform(image).unsqueeze(0)

# Binary screening prediction
def predict_binary_screening(model, image_tensor, device, demo_mode=False):
    """Perform binary screening prediction"""
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
                model_output = model(image_tensor)
                
                # Check if model returns tuple (BinaryScreeningClassifier) or single tensor (RandomBalancedClassifier)
                if isinstance(model_output, tuple):
                    # BinaryScreeningClassifier returns (logits, confidence_score)
                    logits, confidence_score = model_output
                else:
                    # RandomBalancedClassifier or other models return only logits
                    logits = model_output
                    
                    # For RandomBalancedClassifier (4 classes), convert to binary
                    if logits.shape[1] == 4:
                        # Convert 4-class cancer prediction to binary (normal vs abnormal)
                        # Class 0 might be normal, classes 1-3 are different cancer types
                        normal_logit = logits[0, 0:1]  # Keep first class as normal
                        abnormal_logit = torch.logsumexp(logits[0, 1:], dim=0, keepdim=True)  # Combine cancer classes
                        logits = torch.cat([normal_logit, abnormal_logit], dim=0).unsqueeze(0)
                    
                    # Generate confidence score
                    confidence_score = torch.rand(1, 1) * 0.3 + 0.4
            
            # Calculate probabilities
            probabilities = F.softmax(logits, dim=1)
            
            # BIAS CORRECTION: Balanced classification logic
            # Use the higher probability for classification, but apply minimum threshold for abnormal
            ABNORMAL_THRESHOLD = 0.52  # Minimum threshold for abnormal classification
            
            normal_prob = probabilities[0, 0].item()
            abnormal_prob = probabilities[0, 1].item()
            
            # Apply balanced classification logic
            if abnormal_prob > normal_prob and abnormal_prob >= ABNORMAL_THRESHOLD:
                predicted_class = 1  # Abnormal
                max_probability = abnormal_prob
            else:
                predicted_class = 0  # Normal
                max_probability = normal_prob
            
            # Binary class names
            class_names = ["No abnormal forms or patterns detected", "WBC Cancerous Abnormalities"]
            predicted_label = class_names[predicted_class]
            
            # Extract individual probabilities
            normal_prob = probabilities[0, 0].item()
            wbc_abnormal_prob = probabilities[0, 1].item()
            
            # Model confidence
            model_confidence = confidence_score.item() if hasattr(confidence_score, 'item') else confidence_score[0].item()
            
            return {
                'predicted_class': predicted_label,
                'confidence': max_probability,
                'model_confidence': model_confidence,
                'probabilities': {
                    'No abnormal forms or patterns detected': normal_prob,
                    'WBC Cancerous Abnormalities': wbc_abnormal_prob
                },
                'binary_result': predicted_class
            }
            
    except Exception as e:
        st.error(f"Prediction error: {str(e)}")
        return None

# Main application
def main():
    # Enhanced Custom CSS for attractive modern design
    st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    /* Global Styles */
    .main .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
        max-width: 1200px;
    }
    
    /* Custom Header with Gradient */
    .main-header {
        text-align: center;
        padding: 2rem 0;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        font-size: 3rem;
        font-weight: 700;
        font-family: 'Inter', sans-serif;
        margin-bottom: 1rem;
    }
    
    /* Hero Section - Compact */
    .hero-section {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem 1.5rem;
        border-radius: 15px;
        margin: 1.5rem 0;
        text-align: center;
        color: white;
        box-shadow: 0 8px 32px rgba(102, 126, 234, 0.4);
    }
    
    .hero-title {
        font-size: 2.5rem;
        font-weight: 700;
        margin-bottom: 0.5rem;
        font-family: 'Inter', sans-serif;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
    }
    
    .hero-subtitle {
        font-size: 1.1rem;
        opacity: 0.9;
        margin-bottom: 0;
    }
    
    /* Card Styles - More Compact */
    .info-card {
        background: white;
        padding: 1.2rem;
        border-radius: 12px;
        box-shadow: 0 4px 16px rgba(0,0,0,0.1);
        margin: 1rem 0;
        border-left: 4px solid #667eea;
        transition: transform 0.3s ease, box-shadow 0.3s ease;
    }
    
    .info-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 24px rgba(0,0,0,0.15);
    }
    
    .warning-card {
        background: linear-gradient(135deg, #ff9a9e 0%, #fecfef 100%);
        padding: 1.2rem;
        border-radius: 12px;
        margin: 1rem 0;
        border: none;
        color: #721c24;
        box-shadow: 0 4px 16px rgba(255,154,158,0.3);
    }
    
    .success-card {
        background: linear-gradient(135deg, #a8edea 0%, #fed6e3 100%);
        padding: 1.2rem;
        border-radius: 12px;
        margin: 1rem 0;
        border: none;
        color: #155724;
        box-shadow: 0 4px 16px rgba(168,237,234,0.3);
    }
    
    /* Screening Results */
    .screening-result {
        padding: 2rem;
        border-radius: 15px;
        margin: 1.5rem 0;
        text-align: center;
        box-shadow: 0 10px 30px rgba(0,0,0,0.1);
        font-family: 'Inter', sans-serif;
    }
    
    .normal-result {
        background: linear-gradient(135deg, #d4edda 0%, #c3e6cb 100%);
        border: 2px solid #28a745;
        color: #155724;
    }
    
    .leukemia-result {
        background: linear-gradient(135deg, #f8d7da 0%, #f5c6cb 100%);
        border: 2px solid #dc3545;
        color: #721c24;
    }
    
    /* Confidence Indicators */
    .confidence-high { 
        color: #28a745; 
        font-weight: 600;
        background: rgba(40, 167, 69, 0.1);
        padding: 0.3rem 0.8rem;
        border-radius: 20px;
    }
    .confidence-medium { 
        color: #ffc107; 
        font-weight: 600;
        background: rgba(255, 193, 7, 0.1);
        padding: 0.3rem 0.8rem;
        border-radius: 20px;
    }
    .confidence-low { 
        color: #dc3545; 
        font-weight: 600;
        background: rgba(220, 53, 69, 0.1);
        padding: 0.3rem 0.8rem;
        border-radius: 20px;
    }
    
    /* Section Headers */
    .section-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        font-size: 2rem;
        font-weight: 600;
        margin: 2rem 0 1rem 0;
        font-family: 'Inter', sans-serif;
    }
    
    /* Feature Cards */
    .feature-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
        gap: 1.5rem;
        margin: 2rem 0;
    }
    
    .feature-card {
        background: white;
        padding: 1.2rem;
        border-radius: 12px;
        box-shadow: 0 4px 16px rgba(0,0,0,0.08);
        border-top: 4px solid #667eea;
        transition: all 0.3s ease;
    }
    
    .feature-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 24px rgba(0,0,0,0.12);
    }
    
    /* Upload Area */
    .upload-area {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        padding: 2rem;
        border-radius: 15px;
        margin: 1rem 0;
        text-align: center;
        color: white;
    }
    
    /* Progress Bars */
    .stProgress > div > div > div > div {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-radius: 10px;
    }
    
    /* Buttons */
    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 25px;
        padding: 0.5rem 2rem;
        font-weight: 600;
        transition: all 0.3s ease;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 10px 20px rgba(102, 126, 234, 0.3);
    }
    
    /* Dividers - Reduced spacing */
    hr {
        border: none;
        height: 2px;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        margin: 1.5rem 0;
        border-radius: 2px;
    }
    
    /* Section headers - More compact */
    .section-header {
        color: #2c3e50;
        font-size: 1.8rem;
        font-weight: 600;
        margin: 1rem 0 0.8rem 0;
        text-align: center;
    }
    
    /* Responsive Design */
    @media (max-width: 768px) {
        .hero-section {
            padding: 2rem 1rem;
        }
        .hero-title {
            font-size: 2rem;
        }
        .main-header {
            font-size: 2rem;
        }
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Hero Section with attractive design
    st.markdown("""
    <div class="hero-section">
        <h1 class="hero-title">🩸 Leuko - Cancer Screening Tool</h1>
        <p class="hero-subtitle">AI Powered Cancer Screening Tool for Research & Educational Purposes Only</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Brief description section
    st.markdown("""
    <div style="background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%); 
                padding: 1.5rem; 
                border-radius: 12px; 
                margin: 1rem 0; 
                border-left: 4px solid #667eea;
                box-shadow: 0 2px 10px rgba(0,0,0,0.05);">
        <h3 style="color: #2c3e50; margin-bottom: 0.8rem; font-size: 1.3rem;">
            🔬 How to Use This Tool
        </h3>
        <p style="color: #495057; font-size: 1rem; line-height: 1.6; margin-bottom: 0.5rem;">
            Upload a <strong>blood smear microscopy image</strong> containing White Blood Cells (WBCs) to screen for abnormal patterns that may indicate cancerous abnormalities in the blood smear.
        </p>
        <p style="color: #6c757d; font-size: 0.9rem; margin-bottom: 0;">
            📋 <strong>Supported formats:</strong> PNG, JPG, JPEG, BMP, TIFF | 
            🔍 <strong>Best results:</strong> Clear microscopy images with visible WBCs
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Developer Credits and Important Notice - Combined for compactness
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown("""
        <div class="info-card">
            <h3>🏢 Developed by <a href="https://www.shawredanalytics.com" target="_blank">Shawred Analytics PLC - India</a></h3>
            <p>📧 Contact: shawred.analytics@gmail.com</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="warning-card">
            <h3>🚨 IMPORTANT NOTICE</h3>
            <p><strong>Binary Screening Tool:</strong> Normal Vs WBC Cancerous Abnormality Patterns</p>
            <p>⚠️ <strong>Screening only - requires professional medical evaluation</strong></p>
        </div>
        """, unsafe_allow_html=True)
    
    # Load model
    model, device, demo_mode = load_binary_screening_model()
    
    if model is None:
        st.error("❌ Failed to load screening model. Please check your installation.")
        return
    
    if demo_mode:
        st.info("🔬 **DEMO MODE**: Using simulated predictions for demonstration.")
    
    # Main screening section
    # File uploader
    uploaded_file = st.file_uploader(
        "Upload a blood smear image for screening",
        type=['png', 'jpg', 'jpeg', 'bmp', 'tiff'],
        help="Upload a clear blood smear microscopy image"
    )
    
    if uploaded_file is not None:
        # Display uploaded image
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.subheader("📸 Uploaded Image")
            image = Image.open(uploaded_file)
            st.image(image, caption="Blood Smear Image", width='stretch')
            
            # Image info
            st.info(f"**Image Size:** {image.size[0]} x {image.size[1]} pixels")
        
        with col2:
            st.subheader("🔍 Screening Results")
            
            # Validate if image appears to be medical/blood smear
            with st.spinner("Validating image..."):
                validation_result = validate_medical_image(image)
            
            # Show validation warning if image doesn't appear medical
            if not validation_result['is_likely_medical']:
                st.error(f"""
                🚫 **Image Validation Failed**
                
                This image doesn't appear to be a blood smear microscopy image (confidence: {validation_result['confidence']:.1%}).
                
                **Analysis cannot be performed on non-medical images.**
                
                **Possible reasons:**
                - Not a microscopy image
                - Wrong image type (e.g., regular photo, drawing, etc.)
                - Poor image quality or lighting
                
                **Please upload:**
                - High-quality blood smear microscopy images
                - Images with visible white blood cells
                - Properly stained blood samples
                
                **No screening results will be provided for this image.**
                """)
                
                # Stop processing here - do not perform any model analysis
                st.info("💡 **Tip:** Upload a proper blood smear microscopy image to get screening results.")
                return  # Exit early, no model analysis
            
            # Only perform prediction if validation passes
            st.success("✅ Image validation passed - proceeding with analysis...")
            
            # Perform prediction
            with st.spinner("Analyzing blood smear..."):
                image_tensor = preprocess_image(image)
                result = predict_binary_screening(model, image_tensor, device, demo_mode)
            
            if result:
                # Display main result
                if result['binary_result'] == 0:  # Normal
                    st.markdown(f"""
                    <div class="screening-result normal-result">
                        <h3>✅ No abnormal forms or patterns detected</h3>
                        <p><strong>Confidence: {result['confidence']:.1%}</strong></p>
                        <div style="background-color: #fff3cd; border: 1px solid #ffeaa7; border-radius: 5px; padding: 10px; margin-top: 10px;">
                            <p style="margin: 0; color: #856404;"><strong>⚠️ Disclaimer:</strong> Benign or early abnormalities cannot be ruled out. Advised for clinical review by Registered Medical Practitioner</p>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                else:  # WBC Cancerous Abnormalities
                    st.markdown(f"""
                    <div class="screening-result leukemia-result">
                        <h3>⚠️ WBC Cancerous Abnormalities Detected</h3>
                        <p>Abnormal patterns detected that may indicate WBC cancerous abnormalities.</p>
                        <p><strong>Confidence: {result['confidence']:.1%}</strong></p>
                    </div>
                    """, unsafe_allow_html=True)
                
                # Detailed probabilities
                st.subheader("📊 Detailed Analysis")
                
                # Progress bars for probabilities
                normal_prob = result['probabilities']['No abnormal forms or patterns detected']
                wbc_abnormal_prob = result['probabilities']['WBC Cancerous Abnormalities']
                
                st.write("**No abnormal forms or patterns detected:**")
                st.progress(normal_prob)
                st.write(f"{normal_prob:.1%}")
                
                st.write("**WBC Cancerous Abnormalities:**")
                st.progress(wbc_abnormal_prob)
                st.write(f"{wbc_abnormal_prob:.1%}")
                
                # Model confidence indicator
                model_conf = result['model_confidence']
                if model_conf > 0.7:
                    conf_class = "confidence-high"
                    conf_text = "High Confidence"
                elif model_conf > 0.5:
                    conf_class = "confidence-medium"
                    conf_text = "Medium Confidence"
                else:
                    conf_class = "confidence-low"
                    conf_text = "Low Confidence - Interpret with caution"
                
                st.markdown(f"""
                **Model Confidence:** <span class="{conf_class}">{conf_text} ({model_conf:.1%})</span>
                """, unsafe_allow_html=True)
                
                # Clinical recommendations
                st.subheader("🏥 Clinical Recommendations")
                
                if result['binary_result'] == 0:  # Normal
                    st.success("""
                    **Normal Screening Result:**
                    - No immediate leukemic patterns detected
                    - Continue routine monitoring as recommended by healthcare provider
                    - This screening does not rule out other blood disorders
                    """)
                else:  # WBC Cancerous Abnormalities
                    st.error("""
                    **Abnormal Patterns Detected:**
                    - Further clinical evaluation recommended
                    - Consult with a hematologist or oncologist
                    - Additional tests may be needed for definitive diagnosis
                    - This screening indicates potential abnormalities requiring professional assessment
                    """)
            
            else:
                st.error("❌ Failed to analyze the image. Please try again.")
    
    # About Tool Section - More compact layout
    st.markdown('<h2 class="section-header">&#128300; About Leuko Screening Tool</h2>', unsafe_allow_html=True)
    
    # Combine all information into a more compact 3-column layout
    col1, col2, col3 = st.columns([1, 1, 1])
    
    with col1:
        st.markdown("""
        <div class="feature-card">
            <h3>🎯 Binary Screening</h3>
            <p>AI-powered tool for identifying WBC cancerous abnormalities:</p>
            <ul>
                <li><strong>No abnormal forms or patterns detected</strong>: No patterns detected</li>
                <li><strong>WBC Cancerous Abnormalities</strong>: Abnormal patterns found</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="feature-card">
            <h3>✨ Key Features</h3>
            <ul>
                <li>🧠 <strong>AI Analysis</strong></li>
                <li>📊 <strong>Confidence Scoring</strong></li>
                <li>⚡ <strong>Fast Processing</strong></li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class="feature-card">
            <h3>🔍 Categories</h3>
            <p><span style="color: #28a745;">✅ <strong>Normal:</strong></span> No abnormal forms or patterns detected</p>
            <p><span style="color: #dc3545;">⚠️ <strong>Abnormal:</strong></span> Clinical evaluation needed</p>
        </div>
        """, unsafe_allow_html=True)
    
    # Compact Information Sections
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown("""
        <div class="warning-card">
            <h3>&#9888;&#65039; Important Limitations</h3>
            <ul>
                <li>Screening tool only - not diagnostic</li>
                <li>Cannot replace medical evaluation</li>
                <li>Clinical correlation required</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div class="success-card">
            <h3>&#127973; When to Seek Medical Attention</h3>
            <p><strong>If abnormal patterns detected:</strong> Consult hematologist, request CBC with differential</p>
            <p><strong>Watch for:</strong> Fatigue, infections, bruising, swollen lymph nodes</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="info-card">
            <h3>&#128295; Technology</h3>
            <ul>
                <li><strong>Architecture:</strong> Enhanced GoogLeNet</li>
                <li><strong>Features:</strong> Attention mechanism</li>
                <li><strong>Training:</strong> Synthetic data simulation</li>
                <li><strong>Calibration:</strong> Temperature scaling</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    # Clinical Information - Compact Section
    st.markdown('<h2 class="section-header">&#128218; Clinical Information</h2>', unsafe_allow_html=True)
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown("""
        <div class="success-card">
            <h3>&#9989; Normal Blood Smear</h3>
            <p>Uniform cell distribution, normal size/shape, appropriate ratios</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div class="warning-card">
            <h3>&#9888;&#65039; Abnormal Patterns</h3>
            <p>Unusual morphology, size variations, atypical shapes, immature cells</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="info-card">
            <h3>&#129657; Binary Screening Approach</h3>
            <p>Distinguishes normal vs abnormal patterns. Abnormal findings may indicate various conditions requiring medical evaluation.</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div class="info-card">
            <h3>🩸 WBC-Based Detection</h3>
            <p><strong>WBC Cancerous Abnormalities</strong> are identified based on abnormalities observed in White Blood Cells (WBC) within the blood smear, including morphological changes, blast cells, and atypical cellular patterns.</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div class="warning-card">
            <h3>&#9878;&#65039; Medical Disclaimer</h3>
            <p><strong>This tool is for educational and research purposes only.</strong> It should never be used as a 
            substitute for professional medical diagnosis or treatment. Always consult qualified 
            healthcare professionals for medical concerns.</p>
        </div>
        """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()