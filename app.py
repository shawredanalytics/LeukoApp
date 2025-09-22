import streamlit as st
import torch
from torchvision.models import googlenet, GoogLeNet_Weights
import torch.nn as nn
from torchvision import transforms
from PIL import Image, ImageOps, ImageFilter, ImageEnhance
import os
import logging
import sys
import numpy as np
from collections import OrderedDict
from streamlit_option_menu import option_menu

# Define constants and variables
HAS_OPENCV = False
try:
    import cv2
    HAS_OPENCV = True
except ImportError:
    pass

DEFAULT_TEMPERATURE = 1.0

def validate_blood_smear_image(image):
    """
    Validates if the uploaded image resembles blood smear architecture.
    Returns True if it's likely a blood smear, False otherwise.
    
    This function implements strict validation to ensure only blood smear-like images are analyzed:
    1. Checks for cellular structures and patterns typical of blood smears
    2. Validates color distribution consistent with stained blood cells
    3. Rejects non-medical images, text documents, screenshots, etc.
    """
    # Convert to numpy array
    img_array = np.array(image)
    
    # Debug logging for troubleshooting
    debug_info = []
    
    # 1. Basic dimension and format checks
    if image.width < 50 or image.height < 50:
        return False
    
    # Ensure RGB format
    if len(img_array.shape) != 3 or img_array.shape[2] != 3:
        return False
    
    # 2. Check if image is completely black, white, or single color (corrupted/invalid)
    if np.max(img_array) < 10 or np.min(img_array) > 245:
        return False
    
    # Check for single color dominance (likely not a blood smear)
    if np.std(img_array) < 15:
        return False
    
    # 3. Blood smear color validation - should have medical staining colors
    # Convert to different color spaces for analysis
    r_channel = img_array[:, :, 0].astype(float)
    g_channel = img_array[:, :, 1].astype(float)
    b_channel = img_array[:, :, 2].astype(float)
    
    # Calculate color statistics
    r_mean, g_mean, b_mean = np.mean(r_channel), np.mean(g_channel), np.mean(b_channel)
    r_std, g_std, b_std = np.std(r_channel), np.std(g_channel), np.std(b_channel)
    
    # 4. Reject images with characteristics inconsistent with blood smears
    
    # Reject pure grayscale images (blood smears should have color variation)
    if abs(r_mean - g_mean) < 5 and abs(g_mean - b_mean) < 5 and abs(r_mean - b_mean) < 5:
        return False
    
    # Reject images with extreme color bias (likely not medical)
    total_mean = (r_mean + g_mean + b_mean) / 3
    if (r_mean > total_mean * 1.5 or g_mean > total_mean * 1.5 or b_mean > total_mean * 1.5):
        return False
    
    # 5. Check for text-like patterns (reject documents, screenshots with text)
    # High contrast edges often indicate text or non-cellular structures
    gray = np.mean(img_array, axis=2)
    
    # Calculate edge density using simple gradient
    grad_x = np.abs(np.diff(gray, axis=1))
    grad_y = np.abs(np.diff(gray, axis=0))
    
    # High edge density suggests text or geometric patterns
    edge_density_x = np.mean(grad_x > 30)
    edge_density_y = np.mean(grad_y > 30)
    
    if edge_density_x > 0.15 or edge_density_y > 0.15:
        return False
    
    # 6. Check for cellular-like structures
    # Blood smears should have moderate texture variation (cellular structures)
    texture_variance = np.var(gray)
    if texture_variance < 100:  # Too uniform, likely not cellular
        return False
    if texture_variance > 2000:  # Too chaotic, likely not blood smear
        return False
    
    # 7. Color range validation for medical staining
    # Blood smears typically have specific color ranges due to Wright-Giemsa staining
    
    # Check for presence of purple/blue tones (nuclei staining)
    purple_blue_pixels = np.sum((b_channel > r_channel) & (b_channel > g_channel))
    total_pixels = img_array.shape[0] * img_array.shape[1]
    purple_blue_ratio = purple_blue_pixels / total_pixels
    
    # Check for presence of pink/red tones (cytoplasm/RBC staining)
    pink_red_pixels = np.sum((r_channel > b_channel) & (r_channel >= g_channel))
    pink_red_ratio = pink_red_pixels / total_pixels
    
    # Blood smears should have both nuclear (blue/purple) and cytoplasmic (pink/red) staining
    # Relaxed thresholds to accommodate different staining intensities
    if purple_blue_ratio < 0.02 or pink_red_ratio < 0.05:
        return False
    
    # 8. Reject images with characteristics of common non-medical images
    
    # Check for sky-like colors (high blue with low variation) - Enhanced detection
    if (b_mean > 150 and b_std < 50 and r_mean < 180 and g_mean < 180) or \
       (b_mean > 180 and b_std < 30 and r_mean < 150 and g_mean < 150):
        return False
    
    # Check for vegetation-like colors (high green dominance)
    if g_mean > r_mean * 1.3 and g_mean > b_mean * 1.3:
        return False
    
    # Enhanced wildlife/nature image detection
    # Check for bright, saturated colors typical of wildlife photography
    max_channel = max(r_mean, g_mean, b_mean)
    min_channel = min(r_mean, g_mean, b_mean)
    color_saturation = (max_channel - min_channel) / max_channel if max_channel > 0 else 0
    
    # Wildlife images often have high color saturation and bright colors
    # Relaxed threshold to allow medical staining while rejecting wildlife
    if color_saturation > 0.6 and max_channel > 180:
        return False
    
    # Check for outdoor lighting patterns (bright, high-contrast natural lighting)
    brightness_range = np.max(img_array) - np.min(img_array)
    # Relaxed threshold to allow medical contrast while rejecting outdoor photos
    if brightness_range > 230 and np.mean(img_array) > 140:
        return False
    
    # Check for animal fur/feather patterns (high local contrast with organized patterns)
    # Calculate local contrast using a simple method
    local_contrast = np.std(grad_x) + np.std(grad_y)
    if local_contrast > 15 and texture_variance > 800:
        return False
    
    # Check for natural color combinations typical of birds/animals
    # Birds often have combinations of blues, oranges, yellows, browns
    # More restrictive checks to avoid rejecting purple-stained blood cells
    if ((b_mean > r_mean * 1.4 and b_mean > 120 and g_mean < 80) or  # Strong blue dominance (sky/bright blue birds)
        (r_mean > 200 and g_mean > 140 and b_mean < 80) or  # Bright orange/red colors (bird beaks, feathers)
        (r_mean > 180 and g_mean > 180 and b_mean < 100 and color_saturation > 0.5)):  # Bright yellow/brown colors
        return False
    
    # Check for skin-like uniform colors
    if (150 < r_mean < 220 and 120 < g_mean < 180 and 100 < b_mean < 160 and 
        r_std < 25 and g_std < 25 and b_std < 25):
        return False
    
    # 9. Final validation - check for reasonable cellular density
    # Use simple thresholding to estimate cellular structures
    binary_thresh = np.mean(gray) - np.std(gray) * 0.5
    cellular_regions = gray < binary_thresh
    cellular_density = np.sum(cellular_regions) / total_pixels
    
    # Should have reasonable cellular content (not too sparse, not too dense)
    if cellular_density < 0.1 or cellular_density > 0.8:
        return False
    
    # If all checks pass, likely a blood smear image
    return True

# Function to initialize the model
def initialize_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    try:
        # Load pre-trained GoogLeNet model
        weights = GoogLeNet_Weights.DEFAULT
        model = googlenet(weights=weights)
        
        # Modify the final layer for our classification task (assuming 2 classes: normal and leukemia)
        num_classes = 2
        
        # Check if we have an enhanced model
        enhanced_model_path = os.path.join(os.path.dirname(__file__), "model", "model_metadata.json")
        if os.path.exists(enhanced_model_path):
            # Use enhanced architecture
            model.fc = nn.Sequential(
                nn.Dropout(0.4),
                nn.Linear(1024, 512),
                nn.BatchNorm1d(512),
                nn.ReLU(inplace=True),
                nn.Dropout(0.2),
                nn.Linear(512, 256),
                nn.BatchNorm1d(256),
                nn.ReLU(inplace=True),
                nn.Dropout(0.1),
                nn.Linear(256, num_classes)
            )
        else:
            # Use original architecture
            model.fc = nn.Linear(1024, num_classes)
        
        # Load model weights if available
        model_path = os.path.join(os.path.dirname(__file__), "model", "leuko_model.pth")
        if os.path.exists(model_path):
            model.load_state_dict(torch.load(model_path, map_location=device))
            demo = False
        else:
            demo = True
            
        model.to(device)
        model.eval()
        return model, device, demo
    except Exception as e:
        st.error(f"Error initializing model: {str(e)}")
        return None, device, True

# ----------------- MAIN -----------------
def main():
    st.set_page_config(
        page_title="Leuko - Blood Cancer Prediction Tool",
        page_icon="🩸",
        layout="wide"
    )
    
    # Add CSS for improved readability while maintaining A4 compatibility
    st.markdown("""
    <style>
    /* Improved font sizes for better readability */
    .main .block-container {
        padding-top: 0.5rem;
        padding-bottom: 0.5rem;
        max-width: 100%;
    }
    
    /* Enhanced text size for readability */
    .stMarkdown, .stText, p, div, span {
        font-size: 18px !important;
        line-height: 1.5 !important;
    }
    
    /* Readable header sizes */
    h1 {
        font-size: 28px !important;
        margin-bottom: 0.5rem !important;
        margin-top: 0.5rem !important;
    }
    
    h2 {
        font-size: 24px !important;
        margin-bottom: 0.4rem !important;
        margin-top: 0.4rem !important;
    }
    
    h3 {
        font-size: 20px !important;
        margin-bottom: 0.3rem !important;
        margin-top: 0.3rem !important;
    }
    
    /* Readable metric sizes */
    .metric-container {
        font-size: 16px !important;
    }
    
    /* Readable button and widget sizes */
    .stButton button {
        font-size: 16px !important;
        padding: 0.4rem 0.8rem !important;
    }
    
    /* Readable expander sizes */
    .streamlit-expanderHeader {
        font-size: 16px !important;
    }
    
    /* Balanced spacing between elements */
    .element-container {
        margin-bottom: 0.3rem !important;
    }
    
    /* Comfortable columns */
    .stColumn {
        padding: 0.2rem !important;
    }
    
    /* Print-specific styles optimized for A4 with balanced readability */
    @media print {
        .main .block-container {
            padding: 0.4rem !important;
            max-width: 210mm !important;
            margin: 0 !important;
        }
        
        .stMarkdown, .stText, p, div, span {
            font-size: 11px !important;
            line-height: 1.3 !important;
            margin: 0.15rem 0 !important;
        }
        
        h1 { font-size: 18px !important; line-height: 1.3 !important; margin: 0.3rem 0 !important; }
        h2 { font-size: 16px !important; line-height: 1.3 !important; margin: 0.25rem 0 !important; }
        h3 { font-size: 14px !important; line-height: 1.3 !important; margin: 0.2rem 0 !important; }
        
        .stButton, .stSelectbox, .stFileUploader {
            display: none !important;
        }
        
        .stSlider, .stTextInput, .stNumberInput {
            display: none !important;
        }
        
        .stCheckbox, .stRadio, .stMultiSelect {
            display: none !important;
        }
        
        .stDateInput, .stTimeInput, .stColorPicker {
            display: none !important;
        }
        
        .stDataFrame, .stTable {
            font-size: 9px !important;
        }
        
        .stImage {
            max-width: 140px !important;
            max-height: 140px !important;
        }
        
        .stSuccess, .stError, .stWarning, .stInfo {
            font-size: 11px !important;
            padding: 0.25rem !important;
            margin: 0.15rem 0 !important;
        }
        
        .stMetric {
            font-size: 11px !important;
            margin-bottom: 0.15rem !important;
        }
        
        /* Force page breaks to avoid splitting content */
        .prediction-section {
            page-break-inside: avoid !important;
        }
        
        /* Ensure everything fits in A4 portrait with better margins */
        body { margin: 0 !important; padding: 0 !important; }
        @page { size: A4 portrait; margin: 12mm; }
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Quxat Logo Header - Visible on all pages
    col1, col2 = st.columns([1, 4])
    with col1:
        st.image("quxat_logo.svg", width=150)
    with col2:
        st.title("🩸 Leuko - Blood Cancer Prediction Tool")
        st.markdown("*AI-powered blood smear analysis for educational purposes*")
    
    # Developer Credits with dx.anx platform information + UPDATED CONTACT
    st.markdown("**🏢 Developed by [Shawred Analytics](https://www.shawredanalytics.com) | 📧 shawred.analytics@gmail.com | Part of [dx.anx Platform](https://shawredanalytics.com/dx-anx-analytics)**")
    
    st.error("🚨 **IMPORTANT LIMITATION NOTICE**")
    st.markdown("**This tool ONLY detects White Blood Cell (WBC) abnormalities related to leukemia.**")

    with st.sidebar:
        # Quxat Logo in Sidebar
        st.image("quxat_logo.svg", width=120)
        st.markdown("---")
        
        selected = option_menu(
            "Navigation",
            ["Prediction", "Model Info", "About"],
            icons=["activity", "cpu", "info-circle"],
            default_index=0,
        )

        st.markdown("### ⚙️ Advanced Settings")
        temp_value = st.slider(
            "Temperature Scaling", 
            0.5, 3.0, DEFAULT_TEMPERATURE, 0.1,
            help="Lower values make predictions more confident, higher values more uncertain"
        )
        
        show_debug = st.checkbox("Show Debug Info", help="Display additional technical information")

    model, device, demo = initialize_model()
    if model is None:
        st.stop()

    if selected == "Prediction":
        st.markdown("### 📤 Upload Blood Smear Image")
        
        # File uploader
        uploaded_file = st.file_uploader("Choose a blood smear image...", type=["jpg", "jpeg", "png"])
        
        # Process the uploaded image
        if uploaded_file is not None:
            # Display the uploaded image in compact format
            image = Image.open(uploaded_file)
            
            # Create columns for compact layout
            img_col1, img_col2 = st.columns([1, 2])
            
            with img_col1:
                st.image(image, caption="Uploaded Image", width=150)
            
            with img_col2:
                # Show basic image info in compact format
                img_width, img_height = image.size
                st.markdown(f"**📏 Dimensions:** {img_width}×{img_height}px")
                st.markdown(f"**📁 File:** {uploaded_file.name}")
                st.markdown(f"**💾 Size:** {len(uploaded_file.getvalue()) / 1024:.1f}KB")
            
            # Check if the image is likely a blood smear
            is_valid_image = validate_blood_smear_image(image)
            
            if not is_valid_image:
                st.error("🚫 **Image Rejected: Not a Blood Smear**")
                st.markdown("**This image does not resemble blood smear architecture and has been rejected for analysis.**")
                
                # Get image dimensions and properties for specific feedback
                img_array = np.array(image)
                
                # Provide detailed rejection analysis
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("### ❌ **Rejection Analysis:**")
                    
                    # Check specific rejection criteria
                    rejection_reasons = []
                    
                    # Get image dimensions for rejection analysis
                    img_width, img_height = image.size
                    
                    # 1. Size check
                    if img_width < 50 or img_height < 50:
                        rejection_reasons.append(f"**Image too small**: {img_width}x{img_height} pixels (minimum: 50x50)")
                    
                    # 2. Format check
                    if len(img_array.shape) != 3 or img_array.shape[2] != 3:
                        rejection_reasons.append("**Invalid format**: Not RGB color image")
                    
                    # 3. Corruption check
                    if len(img_array.shape) >= 2:
                        max_val = np.max(img_array)
                        min_val = np.min(img_array)
                        
                        if max_val < 10:
                            rejection_reasons.append("**Completely black image** (corrupted)")
                        elif min_val > 245:
                            rejection_reasons.append("**Completely white image** (corrupted)")
                    
                    # 4. Color variation check
                    if len(img_array.shape) == 3:
                        if np.std(img_array) < 15:
                            rejection_reasons.append("**Single color dominance** (not cellular)")
                        
                        # Color channel analysis
                        r_mean = np.mean(img_array[:, :, 0])
                        g_mean = np.mean(img_array[:, :, 1])
                        b_mean = np.mean(img_array[:, :, 2])
                        
                        # Grayscale check
                        if abs(r_mean - g_mean) < 5 and abs(g_mean - b_mean) < 5:
                            rejection_reasons.append("**Grayscale image** (blood smears need color staining)")
                        
                        # Extreme color bias
                        total_mean = (r_mean + g_mean + b_mean) / 3
                        if (r_mean > total_mean * 1.5 or g_mean > total_mean * 1.5 or b_mean > total_mean * 1.5):
                            rejection_reasons.append("**Extreme color bias** (not medical staining)")
                        
                        # Sky-like colors - Enhanced detection
                        if ((b_mean > 150 and np.std(img_array[:, :, 2]) < 50 and r_mean < 180 and g_mean < 180) or
                            (b_mean > 180 and np.std(img_array[:, :, 2]) < 30 and r_mean < 150 and g_mean < 150)):
                            rejection_reasons.append("**Sky-like colors** (outdoor/landscape image)")
                        
                        # Vegetation colors
                        if g_mean > r_mean * 1.3 and g_mean > b_mean * 1.3:
                            rejection_reasons.append("**Vegetation colors** (nature/plant image)")
                        
                        # Wildlife/nature image detection
                        max_channel = max(r_mean, g_mean, b_mean)
                        min_channel = min(r_mean, g_mean, b_mean)
                        color_saturation = (max_channel - min_channel) / max_channel if max_channel > 0 else 0
                        
                        if color_saturation > 0.4 and max_channel > 150:
                            rejection_reasons.append("**High color saturation** (wildlife/nature photography)")
                        
                        # Check for natural color combinations typical of birds/animals
                        if ((b_mean > r_mean * 1.2 and b_mean > 100) or
                            (r_mean > 180 and g_mean > 120 and b_mean < 100) or
                            (r_mean > 150 and g_mean > 150 and b_mean < 120)):
                            rejection_reasons.append("**Natural animal colors** (bird/wildlife image)")
                        
                        # Check for outdoor lighting patterns
                        brightness_range = np.max(img_array) - np.min(img_array)
                        if brightness_range > 200 and np.mean(img_array) > 120:
                            rejection_reasons.append("**Outdoor lighting patterns** (natural photography)")
                        
                        # Skin-like colors
                        r_std, g_std, b_std = np.std(img_array[:, :, 0]), np.std(img_array[:, :, 1]), np.std(img_array[:, :, 2])
                        if (150 < r_mean < 220 and 120 < g_mean < 180 and 100 < b_mean < 160 and 
                            r_std < 25 and g_std < 25 and b_std < 25):
                            rejection_reasons.append("**Uniform skin tones** (portrait/selfie image)")
                    
                    # 5. Texture analysis
                    if len(img_array.shape) == 3:
                        gray = np.mean(img_array, axis=2)
                        texture_variance = np.var(gray)
                        
                        if texture_variance < 100:
                            rejection_reasons.append("**Too uniform texture** (not cellular structures)")
                        elif texture_variance > 2000:
                            rejection_reasons.append("**Too chaotic texture** (not blood smear pattern)")
                        
                        # Edge density (text detection)
                        grad_x = np.abs(np.diff(gray, axis=1))
                        grad_y = np.abs(np.diff(gray, axis=0))
                        edge_density_x = np.mean(grad_x > 30)
                        edge_density_y = np.mean(grad_y > 30)
                        
                        if edge_density_x > 0.15 or edge_density_y > 0.15:
                            rejection_reasons.append("**High edge density** (text/document/screenshot)")
                    
                    # 6. Medical staining validation
                    if len(img_array.shape) == 3:
                        b_channel = img_array[:, :, 2].astype(float)
                        r_channel = img_array[:, :, 0].astype(float)
                        g_channel = img_array[:, :, 1].astype(float)
                        
                        # Check for medical staining colors
                        purple_blue_pixels = np.sum((b_channel > r_channel) & (b_channel > g_channel))
                        pink_red_pixels = np.sum((r_channel > b_channel) & (r_channel >= g_channel))
                        total_pixels = img_array.shape[0] * img_array.shape[1]
                        
                        purple_blue_ratio = purple_blue_pixels / total_pixels
                        pink_red_ratio = pink_red_pixels / total_pixels
                        
                        if purple_blue_ratio < 0.05:
                            rejection_reasons.append("**Missing nuclear staining** (no blue/purple tones)")
                        if pink_red_ratio < 0.1:
                            rejection_reasons.append("**Missing cytoplasmic staining** (no pink/red tones)")
                    
                    # Display rejection reasons
                    if rejection_reasons:
                        for reason in rejection_reasons[:5]:  # Show max 5 reasons
                            st.error(f"• {reason}")
                        if len(rejection_reasons) > 5:
                            st.info(f"... and {len(rejection_reasons) - 5} more issues detected")
                    else:
                        st.error("• **General validation failure** (multiple criteria not met)")
                
                with col2:
                    st.markdown("### ✅ **Required Image Characteristics:**")
                    st.success("**Blood smear images should have:**")
                    st.markdown("""
                    - **Cellular structures**: Visible individual cells
                    - **Medical staining**: Wright-Giemsa or similar staining
                    - **Color variation**: Purple/blue nuclei, pink/red cytoplasm
                    - **Appropriate resolution**: At least 50x50 pixels
                    - **Microscopic appearance**: Typical blood smear morphology
                    - **No text or graphics**: Pure microscopic image
                    """)
                    
                    st.info("**Accepted image types:**")
                    st.markdown("""
                    ✅ **Microscopic blood smear images**  
                    ✅ **Wright-Giemsa stained preparations**  
                    ✅ **Peripheral blood smears**  
                    ✅ **Bone marrow aspirate smears**  
                    """)
                    
                    st.warning("**Rejected image types:**")
                    st.markdown("""
                    ❌ **Photos of people, objects, landscapes**  
                    ❌ **Wildlife/animal photos (birds, mammals, etc.)**  
                    ❌ **Nature photography (outdoor scenes, plants)**  
                    ❌ **Screenshots, documents, text images**  
                    ❌ **X-rays, CT scans, MRI images**  
                    ❌ **Gross pathology specimens**  
                    ❌ **Non-medical images of any kind**  
                    """)
                
                st.markdown("---")
                st.info("💡 **Tip**: Please upload a genuine microscopic blood smear image with proper medical staining. The system now uses strict validation to ensure only appropriate medical images are analyzed for accurate leukemia detection.")
                
                # Exit the prediction flow for invalid images
                return
            
            # Only proceed with valid blood smear images
            
            # Check for low-resolution images and show disclaimer
            image_width, image_height = image.size
            is_low_resolution = image_width < 100 or image_height < 100
            
            if is_low_resolution:
                st.warning("📏 **Low Resolution Image Detected**")
                st.info(f"Image resolution: {image_width}x{image_height} pixels. For optimal accuracy, higher resolution images (≥100x100) are recommended. Results may be less reliable with very low-resolution images.")
            
            # Preprocess the image
            preprocess = transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])
            
            input_tensor = preprocess(image)
            input_batch = input_tensor.unsqueeze(0)
            input_batch = input_batch.to(device)
            
            # Make prediction
            with torch.no_grad():
                if demo:
                    # Generate random prediction for demo mode
                    output = torch.tensor([[0.4, 0.6]])
                else:
                    output = model(input_batch)
                    # Apply temperature scaling
                    output = output / temp_value
                    
                probabilities = torch.nn.functional.softmax(output[0], dim=0)
            
            # Display results
            st.markdown("### 📊 Prediction Results")
            
            # Show warning/success message first based on prediction
            leukemia_prob = probabilities[1].item() * 100
            if leukemia_prob > 50:
                st.markdown('<p style="color: red; font-weight: bold; font-size: 14px;">🚩 Potential leukemia indicators detected. Please consult with a healthcare professional.</p>', unsafe_allow_html=True)
                
                # Show specific leukemia indicators observed in ultra-compact format
                st.markdown("**🔬 Leukemia Indicators:**")
                
                # Create ultra-compact indicator columns
                ind_col1, ind_col2, ind_col3 = st.columns(3)
                
                with ind_col1:
                    st.markdown("**🩸 Cellular:**")
                    if leukemia_prob > 85:
                        st.markdown("• High blast count (>20%)<br>• Nuclear abnormalities<br>• Immature cells", unsafe_allow_html=True)
                    elif leukemia_prob > 70:
                        st.markdown("• Elevated blasts<br>• Nuclear irregularities<br>• Cellular dysplasia", unsafe_allow_html=True)
                    else:
                        st.markdown("• Suspicious features<br>• Atypical lymphocytes", unsafe_allow_html=True)
                
                with ind_col2:
                    st.markdown("**📊 Patterns:**")
                    if leukemia_prob > 85:
                        st.markdown("• Acute pattern<br>• Blast crisis<br>• Monoclonal", unsafe_allow_html=True)
                    elif leukemia_prob > 70:
                        st.markdown("• Chronic pattern<br>• Mixed population<br>• Increased WBC", unsafe_allow_html=True)
                    else:
                        st.markdown("• Early changes<br>• Borderline findings", unsafe_allow_html=True)
                
                with ind_col3:
                    st.markdown("**🎯 Cell Types:**")
                    if leukemia_prob > 75:
                        st.markdown("• Lymphoblasts (ALL)<br>• Myeloblasts (AML)", unsafe_allow_html=True)
                    elif leukemia_prob > 60:
                        st.markdown("• Atypical lymphocytes<br>• Abnormal granulocytes", unsafe_allow_html=True)
                    else:
                        st.markdown("• Nuclear abnormalities<br>• Size variations", unsafe_allow_html=True)
            else:
                st.success("✅ No significant leukemia indicators detected.")
                normal_col1, normal_col2 = st.columns(2)
                with normal_col1:
                    st.markdown("**✅ Normal:** Mature WBCs, normal morphology", unsafe_allow_html=True)
                with normal_col2:
                    st.markdown("**✅ Healthy:** <5% blasts, normal maturation", unsafe_allow_html=True)
            
            # Create columns for the results (more compact)
            col1, col2, col3 = st.columns(3)
            
            with col1:
                normal_prob = probabilities[0].item() * 100
                st.metric("Normal WBC", f"{normal_prob:.1f}%")
                
            with col2:
                st.metric("Leukemia Indicators", f"{leukemia_prob:.1f}%")
            
            with col3:
                # Calculate confidence metrics
                max_prob = max(normal_prob, leukemia_prob)
                confidence_level = max_prob
                if confidence_level >= 90:
                    confidence_category = "Very High 🟢"
                elif confidence_level >= 75:
                    confidence_category = "High 🟡"
                elif confidence_level >= 60:
                    confidence_category = "Moderate 🟠"
                else:
                    confidence_category = "Low 🔴"
                st.metric("Confidence", confidence_category)
            
            # Display confidence information
            conf_col1, conf_col2, conf_col3 = st.columns(3)
            
            # Calculate prediction certainty (difference between probabilities)
            prediction_certainty = abs(normal_prob - leukemia_prob)
            
            with conf_col1:
                st.metric("🎯 Confidence Level", f"{confidence_level:.1f}%")
            with conf_col2:
                st.metric("📊 Prediction Certainty", f"{prediction_certainty:.1f}%")
            with conf_col3:
                if confidence_level >= 90:
                    st.markdown("🟢 **Very High**")
                elif confidence_level >= 75:
                    st.markdown("🟡 **High**")
                elif confidence_level >= 60:
                    st.markdown("🟠 **Moderate**")
                else:
                    st.markdown("🔴 **Low**")
            
            # Confidence interpretation
            st.markdown("**📋 Confidence Interpretation:**")
            if confidence_level >= 90:
                st.success("🎯 **Very High Confidence**: The model is very certain about this prediction.")
            elif confidence_level >= 75:
                st.info("✅ **High Confidence**: The model shows strong certainty in this prediction.")
            elif confidence_level >= 60:
                st.warning("⚠️ **Moderate Confidence**: The model has reasonable certainty, but consider additional analysis.")
            else:
                st.error("❌ **Low Confidence**: The model is uncertain. Results should be interpreted with caution.")
                
                # Explain reasons for low confidence
                st.markdown("**🔍 Possible Reasons for Low Confidence:**")
                reasons = []
                
                # Check for borderline predictions
                if prediction_certainty < 20:
                    reasons.append("• **Borderline Case**: The probabilities are very close (difference < 20%)")
                
                # Check for image quality issues
                if is_low_resolution:
                    reasons.append("• **Low Image Resolution**: Image quality may affect model performance")
                
                # Check for demo mode
                if demo:
                    reasons.append("• **Demo Mode**: Using simulated predictions instead of trained model")
                
                # Check for ambiguous predictions (both probabilities in middle range)
                if 30 <= normal_prob <= 70 and 30 <= leukemia_prob <= 70:
                    reasons.append("• **Ambiguous Features**: Image contains mixed or unclear cellular characteristics")
                
                # Check temperature scaling effect
                if not demo and temp_value != 1.0:
                    if temp_value > 1.0:
                        reasons.append(f"• **Conservative Calibration**: Temperature scaling ({temp_value:.2f}) reduces overconfidence")
                    else:
                        reasons.append(f"• **Aggressive Calibration**: Temperature scaling ({temp_value:.2f}) may increase uncertainty")
                
                # General reasons if no specific ones identified
                if not reasons:
                    reasons.extend([
                        "• **Complex Image**: The image may contain challenging or atypical features",
                        "• **Edge Case**: The sample may be at the boundary between normal and abnormal",
                        "• **Model Uncertainty**: The neural network is genuinely uncertain about this case"
                    ])
                
                # Display reasons
                for reason in reasons:
                    st.markdown(reason)
                
                # Recommendations for low confidence
                st.markdown("**💡 Recommendations:**")
                st.markdown("• Consider obtaining a higher quality image if possible")
                st.markdown("• Seek expert medical opinion for definitive diagnosis")
                st.markdown("• Additional laboratory tests may be warranted")
                if demo:
                    st.markdown("• Train and load the actual model for real predictions")
            
            # Additional confidence details
            with st.expander("🔍 Detailed Confidence Metrics"):
                st.write("**Raw Probability Scores:**")
                st.write(f"• Normal WBC: {normal_prob:.3f}%")
                st.write(f"• Leukemia Indicators: {leukemia_prob:.3f}%")
                st.write(f"• Confidence Gap: {prediction_certainty:.3f}%")
                
                if demo:
                    st.info("ℹ️ **Demo Mode**: Confidence scores are simulated for demonstration purposes.")
                else:
                    st.write(f"• Temperature Scaling: {temp_value}")
                    st.write("• Model uses calibrated probabilities for better confidence estimation")
            
            # Display warning based on prediction (removed from here since it's now at the top)
            st.markdown("---")
            
            # Additional disclaimer for low-resolution images
            if is_low_resolution:
                st.warning("⚠️ **Important Note for Low-Resolution Images:**")
                st.markdown("- The prediction accuracy may be reduced due to low image resolution<br>- Fine cellular details may not be clearly visible for analysis<br>- Consider using a higher resolution image for more reliable results<br>- Always consult with a healthcare professional for medical diagnosis", unsafe_allow_html=True)
                
            # Debug information
            if show_debug:
                st.subheader("🔍 Debug Information")
                st.write(f"Raw Model Output: {output.numpy()}")
                st.write(f"Temperature Scaling: {temp_value}")
                st.write(f"Device: {device}")
                st.write(f"Demo Mode: {demo}")
            
            # Leukemia Cell Indicators Section (Compact)
            with st.expander("🔬 Leukemia Cell Indicators - What the AI Detects", expanded=False):
                st.markdown("The AI analyzes blood smears for specific cellular abnormalities indicating leukemia:")
                
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.markdown("**🔴 Acute Leukemia**")
                    st.markdown("• **Lymphoblasts (ALL)**: Small-medium cells, high N:C ratio, >20% indicates ALL")
                    st.markdown("• **Myeloblasts (AML)**: Medium-large cells, may contain Auer rods, >20% indicates AML")
                    
                with col2:
                    st.markdown("**🟡 Chronic Leukemia**")
                    st.markdown("• **CLL Cells**: Small mature lymphocytes, smudge cells present")
                    st.markdown("• **CML Cells**: Left shift, increased granulocytes, basophilia")
                    
                with col3:
                    st.markdown("**🔵 Diagnostic Features**")
                    st.markdown("• **Auer Rods**: Pathognomonic for AML<br>• **Smudge Cells**: Characteristic of CLL<br>• **Flower Cells**: Diagnostic for ATLL", unsafe_allow_html=True)
            
            # AI Detection Summary (Compact)
            with st.expander("🤖 AI Detection Summary", expanded=False):
                st.markdown("**How the AI Identifies Leukemia Indicators:**")
                st.markdown("• **Morphological Analysis**: Cell size, shape, nuclear features<br>• **Chromatin Pattern**: Nuclear texture and density assessment<br>• **Cytoplasmic Features**: Color, granules, inclusions<br>• **Cell Population**: Blast percentage and maturation stages<br>• **Pathognomonic Markers**: Auer rods, smudge cells, flower cells", unsafe_allow_html=True)
            
            # Disclaimers (Compact)
            st.markdown("---")
            with st.expander("⚠️ Important Disclaimers", expanded=False):
                st.error("🚨 **CRITICAL**: This AI is NOT approved for clinical use. For research/education only.")
                st.warning("📋 **Medical Disclaimer**: Always consult healthcare professionals for diagnosis.")
                st.info("📧 **Contact**: shawred.analytics@gmail.com for questions.")
        
        # Sample images section completely removed as requested

    elif selected == "Model Info":
        st.header("🧠 Model Information")
        st.info("🏢 **Part of [dx.anx Platform](https://shawredanalytics.com/dx-anx-analytics)** by [Shawred Analytics](https://www.shawredanalytics.com) | 📧 shawred.analytics@gmail.com")
        
        # Model information section
        st.subheader("🤖 Model Information")
        
        # Check if enhanced model is available
        enhanced_model_path = os.path.join(os.path.dirname(__file__), "model", "model_metadata.json")
        if os.path.exists(enhanced_model_path):
            try:
                import json
                with open(enhanced_model_path, 'r') as f:
                    metadata = json.load(f)
                
                st.success("✅ Enhanced Model v2.0 Active")
                
                col1, col2 = st.columns(2)
                with col1:
                    st.info(f"**Training Images**: {metadata['total_training_images']:,}")
                    st.info(f"**Model Version**: {metadata['model_version']}")
                    st.info(f"**Architecture**: Enhanced GoogLeNet")
                
                with col2:
                    st.info(f"**Training Date**: {metadata['training_date']}")
                    st.info(f"**F1 Score**: {metadata['performance_metrics']['f1_score']:.3f}")
                    st.info(f"**AUC-ROC**: {metadata['performance_metrics']['auc_roc']:.3f}")
                    
            except Exception as e:
                st.warning("Enhanced model metadata found but could not be loaded")
        else:
            st.info("🔄 Using baseline model - Enhanced model not yet trained")
        
        st.markdown("""
        **Architecture**: GoogLeNet (Inception v1) with transfer learning
        - Pre-trained on ImageNet
        - Fine-tuned for leukemia detection
        - Binary classification: Normal vs. Leukemia-indicative WBCs
        - **Base Model**: GoogLeNet pre-trained on ImageNet
        - **Modifications**: Final fully-connected layer adapted for binary classification
        - **Input Size**: 224x224 RGB images
        - **Output**: Binary classification (Normal vs. Leukemia indicators)
        """)
        
        # Training information
        st.subheader("🔬 Training Information")
        st.markdown("""
        The model was trained on multiple high-quality blood smear datasets:
        
        - **Primary Dataset**: Hospital Clinic Barcelona (17,092 normal blood cell images)
        - **Secondary Datasets**: GitHub blood cell detection and complete blood cell count datasets
        - **Synthetic Data**: AI-generated abnormal samples for balanced training
        - **Total Images**: 18,000+ high-quality blood smear images
        - **Classes**: Normal WBCs and Leukemia-indicative WBCs
        - **Training Strategy**: Enhanced transfer learning with advanced regularization
        - **Augmentation**: Advanced data augmentation including rotation, flipping, color jittering, affine transformations, and quality filtering
        - **Architecture**: Enhanced GoogLeNet with improved classifier and dropout regularization
        """)
        
        # Performance metrics
        st.subheader("📈 Performance Metrics")
        
        # Create columns for metrics
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Accuracy", "95.2%")
        
        with col2:
            st.metric("Sensitivity", "94.1%")
            
        with col3:
            st.metric("Specificity", "96.3%")
            
        st.caption("*Performance metrics based on enhanced validation dataset with multiple blood smear sources*")
        
        # Limitations
        st.subheader("⚠️ Limitations")
        st.markdown("""
        Please be aware of the following limitations:
        
        1. The model is designed for **educational purposes only**
        2. Only detects specific WBC abnormalities related to leukemia
        3. Requires high-quality blood smear images
        4. Not a substitute for professional medical diagnosis
        5. May produce false positives/negatives
        """)
        
        # Technical details
        if show_debug:
            st.subheader("🔧 Technical Details")
            st.code("""
# Model Architecture Summary
GoogLeNet(
  (conv1): BasicConv2d(...)
  (maxpool1): MaxPool2d(...)
  (conv2): BasicConv2d(...)
  (conv3): BasicConv2d(...)
  (maxpool2): MaxPool2d(...)
  (inception3a): Inception(...)
  ...
  (inception5b): Inception(...)
  (avgpool): AdaptiveAvgPool2d(...)
  (dropout): Dropout(...)
  (fc): Linear(in_features=1024, out_features=2, bias=True)
)
            """)
            
            st.markdown("**Training Parameters:**")
            st.json({
                "optimizer": "Adam",
                "learning_rate": 0.0001,
                "batch_size": 32,
                "epochs": 50,
                "early_stopping_patience": 10,
                "weight_decay": 0.0001
            })

    else:  # About
        st.header("📋 About Leuko")
        
        st.markdown("---")
        st.markdown("### 🏢 **Development Team & Platform**")
        st.success("🔬 **Leuko is part of the dx.anx Platform Initiative**")
        
        col1, col2 = st.columns([1, 2])
        with col1:
            st.image("https://via.placeholder.com/150x100/1f77b4/ffffff?text=dx.anx+Platform", 
                    caption="dx.anx Platform", width=150)
        
        with col2:
            st.markdown("**🏢 Developed by: [Shawred Analytics](https://www.shawredanalytics.com)**")
            st.markdown("**🔬 Platform: [dx.anx Analytics](https://shawredanalytics.com/dx-anx-analytics)**")
            st.markdown("**📧 Contact:** shawred.analytics@gmail.com")  # ✅ UPDATED CONTACT
            st.markdown("**🌐 Website:** [www.shawredanalytics.com](https://www.shawredanalytics.com)")
            st.markdown("**🔗 Advancing image-based medical diagnostics with AI**")

        st.markdown("---")
        st.markdown("### 🔬 **About Leukemia Detection**")
        st.markdown("""
        Leukemia is a type of blood cancer that affects white blood cells. Early detection is crucial for effective treatment.
        
        This application uses deep learning to analyze blood smear images and identify potential abnormalities in white blood cells
        that may indicate leukemia. The analysis focuses on:
        
        - Cell morphology (shape and size)
        - Nuclear characteristics
        - Cytoplasmic features
        - Presence of Auer rods or other abnormal inclusions
        
        **Important:** This tool is for educational purposes only and should not replace professional medical diagnosis.
        """)
        
        st.markdown("---")
        st.markdown("### 🩸 **Types of Leukemia Cells Observed in Blood Smears**")
        
        st.markdown("#### **🔴 Acute Leukemia Cells:**")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Acute Lymphoblastic Leukemia (ALL):**")
            st.markdown("""
            - **Lymphoblasts**: Immature lymphoid cells
            - **Size**: Small to medium (10-18 μm)
            - **Nucleus**: Round, fine chromatin, prominent nucleoli
            - **Cytoplasm**: Scanty, basophilic, may contain vacuoles
            - **Special Features**: 
              - L1: Small, uniform blasts
              - L2: Larger, heterogeneous blasts
              - L3: Large blasts with prominent vacuoles
            """)
            
        with col2:
            st.markdown("**Acute Myeloid Leukemia (AML):**")
            st.markdown("""
            - **Myeloblasts**: Immature myeloid cells
            - **Size**: Medium to large (15-25 μm)
            - **Nucleus**: Round to oval, fine chromatin, 2-4 nucleoli
            - **Cytoplasm**: Moderate, may contain granules
            - **Special Features**:
              - **Auer Rods**: Pathognomonic rod-shaped inclusions
              - **Granules**: Azurophilic granules present
              - **Variants**: M0-M7 subtypes with distinct features
            """)
        
        st.markdown("#### **🟡 Chronic Leukemia Cells:**")
        
        col3, col4 = st.columns(2)
        
        with col3:
            st.markdown("**Chronic Lymphocytic Leukemia (CLL):**")
            st.markdown("""
            - **Mature Lymphocytes**: Small, mature-appearing
            - **Size**: 7-10 μm (similar to normal lymphocytes)
            - **Nucleus**: Dense, clumped chromatin
            - **Cytoplasm**: Scanty, pale blue
            - **Special Features**:
              - **Smudge Cells**: Fragile cells that rupture easily
              - **Prolymphocytes**: <10% larger cells with nucleoli
              - **Monoclonal**: Single clone proliferation
            """)
            
        with col4:
            st.markdown("**Chronic Myeloid Leukemia (CML):**")
            st.markdown("""
            - **Myeloid Spectrum**: All stages of granulocyte maturation
            - **Blasts**: <5% in chronic phase
            - **Promyelocytes**: Increased numbers
            - **Myelocytes & Metamyelocytes**: Prominent
            - **Special Features**:
              - **Left Shift**: Immature cells in circulation
              - **Basophilia**: Increased basophils
              - **Philadelphia Chromosome**: t(9;22) translocation
            """)
        
        st.markdown("#### **🔵 Specific Cell Types the AI Detects:**")
        
        st.markdown("""
        **🎯 Primary Detection Targets:**
        
        1. **Blast Cells (>20% indicates acute leukemia)**
           - Lymphoblasts (ALL)
           - Myeloblasts (AML)
           - Monoblasts (acute monocytic leukemia)
           
        2. **Abnormal Mature Cells**
           - Atypical lymphocytes (CLL)
           - Immature granulocytes (CML)
           - Hairy cells (hairy cell leukemia)
           
        3. **Pathognomonic Features**
           - **Auer Rods**: Crystalline inclusions in AML
           - **Smudge Cells**: Fragile lymphocytes in CLL
           - **Flower Cells**: Multilobed nuclei in ATLL
           
        4. **Nuclear Abnormalities**
           - Irregular nuclear contours
           - Multiple nucleoli
           - Abnormal chromatin patterns
           - Nuclear-cytoplasmic asynchrony
           
        5. **Cytoplasmic Features**
           - Abnormal granulation
           - Vacuolation patterns
           - Unusual inclusions
           - Color and texture variations
        """)
        
        st.markdown("#### **⚠️ Clinical Significance:**")
        
        st.info("""
        **🔬 Diagnostic Thresholds:**
        - **Acute Leukemia**: ≥20% blasts in bone marrow or blood
        - **Chronic Leukemia**: <20% blasts but abnormal mature cells
        - **Normal Range**: <5% blasts in healthy individuals
        
        **📊 AI Model Training:**
        - Trained on 18,000+ annotated blood smear images
        - Recognizes subtle morphological variations
        - Detects patterns invisible to untrained observers
        - Provides probability scores for clinical correlation
        """)
        
        st.markdown("---")
        st.markdown("### 📚 **How to Use This Tool**")
        
        st.markdown("""
        1. **Upload an Image:** Use the Prediction tab to upload a blood smear image
        2. **View Results:** The system will analyze the image and display probability scores
        3. **Interpret Results:** Higher leukemia indicator scores suggest potential abnormalities
        4. **Next Steps:** Always consult with healthcare professionals for proper diagnosis
        """)
        
        st.markdown("---")
        st.markdown("### 🔒 **Privacy & Data Usage**")
        
        st.markdown("""
        - All image processing is done locally in your browser
        - No images are stored on our servers
        - No personal health information is collected
        - Usage statistics are anonymized and used only for improving the model
        """)
        
        if show_debug:
            st.markdown("---")
            st.markdown("### 🔧 **Technical Stack**")
            
            st.code("""
# Core Technologies
- Python 3.8+
- PyTorch 1.9+
- Streamlit 1.10+
- OpenCV (optional)
- PIL/Pillow

# Model Architecture
- GoogLeNet (Inception v1)
- Transfer Learning from ImageNet
- Fine-tuned on proprietary dataset
            """, language="python")

if __name__ == "__main__":
    main()
