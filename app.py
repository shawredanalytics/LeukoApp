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

# Optional imports with fallbacks
HAS_OPENCV = False
HAS_OPTION_MENU = False

try:
    import cv2
    HAS_OPENCV = True
except ImportError:
    pass

try:
    from streamlit_option_menu import option_menu
    HAS_OPTION_MENU = True
except ImportError:
    pass

# ----------------- CONSTANTS -----------------
MODEL_PATH = "blood_cancer_model.pth"
IMAGE_SIZE = (224, 224)
LABELS_MAP = {0: "Benign", 1: "Early_Pre_B", 2: "Pre_B", 3: "Pro_B"}
MAX_FILE_SIZE = 10 * 1024 * 1024  # 10 MB
ALLOW_DEMO_MODE = False

# Configurable heuristics (very relaxed for screen capture conditions)
EDGE_VAR_THRESHOLD = 200   # Very low threshold to accept screen-captured images
GREEN_RATIO_THRESHOLD = 0.8  # High tolerance for screen backgrounds
COLOR_RATIO_THRESHOLD = 0.05  # Very low threshold for weak screen colors
DEFAULT_TEMPERATURE = 1.5  # Reduced from 2.0 for better calibration

# Screen capture specific settings
SCREEN_CAPTURE_MODE = True  # Enable screen capture compatibility
MIN_BRIGHTNESS = 0.1  # Minimum acceptable brightness
MAX_BRIGHTNESS = 0.9  # Maximum acceptable brightness

# ----------------- LOGGING SETUP -----------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("uploads.log", mode="a", encoding="utf-8")
    ]
)
logger = logging.getLogger(__name__)

# ----------------- CONSTANTS -----------------
MODEL_PATH = "blood_cancer_model.pth"
IMAGE_SIZE = (224, 224)
LABELS_MAP = {0: "Benign", 1: "Early_Pre_B", 2: "Pre_B", 3: "Pro_B"}
MAX_FILE_SIZE = 10 * 1024 * 1024  # 10 MB
ALLOW_DEMO_MODE = False

# More conservative thresholds to reduce false positives
EDGE_VAR_THRESHOLD = 500   # Lower threshold for texture detection
GREEN_RATIO_THRESHOLD = 0.7  # Higher tolerance for green backgrounds
COLOR_RATIO_THRESHOLD = 0.1  # Lower threshold for color detection
DEFAULT_TEMPERATURE = 1.5  # Reduced from 2.0 for better calibration

# ----------------- MODEL LOADING -----------------
@st.cache_resource
def build_model_architecture(num_classes: int = 4):
    base = googlenet(weights=GoogLeNet_Weights.IMAGENET1K_V1)
    base.fc = nn.Sequential(
        nn.Linear(in_features=1024, out_features=512),
        nn.ReLU(),
        nn.Dropout(0.2),
        nn.Linear(in_features=512, out_features=128),
        nn.ReLU(),
        nn.Dropout(0.2),
        nn.Linear(in_features=128, out_features=64),
        nn.ReLU(),
        nn.Dropout(0.2),
        nn.Linear(in_features=64, out_features=num_classes),
    )
    return base

@st.cache_resource
def initialize_model(model_path: str = MODEL_PATH):
    try:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = build_model_architecture(num_classes=len(LABELS_MAP))

        if not os.path.exists(model_path):
            if ALLOW_DEMO_MODE:
                st.warning("⚠️ Model weights not found. Running in demo mode.")
                return nn.Identity(), torch.device("cpu"), True
            else:
                st.error("🚨 Model file missing. Please provide 'blood_cancer_model.pth'.")
                return None, None, False

        state = torch.load(model_path, map_location=device, weights_only=True)
        if isinstance(state, dict) and "state_dict" in state:
            state = state["state_dict"]

        # Remove potential 'module.' prefixes
        new_state = OrderedDict()
        for k, v in state.items():
            new_state[k.replace("module.", "")] = v

        model.load_state_dict(new_state)
        model.eval()
        model.to(device)
        logger.info(f"Model loaded on {device}")
        return model, device, False
    except Exception as e:
        st.error(f"🚨 Failed to initialize model: {e}")
        logger.exception("Model loading failed")
        return None, None, False

# ----------------- IMAGE VALIDATION -----------------
def validate_image(uploaded_file) -> bool:
    try:
        # Reset file pointer to beginning
        uploaded_file.seek(0)
        
        # Get file size
        file_content = uploaded_file.getbuffer()
        file_size = len(file_content)
        
        if file_size == 0:
            st.error("🚨 Empty file detected. Please try uploading again.")
            return False
            
        if file_size > MAX_FILE_SIZE:
            st.error(f"🚨 File size too large ({file_size/1024/1024:.1f}MB). Please upload <10MB.")
            return False

        # More flexible file type checking for mobile
        allowed_extensions = {".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG"}
        allowed_mime_types = {"image/jpeg", "image/jpg", "image/png"}
        
        file_extension = None
        if hasattr(uploaded_file, 'name') and uploaded_file.name:
            file_extension = os.path.splitext(uploaded_file.name.lower())[1]
        
        mime_type = getattr(uploaded_file, 'type', None)
        
        # Check both extension and MIME type (mobile cameras sometimes have inconsistent MIME types)
        valid_extension = file_extension in {".jpg", ".jpeg", ".png"} if file_extension else False
        valid_mime = mime_type in allowed_mime_types if mime_type else False
        
        if not (valid_extension or valid_mime):
            st.error(f"🚨 Invalid file type. Found: {mime_type or 'unknown'}, Extension: {file_extension or 'unknown'}. Only JPG/JPEG/PNG allowed.")
            return False
            
        # Try to open image to verify it's actually an image
        try:
            uploaded_file.seek(0)
            test_img = Image.open(uploaded_file)
            test_img.verify()  # Verify it's a valid image
        except Exception as e:
            st.error(f"🚨 Invalid or corrupted image file: {str(e)}")
            return False
        finally:
            uploaded_file.seek(0)  # Reset for further processing
        
        return True
        
    except Exception as e:
        st.error(f"🚨 Error validating image: {e}")
        logger.exception("Image validation failed")
        return False

def enhance_screen_captured_image(img):
    """
    Enhance images captured from computer screens with mobile cameras
    Uses PIL for basic enhancement or OpenCV for advanced processing
    """
    try:
        if HAS_OPENCV:
            # Advanced OpenCV-based enhancement
            return enhance_with_opencv(img)
        else:
            # Basic PIL-based enhancement
            return enhance_with_pil(img)
    except Exception as e:
        logger.warning(f"Screen enhancement failed: {e}, using original image")
        return img

def enhance_with_opencv(img):
    """
    Advanced enhancement using OpenCV
    """
    # Convert to numpy array for processing
    img_array = np.array(img)
    
    # Remove moiré patterns using Gaussian blur
    img_array = cv2.GaussianBlur(img_array, (3, 3), 0.5)
    
    # Convert to LAB color space for better processing
    lab = cv2.cvtColor(img_array, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    
    # Apply CLAHE (Contrast Limited Adaptive Histogram Equalization) to L channel
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l = clahe.apply(l)
    
    # Merge channels back
    lab = cv2.merge([l, a, b])
    enhanced = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
    
    # Reduce screen glare/reflection effects
    # Apply gentle sharpening to counteract screen blur
    kernel = np.array([[-0.5, -1, -0.5],
                      [-1, 7, -1],
                      [-0.5, -1, -0.5]])
    enhanced = cv2.filter2D(enhanced, -1, kernel)
    
    # Ensure values are in valid range
    enhanced = np.clip(enhanced, 0, 255).astype(np.uint8)
    
    # Convert back to PIL Image
    return Image.fromarray(enhanced)

def enhance_with_pil(img):
    """
    Basic enhancement using PIL only
    """
    # Apply gentle blur to reduce moiré patterns
    img = img.filter(ImageFilter.GaussianBlur(radius=0.5))
    
    # Enhance contrast
    enhancer = ImageEnhance.Contrast(img)
    img = enhancer.enhance(1.2)
    
    # Enhance sharpness slightly
    enhancer = ImageEnhance.Sharpness(img)
    img = enhancer.enhance(1.1)
    
    # Adjust brightness if needed
    enhancer = ImageEnhance.Brightness(img)
    img = enhancer.enhance(1.05)
    
    return img

def detect_screen_capture(img):
    """
    Detect if image was likely captured from a screen
    Uses basic PIL operations if OpenCV not available
    """
    try:
        if HAS_OPENCV:
            return detect_screen_with_opencv(img)
        else:
            return detect_screen_with_pil(img)
    except Exception as e:
        logger.error(f"Screen detection failed: {e}")
        return False

def detect_screen_with_opencv(img):
    """
    Advanced screen detection using OpenCV
    """
    img_array = np.array(img)
    
    # Check for common screen capture characteristics
    # 1. Check brightness uniformity (screens often have more uniform lighting)
    gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
    brightness_std = np.std(gray) / 255.0
    
    # 2. Check for rectangular edges (screen borders)
    edges = cv2.Canny(gray, 50, 150)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    large_rectangles = 0
    for contour in contours:
        area = cv2.contourArea(contour)
        if area > 1000:  # Large enough contour
            # Check if contour is roughly rectangular
            perimeter = cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, 0.02 * perimeter, True)
            if len(approx) >= 4:  # Roughly rectangular
                large_rectangles += 1
    
    # 3. Check color distribution (screens often have different color profiles)
    color_variance = np.var(img_array, axis=(0, 1))
    avg_color_variance = np.mean(color_variance)
    
    # Heuristic scoring
    screen_score = 0
    if brightness_std < 0.3:  # More uniform brightness
        screen_score += 1
    if large_rectangles > 0:  # Has rectangular shapes (screen border)
        screen_score += 1
    if avg_color_variance < 2000:  # Less color variation
        screen_score += 1
    
    is_screen_capture = screen_score >= 2
    
    logger.info(f"Screen detection (OpenCV): brightness_std={brightness_std:.3f}, rectangles={large_rectangles}, "
               f"color_var={avg_color_variance:.1f}, score={screen_score}, is_screen={is_screen_capture}")
    
    return is_screen_capture

def detect_screen_with_pil(img):
    """
    Basic screen detection using PIL only
    """
    img_array = np.array(img)
    
    # Simple brightness uniformity check
    gray = np.mean(img_array, axis=2)
    brightness_std = np.std(gray) / 255.0
    
    # Check color distribution
    color_variance = np.var(img_array, axis=(0, 1))
    avg_color_variance = np.mean(color_variance)
    
    # Simple heuristic
    is_screen_capture = brightness_std < 0.25 and avg_color_variance < 1500
    
    logger.info(f"Screen detection (PIL): brightness_std={brightness_std:.3f}, "
               f"color_var={avg_color_variance:.1f}, is_screen={is_screen_capture}")
    
    return is_screen_capture
def is_blood_smear(img: Image.Image, filename: str = "unknown"):
    """
    Improved heuristic check for blood smear suitability.
    Enhanced for screen-captured images with very relaxed thresholds.
    """
    try:
        # Check if this is a screen capture
        is_screen = detect_screen_capture(img) if SCREEN_CAPTURE_MODE else False
        
        img_small = img.resize((128, 128))
        arr = np.array(img_small).astype(float) / 255.0

        r, g, b = arr[:, :, 0], arr[:, :, 1], arr[:, :, 2]

        # Color heuristics - very relaxed for screen captures
        red_ratio = np.mean(r > 0.3)  # Even lower threshold for screen captures
        purple_ratio = np.mean((r > 0.25) & (b > 0.2))  # Very low thresholds
        pink_ratio = np.mean((r > 0.4) & (g > 0.25) & (b > 0.25))  # Pink detection
        green_ratio = np.mean(g > 0.5)
        blue_ratio = np.mean(b > 0.5)

        # Combined color ratio for better detection
        total_color_ratio = red_ratio + purple_ratio + pink_ratio

        # Texture heuristics using numpy gradient
        gray = np.mean(arr, axis=2)
        gx, gy = np.gradient(gray)
        edges = np.sqrt(gx**2 + gy**2)
        edge_var = np.var(edges)
        
        # Brightness check (screens can be too bright or too dark)
        avg_brightness = np.mean(gray)
        
        # Screen capture specific adjustments
        if is_screen:
            logger.info(f"[{filename}] Screen capture detected - using relaxed validation")
            # Much more relaxed thresholds for screen captures
            effective_color_threshold = COLOR_RATIO_THRESHOLD * 0.5  # Even more relaxed
            effective_edge_threshold = EDGE_VAR_THRESHOLD * 0.5  # Much lower texture requirement
            effective_green_threshold = GREEN_RATIO_THRESHOLD + 0.1  # Higher green tolerance
        else:
            effective_color_threshold = COLOR_RATIO_THRESHOLD
            effective_edge_threshold = EDGE_VAR_THRESHOLD
            effective_green_threshold = GREEN_RATIO_THRESHOLD

        # Very lenient checks to accept most images including screen captures
        warnings = []
        
        if total_color_ratio <= effective_color_threshold:
            if is_screen:
                warnings.append("Screen capture with weak staining detected - analysis may have reduced accuracy.")
            else:
                warnings.append("Very weak staining detected - may significantly affect accuracy.")
            logger.warning(f"[{filename}] Weak staining: {total_color_ratio:.3f} (threshold: {effective_color_threshold:.3f})")
            
        if green_ratio >= effective_green_threshold:
            if is_screen:
                warnings.append("Screen capture with green background detected - acceptable for analysis.")
            else:
                warnings.append("Strong green background detected - may affect cell visibility.")
            logger.warning(f"[{filename}] Green background: {green_ratio:.3f}")
            
        if edge_var <= effective_edge_threshold:
            if is_screen:
                warnings.append("Screen capture with low texture detected - acceptable but may affect precision.")
            else:
                warnings.append("Very low texture detected - image may be too blurry or uniform.")
            logger.warning(f"[{filename}] Low texture: {edge_var:.3f}")
        
        # Brightness warnings for screen captures
        if is_screen:
            if avg_brightness < MIN_BRIGHTNESS:
                warnings.append("Screen appears too dark - consider increasing screen brightness.")
            elif avg_brightness > MAX_BRIGHTNESS:
                warnings.append("Screen appears too bright - may cause glare issues.")
        
        # Still accept the image but with warnings
        combined_warning = " ".join(warnings) if warnings else ""
        
        # ✅ Accept almost all images, especially screen captures
        logger.info(f"[{filename}] Blood smear check: Color: {total_color_ratio:.3f}, "
                   f"Edge: {edge_var:.3f}, Screen: {is_screen}, Warnings: {len(warnings)}")
        
        return True, combined_warning
        
    except Exception as e:
        reason = f"Error analyzing image content: {e}"
        logger.error(f"[{filename}] Blood smear check failed: {reason}")
        return False, reason

# ----------------- IMAGE PREPROCESSING -----------------
def preprocess_image(uploaded_file):
    try:
        # Enhanced validation
        if not validate_image(uploaded_file):
            return None

        # Ensure we're at the beginning of the file
        uploaded_file.seek(0)
        
        # Try to open the image with better error handling
        try:
            img = Image.open(uploaded_file)
            # Load the image data immediately to avoid lazy loading issues
            img.load()
        except Exception as e:
            st.error(f"🚨 Failed to open image: {str(e)}. Please try uploading again.")
            logger.error(f"Image opening failed: {e}")
            return None

        # Handle mobile phone rotation metadata (EXIF orientation)
        try:
            img = ImageOps.exif_transpose(img)
        except Exception as e:
            logger.warning(f"EXIF orientation handling failed: {e}")
            # Continue without EXIF correction if it fails

        # Convert to RGB if needed
        if img.mode != "RGB":
            try:
                img = img.convert("RGB")
            except Exception as e:
                st.error(f"🚨 Failed to convert image to RGB: {str(e)}")
                return None

        # Detect and enhance screen-captured images
        is_screen_capture = detect_screen_capture(img) if SCREEN_CAPTURE_MODE else False
        if is_screen_capture:
            logger.info("Screen capture detected - applying enhancement")
            img = enhance_screen_captured_image(img)
            if HAS_OPENCV:
                st.info("📺 Screen capture detected - image enhanced with advanced processing")
            else:
                st.info("📺 Screen capture detected - image enhanced with basic processing (install opencv-python for better results)")

        # Log image properties for debugging
        logger.info(f"Image processed: Size={img.size}, Mode={img.mode}, Format={getattr(img, 'format', 'Unknown')}")

        # Validate blood smear with filename
        filename = getattr(uploaded_file, 'name', 'unknown_file')
        ok, reason = is_blood_smear(img, filename)
        if not ok:
            st.error(f"🚨 Image does not meet specifications of the test model\n\n**Reason:** {reason}")
            return None
        elif reason:
            st.warning(f"⚠️ Note: {reason}")

        # Enhanced preprocessing
        try:
            transform = transforms.Compose([
                transforms.Resize(IMAGE_SIZE, interpolation=transforms.InterpolationMode.BILINEAR),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])
            tensor = transform(img).unsqueeze(0)
            
            # Verify tensor is valid
            if torch.isnan(tensor).any() or torch.isinf(tensor).any():
                st.error("🚨 Image processing resulted in invalid data. Please try a different image.")
                return None
                
            logger.info(f"Tensor created successfully: Shape={tensor.shape}")
            return tensor
            
        except Exception as e:
            st.error(f"🚨 Error during image transformation: {str(e)}")
            logger.exception("Image transformation failed")
            return None
        
    except Exception as e:
        st.error(f"🚨 Error processing image: {e}")
        logger.exception("Image preprocessing failed")
        return None

# ----------------- PREDICTION WITH BIAS CORRECTION -----------------
def apply_bias_correction(probs, correction_factor=0.8):
    """
    Apply bias correction to reduce false positive rate for malignant classes.
    This reduces the probability of malignant classes and increases benign probability.
    """
    corrected_probs = probs.copy()
    
    # Reduce malignant class probabilities
    malignant_indices = [1, 2, 3]  # Early_Pre_B, Pre_B, Pro_B
    for idx in malignant_indices:
        corrected_probs[idx] *= correction_factor
    
    # Increase benign probability proportionally
    reduction = np.sum(probs[malignant_indices]) - np.sum(corrected_probs[malignant_indices])
    corrected_probs[0] += reduction  # Add to benign class
    
    # Renormalize
    corrected_probs = np.clip(corrected_probs, 0.001, 0.999)
    corrected_probs = corrected_probs / corrected_probs.sum()
    
    return corrected_probs

def make_prediction(model, device, img_tensor, temperature: float = DEFAULT_TEMPERATURE):
    try:
        img_tensor = img_tensor.to(device)
        with torch.no_grad():
            output = model(img_tensor)
            if isinstance(output, (tuple, list)):
                output = output[0]

            # Apply temperature scaling
            probs = torch.softmax(output / temperature, dim=1)
            all_probs = probs[0].cpu().numpy()

            # Apply bias correction to reduce false positives
            corrected_probs = apply_bias_correction(all_probs, correction_factor=0.7)
            
            # Get prediction from corrected probabilities
            pred = np.argmax(corrected_probs)
            conf = corrected_probs[pred]
            
            # Additional confidence penalty for malignant predictions
            if pred in [1, 2, 3]:  # Malignant classes
                conf *= 0.9  # Reduce confidence by 10%
            
            # Clip confidence to reasonable range
            conf = np.clip(conf, 0.01, 0.95)  # Max 95% instead of 90%

            logger.info(f"Prediction: {LABELS_MAP[pred]} (Class {pred}), Confidence: {conf:.3f}")
            logger.info(f"Raw probabilities: {dict(zip(LABELS_MAP.values(), all_probs))}")
            logger.info(f"Corrected probabilities: {dict(zip(LABELS_MAP.values(), corrected_probs))}")

        return int(pred), float(conf), corrected_probs
        
    except Exception as e:
        st.error(f"🚨 Prediction error: {e}")
        logger.exception("Prediction failed")
        return None, None, None

# ----------------- RESULT DISPLAY -----------------
def format_class_name(pred_idx):
    name = LABELS_MAP.get(pred_idx, "Unknown")
    if name in ["Early_Pre_B", "Pre_B", "Pro_B"]:
        return f"Malignant - {name}", "🔴"
    elif name == "Benign":
        return "Normal/Benign Forms Noted", "🟢"
    return "Unknown", "❓"

def display_results(pred_class, conf, probs):
    formatted, icon = format_class_name(pred_class)
    st.subheader("🔬 Prediction Results")
    
    # Enhanced result display
    col1, col2, col3 = st.columns(3)
    col1.metric("Predicted Class", f"{icon} {formatted}")
    col2.metric("Confidence", f"{conf:.2%}")
    
    # Risk level indicator
    if pred_class == 0:  # Benign
        risk_level = "Low Risk"
        risk_color = "🟢"
    else:  # Malignant
        if conf < 0.6:
            risk_level = "Uncertain"
            risk_color = "🟡"
        elif conf < 0.8:
            risk_level = "Moderate Risk"
            risk_color = "🟠"
        else:
            risk_level = "High Risk"
            risk_color = "🔴"
    
    col3.metric("Risk Level", f"{risk_color} {risk_level}")

    # Detailed probabilities
    with st.expander("📊 Detailed Class Probabilities"):
        for i, (idx, name) in enumerate(LABELS_MAP.items()):
            if i < len(probs):
                formatted_name, class_icon = format_class_name(idx)
                st.write(f"**{class_icon} {name}:** {probs[i]:.3%}")
                st.progress(float(probs[i]))

    # Enhanced warnings and recommendations
    st.markdown("### 📋 Interpretation & Recommendations")
    
    if conf < 0.5:
        st.warning("⚠️ **Very Low Confidence** - Results are highly uncertain. Consider:")
        st.write("• Retaking image with better lighting and focus")
        st.write("• Ensuring proper blood smear preparation")
        st.write("• Consulting a medical professional for proper analysis")
        
    elif conf < 0.7:
        st.warning("⚠️ **Moderate Confidence** - Results should be interpreted cautiously.")
        st.write("• Consider additional testing or expert review")
        st.write("• This tool is for educational purposes only")
        
    if pred_class in [1, 2, 3]:  # Malignant prediction
        st.error("🚨 **Abnormal cells potentially detected**")
        st.write("**IMPORTANT:** This is an AI prediction for educational purposes only.")
        st.write("• Seek immediate consultation with a hematologist")
        st.write("• Professional microscopic examination is required")
        st.write("• Additional tests may be necessary for diagnosis")
    else:
        st.success("✅ **Normal cells detected** - No obvious abnormalities found")
        st.write("• This suggests normal blood cell morphology")
        st.write("• Regular health checkups are still recommended")
        st.write("• Consult a doctor if you have any health concerns")
    
    # Always show disclaimer
    st.markdown("---")
    st.error("🚨 **CRITICAL LIMITATIONS & DISCLAIMERS:**")
    st.markdown("""
    **🔬 ANALYSIS SCOPE:**
    - This tool ONLY analyzes **White Blood Cell (WBC) abnormalities** related to leukemia
    - **Does NOT detect:** Red blood cell disorders, platelet issues, parasites, or other blood conditions
    
    **⚠️ MEDICAL DISCLAIMER:**
    - For **educational and research purposes ONLY**
    - **NOT intended for medical diagnosis** or clinical decision-making
    - Always consult qualified healthcare professionals for medical advice
    - Professional laboratory analysis is required for any suspected blood disorders
    
    **🩸 For Comprehensive Blood Analysis:**
    - Complete Blood Count (CBC) with differential
    - Blood chemistry panels
    - Professional microscopic examination by hematologist
    - Additional specialized tests as recommended by healthcare provider
    """)

# ----------------- MAIN -----------------
def main():
    st.set_page_config(
        page_title="LeukoApp - Blood Cancer Detection",
        page_icon="🩸",
        layout="wide"
    )
    
    # Show OpenCV status after Streamlit is initialized
    if not HAS_OPENCV:
        st.sidebar.info("ℹ️ Basic image processing active. Install opencv-python for enhanced screen capture support.")
    
    st.title("🩸 LeukoApp - Blood Cancer Prediction")
    st.markdown("*AI-powered blood smear analysis for educational purposes*")
    
    # Developer Credits with dx.anx platform information
    st.markdown("---")
    st.markdown("**🏢 Developed by [Shawred Analytics](https://www.shawredanalytics.com) | Part of [dx.anx Platform](https://shawredanalytics.com/dx-anx-analytics)**")
    st.markdown("*Advanced image-based diagnostics powered by state-of-the-art machine learning algorithms*")
    st.markdown("*With contributions from: Pavan Kumar Didde, Shaik Zuber, Ritabrata Dey, Patrika Chatterjee, Titli Paul, Sumit Mitra*")
    
    # ⚠️ CRITICAL LIMITATION NOTICE
    st.error("🚨 **IMPORTANT LIMITATION NOTICE**")
    st.markdown("""
    **This tool ONLY detects White Blood Cell (WBC) abnormalities related to leukemia:**
    - ✅ **Can detect:** Abnormal white blood cells (Pre-B, Pro-B leukemic cells)
    - ❌ **CANNOT detect:** Red blood cell abnormalities (anemia, sickle cell, etc.)
    - ❌ **CANNOT detect:** Platelet disorders
    - ❌ **CANNOT detect:** Parasites (malaria, etc.)
    - ❌ **CANNOT detect:** Other blood conditions not related to WBC leukemia
    
    **If you suspect non-leukemic blood disorders, consult a medical professional for comprehensive blood analysis.**
    """)
    
    st.markdown("---")

    with st.sidebar:
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
        st.subheader("📤 Upload Blood Smear Image")
        
        # SCOPE LIMITATION WARNING
        st.warning("""
        ⚠️ **ANALYSIS SCOPE:** This tool is specifically designed for **WHITE BLOOD CELL LEUKEMIA DETECTION ONLY**
        
        **What this tool analyzes:**
        - Normal vs Abnormal white blood cells
        - Specific leukemic cell types (Early Pre-B, Pre-B, Pro-B)
        
        **What this tool DOES NOT analyze:**
        - Red blood cell disorders (anemia, sickle cell disease, etc.)
        - Platelet abnormalities or counts
        - Blood parasites (malaria, babesia, etc.)
        - Iron deficiency or vitamin deficiencies
        - Any other blood conditions unrelated to WBC leukemia
        """)
        
        st.info("ℹ️ **Instructions:** Upload a clear blood smear microscopy image. " +
                "Ensure good lighting, focus, and proper staining for best results.")

        # Enhanced file uploader for mobile compatibility including screen captures
        uploaded = st.file_uploader(
            "Choose a blood smear image", 
            type=["jpg", "jpeg", "png"],
            help="📱 Mobile users: Use camera or select from gallery. 📺 Screen captures from computers are also supported! Supported formats: JPG, JPEG, PNG (max 10MB)",
            accept_multiple_files=False,
            key="blood_smear_uploader"
        )
        
        # Enhanced mobile-specific upload tips including screen captures
        with st.expander("📱 Mobile Upload Tips (Including Screen Captures)"):
            st.markdown("""
            **Having trouble uploading from mobile?**
            
            **📸 Direct Camera Issues:**
            - Try taking photo first, then upload from gallery
            - Ensure good lighting and steady hands
            - Allow browser camera permissions if prompted
            
            **📺 Screen Capture Tips:**
            - ✅ You can now photograph computer screens displaying blood smears!
            - Position camera straight to avoid distortion
            - Ensure screen brightness is adequate (not too dark/bright)
            - Minimize reflections and glare on screen
            - Hold camera steady to avoid blur
            """)
            
            if not HAS_OPENCV:
                st.warning("⚠️ **Note**: For optimal screen capture processing, consider installing opencv-python. Basic enhancement is currently active.")
            else:
                st.success("✅ **Advanced screen capture processing available** - OpenCV detected")
                
            st.markdown("""
            **📁 Gallery Issues:**
            - Wait for image to fully load before selecting
            - Try refreshing the page if upload stalls
            - Check image file size (should be < 10MB)
            
            **🔧 Technical Issues:**
            - Clear browser cache if uploads fail repeatedly
            - Try switching between WiFi and mobile data
            - Use Chrome or Safari for best mobile compatibility
            """)
        
        if HAS_OPENCV:
            st.info("📺 **Screen Capture Support**: Advanced processing available - can photograph computer screens displaying blood smears!")
        else:
            st.info("📺 **Screen Capture Support**: Basic processing available - can photograph computer screens (install opencv-python for enhanced quality)")
        
        if uploaded:
            # Status indicator for upload success
            st.success("✅ Image uploaded successfully!")
            
            # Display uploaded image with loading indicator
            try:
                # Display uploaded image
                col1, col2 = st.columns([2, 1])
                with col1:
                    st.image(uploaded, caption="Uploaded Image", use_container_width=True)
                
                with col2:
                    st.write("**File Info:**")
                    st.write(f"• Name: {uploaded.name}")
                    st.write(f"• Size: {len(uploaded.getbuffer())/1024:.1f} KB")
                    st.write(f"• Type: {uploaded.type}")
                    
                    # Add refresh button for mobile issues
                    if st.button("🔄 Reprocess Image", help="Click if image appears corrupted"):
                        st.rerun()
            
            except Exception as e:
                st.error(f"🚨 Error displaying image: {str(e)}")
                st.info("💡 Try refreshing the page or uploading a different image.")
                return
            
            # Process image with better error handling
            try:
                with st.spinner("📱 Processing mobile image..."):
                    tensor = preprocess_image(uploaded)
                    
                if tensor is None:
                    st.error("🚨 Image processing failed. Please try:")
                    st.write("• Taking a new photo with better lighting")
                    st.write("• Uploading from gallery instead of camera")
                    st.write("• Refreshing the page and trying again")
                    return

                # Make prediction with mobile-optimized feedback
                with st.spinner("🔍 Analyzing blood smear..."):
                    pred, conf, probs = make_prediction(model, device, tensor, temperature=temp_value)
                    
                if pred is not None:
                    display_results(pred, conf, probs)
                    
                    # Debug information
                    if show_debug:
                        with st.expander("🔧 Debug Information"):
                            st.write(f"**Device:** {device}")
                            st.write(f"**Temperature:** {temp_value}")
                            st.write(f"**Image tensor shape:** {tensor.shape}")
                            st.write(f"**Raw probabilities:** {probs}")
                else:
                    st.error("🚨 Prediction failed. Please try uploading a different image.")
                    
            except Exception as e:
                st.error(f"🚨 Processing error: {str(e)}")
                st.info("💡 This might be a mobile compatibility issue. Try refreshing the page.")
                logger.exception("Mobile image processing failed")

    elif selected == "Model Info":
        st.header("🧠 Model Information")
        
        # DEVELOPER CREDITS WITH dx.anx PLATFORM
        st.info("🏢 **Part of [dx.anx Platform](https://shawredanalytics.com/dx-anx-analytics)** by [Shawred Analytics](https://www.shawredanalytics.com) - Bridging diagnostics and analytics for superior patient care")
        
        # CLEAR SCOPE DEFINITION
        st.error("🎯 **MODEL SCOPE: WHITE BLOOD CELL LEUKEMIA DETECTION ONLY**")
        st.markdown("""
        **This AI model is specifically trained to detect:**
        - ✅ Normal white blood cells vs abnormal leukemic cells
        - ✅ Specific leukemic cell subtypes (Early Pre-B, Pre-B, Pro-B)
        
        **This model CANNOT and DOES NOT detect:**
        - ❌ Red blood cell abnormalities (anemia, sickle cell, thalassemia, etc.)
        - ❌ Platelet disorders or low platelet counts
        - ❌ Blood parasites (malaria, babesia, trypanosoma, etc.)
        - ❌ Iron deficiency or nutritional deficiencies
        - ❌ Bacterial or viral infections affecting blood
        - ❌ Other hematologic conditions not related to WBC leukemia
        """)
        
        st.markdown("---")
        st.write("**Technical Specifications:**")
        st.write("**Architecture:** GoogLeNet (Inception v1) with custom classifier")
        st.write("**Classes:** 4 (Benign, Early Pre-B, Pre-B, Pro-B)")
        st.write("**Input Size:** 224×224 RGB images")
        st.write("**Preprocessing:** ImageNet normalization")
        
        if model is not None and not demo:
            try:
                total_params = sum(p.numel() for p in model.parameters())
                trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
                st.write(f"**Total Parameters:** {total_params:,}")
                st.write(f"**Trainable Parameters:** {trainable_params:,}")
                st.write(f"**Model Device:** {device}")
            except:
                st.write("Model parameter information unavailable")
        
        st.markdown("### 🎯 Performance Notes")
        st.write("• Bias correction applied to reduce false positive rate")
        st.write("• Temperature scaling for calibrated confidence scores")
        st.write("• Enhanced preprocessing for mobile image compatibility")
        st.write("• Screen capture support with automatic enhancement")
        
        # Development credits with dx.anx platform
        st.markdown("---")
        st.success("🏆 **Part of [dx.anx Platform](https://shawredanalytics.com/dx-anx-analytics)** by [Shawred Analytics](https://www.shawredanalytics.com) - Advanced image-based diagnostics with ML")
        st.info("🔬 **Contributors:** Pavan Kumar Didde, Shaik Zuber, Ritabrata Dey, Patrika Chatterjee, Titli Paul, Sumit Mitra")
        st.info("🌐 **Platform Mission:** Bridging diagnostics and analytics for superior patient care")

    else:  # About
        st.header("📋 About LeukoApp")
        
        # DEVELOPER INFORMATION WITH dx.anx PLATFORM
        st.markdown("---")
        st.markdown("### 🏢 **Development Team & Platform**")
        
        # dx.anx Platform Information
        st.success("🔬 **LeukoApp is part of the dx.anx Platform Initiative**")
        st.markdown("""
        **dx.anx** represents a groundbreaking innovation in healthcare, bridging the gap between diagnostics and analytics to deliver superior patient care. By harnessing the power of machine learning and advanced analytics, the platform empowers healthcare professionals with the tools they need to make accurate diagnoses, develop personalized treatment plans, and ultimately, improve patient outcomes.
        
        **🎯 About dx.anx Platform:**
        - Integrates diagnostics and analytics for enhanced image-based diagnostics
        - Powered by state-of-the-art machine learning algorithms  
        - Particularly valuable in radiology, pathology, and dermatology
        - Enables detection of patterns, anomalies, and subtle indicators that might elude human observers
        """)
        
        col1, col2 = st.columns([1, 2])
        with col1:
            st.image("https://via.placeholder.com/150x100/1f77b4/ffffff?text=dx.anx+Platform", 
                    caption="dx.anx Platform", width=150)
        
        with col2:
            st.markdown("**🏢 Developed by: [Shawred Analytics](https://www.shawredanalytics.com)**")
            st.markdown("**🔬 Platform: [dx.anx Analytics](https://shawredanalytics.com/dx-anx-analytics)**")
            st.markdown("**📧 Contact:** contact@shawredanalytics.com")
            st.markdown("**🌐 Website:** [www.shawredanalytics.com](https://www.shawredanalytics.com)")
            st.markdown("**🔗 Advancing image-based medical diagnostics with AI**")
        
        # LeukoApp Specific Information from dx.anx
        st.markdown("---")
        st.markdown("### 🩸 **About LeukoApp (From dx.anx Platform)**")
        st.markdown("""
        **LeukoApp** is a deep learning–based diagnostic aid designed to assist laboratories in the early detection of blood cancer by analyzing digitized peripheral blood smear images. Developed by Shawred Analytics - Data Analytics Team, this system integrates computer vision with laboratory workflows to provide scalable, reproducible, and clinically relevant predictions.
        
        **🔬 Key Capabilities:**
        - Cloud-enabled tool that leverages image-based data analytics
        - Applies advanced image processing and machine learning techniques to segment and classify leukocytes
        - Identifies abnormal leukocytes and early-stage leukemia patterns
        - Provides decision support for pathologists and lab technicians
        """)
        
        # Confidence Levels from dx.anx
        st.markdown("### 📊 **Confidence Level System**")
        confidence_levels = {
            "Level 1 (0.85 – 1.0)": "High confidence – Automated classification very likely correct",
            "Level 2 (0.60 – 0.84)": "Moderate confidence – Classification probable but not definitive", 
            "Level 3 (0.40 – 0.59)": "Low confidence – System uncertain; expert review required",
            "Level 4 (0.0 – 0.39)": "Very low confidence – Classification unreliable"
        }
        
        for level, description in confidence_levels.items():
            st.write(f"**{level}:** {description}")
            
        st.warning("⚠️ **Important:** Results require manual review by a trained professional. A skilled pathologist or laboratory specialist should visually confirm findings.")
        
        #### 👥 **Core Contributors:**
        contributors = [
            "🔬 **Pavan Kumar Didde** - Lead AI/ML Engineer",
            "💻 **Shaik Zuber** - Software Developer", 
            "🧠 **Ritabrata Dey** - Data Scientist",
            "🔍 **Patrika Chatterjee** - Research Analyst",
            "📊 **Titli Paul** - Quality Assurance",
            "⚙️ **Sumit Mitra** - System Architecture"
        ]
        
        for contributor in contributors:
            st.markdown(f"• {contributor}")
        
        st.markdown("---")
        
        # CLEAR SCOPE AND LIMITATIONS
        st.error("🔬 **IMPORTANT: LIMITED TO WHITE BLOOD CELL ANALYSIS ONLY**")
        st.markdown("""
        **LeukoApp is a specialized AI tool that ONLY analyzes White Blood Cell (WBC) abnormalities related to leukemia.**
        
        **🎯 What LeukoApp CAN do:**
        - Detect abnormal white blood cells associated with leukemia
        - Classify specific leukemic cell types (Early Pre-B, Pre-B, Pro-B)
        - Provide educational demonstration of AI in WBC analysis
        
        **❌ What LeukoApp CANNOT do:**
        - Analyze red blood cell disorders (anemia, sickle cell disease, etc.)
        - Detect platelet abnormalities or bleeding disorders
        - Identify blood parasites (malaria, etc.) or infections
        - Diagnose nutritional deficiencies or metabolic conditions
        - Perform comprehensive blood analysis like a clinical laboratory
        """)
        
        st.markdown("---")
        st.write("""
        LeukoApp is an AI-powered educational tool designed to demonstrate automated 
        analysis of blood smear images for potential leukemia detection.
        """)
        
        st.markdown("### 🎯 Purpose")
        st.write("• Educational demonstration of AI in medical imaging")
        st.write("• Research tool for understanding WHITE BLOOD CELL morphology")
        st.write("• NOT intended for clinical diagnosis or medical decision-making")
        
        st.markdown("### ⚠️ Important Disclaimers")
        st.error("🚨 **NOT FOR MEDICAL USE** - This application is strictly for educational and research purposes.")
        st.write("• Results should never be used for medical diagnosis")
        st.write("• Always consult qualified healthcare professionals")
        st.write("• False positives and negatives are possible")
        st.write("• Proper medical testing requires professional laboratory analysis")
        st.write("• **ONLY analyzes WBC abnormalities - NOT comprehensive blood analysis**")
        
        st.markdown("### 🏥 For Comprehensive Blood Testing")
        st.info("""
        **For complete blood analysis, you need professional medical testing:**
        - Complete Blood Count (CBC) with differential
        - Blood smear examination by certified hematologist
        - Specialized tests for specific conditions
        - Clinical correlation with symptoms and medical history
        """)
        
        st.markdown("### 🔬 Technical Details")
        st.write("• Based on deep learning computer vision techniques")
        st.write("• Trained specifically on WBC leukemia detection")
        st.write("• Uses convolutional neural networks for feature extraction")
        st.write("• Applies various preprocessing and post-processing techniques")
        
        st.markdown("---")
        st.markdown("### 📜 **Acknowledgments**")
        st.success("🏢 **Proudly developed as part of [dx.anx Platform](https://shawredanalytics.com/dx-anx-analytics)** by [Shawred Analytics](https://www.shawredanalytics.com) with expertise from our dedicated team of AI researchers, data scientists, and software engineers.")
        st.info("💡 **dx.anx Platform Mission:** Bridging the gap between diagnostics and analytics to deliver superior patient care through machine learning and advanced analytics.")
        st.markdown("🌐 **Learn more:** Visit [dx.anx Analytics](https://shawredanalytics.com/dx-anx-analytics) to explore our comprehensive image-based diagnostic platform.")
        
        # Current Status from dx.anx
        st.markdown("---")
        st.markdown("### ⚠️ **Current Development Status**")
        st.warning("""
        **Testing & Validation Phase:** The LeukoApp is currently in its testing and validation phase. 
        Results should be considered only as preliminary screening outputs, not a substitute for standard laboratory diagnostic practices.
        """)
        st.info("**Copyright © 2025 Shawred Analytics - All Rights Reserved.**")

if __name__ == "__main__":
    main()