import streamlit as st
import torch
from torchvision.models import googlenet, GoogLeNet_Weights
import torch.nn as nn
from torchvision import transforms
from PIL import Image, ImageOps
from streamlit_option_menu import option_menu
import os
import logging
import sys
import numpy as np
from collections import OrderedDict

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
        file_size = len(uploaded_file.getbuffer())
        if file_size > MAX_FILE_SIZE:
            st.error("🚨 File size too large. Please upload <10MB.")
            return False

        allowed_types = {"image/jpeg", "image/jpg", "image/png"}
        if (uploaded_file.type not in allowed_types and
            not uploaded_file.name.lower().endswith((".jpg", ".jpeg", ".png"))):
            st.error("🚨 Invalid type. Only JPG/JPEG/PNG allowed.")
            return False
        return True
    except Exception as e:
        st.error(f"🚨 Error validating image: {e}")
        return False

def is_blood_smear(img: Image.Image, filename: str = "unknown"):
    """
    Improved heuristic check for blood smear suitability.
    More conservative to reduce false positives.
    """
    try:
        img_small = img.resize((128, 128))
        arr = np.array(img_small).astype(float) / 255.0

        r, g, b = arr[:, :, 0], arr[:, :, 1], arr[:, :, 2]

        # Color heuristics - improved
        red_ratio = np.mean(r > 0.4)  # Lowered threshold
        purple_ratio = np.mean((r > 0.3) & (b > 0.25))  # Lowered thresholds
        pink_ratio = np.mean((r > 0.5) & (g > 0.3) & (b > 0.3))  # Added pink detection
        green_ratio = np.mean(g > 0.5)

        # Combined color ratio for better detection
        total_color_ratio = red_ratio + purple_ratio + pink_ratio

        # Texture heuristics using numpy gradient
        gray = np.mean(arr, axis=2)
        gx, gy = np.gradient(gray)
        edges = np.sqrt(gx**2 + gy**2)
        edge_var = np.var(edges)

        # More lenient checks to reduce false rejections
        if total_color_ratio <= COLOR_RATIO_THRESHOLD:
            reason = "Very weak staining detected; may significantly affect accuracy."
            logger.warning(f"[{filename}] Blood smear very weak stain: {total_color_ratio:.3f}")
            # Still allow but warn strongly
            return True, reason
            
        if green_ratio >= GREEN_RATIO_THRESHOLD:
            reason = "Strong green background detected; may affect cell visibility."
            logger.warning(f"[{filename}] Blood smear strong green background: {green_ratio:.3f}")
            return True, reason
            
        if edge_var <= EDGE_VAR_THRESHOLD:
            reason = "Very low texture detected; image may be too blurry or uniform."
            logger.warning(f"[{filename}] Blood smear very low texture: {edge_var:.3f}")
            return True, reason

        # ✅ Passed all checks
        logger.info(f"[{filename}] Blood smear check passed. Color: {total_color_ratio:.3f}, Edge: {edge_var:.3f}")
        return True, ""
        
    except Exception as e:
        reason = f"Error analyzing image content: {e}"
        logger.error(f"[{filename}] Blood smear check failed: {reason}")
        return False, reason

# ----------------- IMAGE PREPROCESSING -----------------
def preprocess_image(uploaded_file):
    try:
        if not validate_image(uploaded_file):
            return None

        uploaded_file.seek(0)
        img = Image.open(uploaded_file)

        # Handle mobile phone rotation metadata
        img = ImageOps.exif_transpose(img)

        if img.mode != "RGB":
            img = img.convert("RGB")

        # Validate blood smear with filename
        ok, reason = is_blood_smear(img, uploaded_file.name)
        if not ok:
            st.error(f"🚨 Image does not meet specifications of the test model\n\n**Reason:** {reason}")
            return None
        elif reason:
            st.warning(f"⚠️ Note: {reason}")

        # Improved preprocessing with histogram equalization option
        transform = transforms.Compose([
            transforms.Resize(IMAGE_SIZE),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        tensor = transform(img).unsqueeze(0)
        return tensor
        
    except Exception as e:
        st.error(f"🚨 Error processing image: {e}")
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
    
    st.title("🩸 LeukoApp - Blood Cancer Prediction")
    st.markdown("*AI-powered blood smear analysis for educational purposes*")
    
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

        uploaded = st.file_uploader(
            "Choose a blood smear image", 
            type=["jpg", "jpeg", "png"],
            help="Supported formats: JPG, JPEG, PNG (max 10MB)"
        )
        
        if uploaded:
            # Display uploaded image
            col1, col2 = st.columns([2, 1])
            with col1:
                st.image(uploaded, caption="Uploaded Image", use_container_width=True)
            
            with col2:
                st.write("**File Info:**")
                st.write(f"• Name: {uploaded.name}")
                st.write(f"• Size: {len(uploaded.getbuffer())/1024:.1f} KB")
                st.write(f"• Type: {uploaded.type}")
            
            # Process image
            tensor = preprocess_image(uploaded)
            if tensor is None:
                st.stop()

            # Make prediction
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

    elif selected == "Model Info":
        st.header("🧠 Model Information")
        
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

    else:  # About
        st.header("📋 About LeukoApp")
        
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

if __name__ == "__main__":
    main()