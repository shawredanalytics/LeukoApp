import streamlit as st
import torch
from torchvision.models import googlenet, GoogLeNet_Weights
import torch.nn as nn
from torchvision import transforms
from PIL import Image
from streamlit_option_menu import option_menu
import os
import logging
import sys

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Constants
MODEL_PATH = 'blood_cancer_model.pth'
IMAGE_SIZE = (128, 128)
LABELS_MAP = {0: 'Benign', 1: 'Early_Pre_B', 2: 'Pre_B', 3: 'Pro_B'}

# Page configuration (must be first Streamlit command)
st.set_page_config(
    page_title="LeukoApp - Blood Cancer Prediction",
    page_icon="🩸",
    layout="wide",
    initial_sidebar_state="expanded"
)

@st.cache_resource
def initialize_model():
    """
    Initialize and load the trained model.
    Uses Streamlit's caching to avoid reloading the model on every run.
    """
    try:
        # Create model architecture
        model = googlenet(weights=GoogLeNet_Weights.IMAGENET1K_V1)
        
        # Modify final classifier layer
        model.fc = nn.Sequential(
            nn.Linear(in_features=1024, out_features=512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(in_features=512, out_features=128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(in_features=128, out_features=64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(in_features=64, out_features=4)
        )
        
        # Check if model file exists
        if not os.path.exists(MODEL_PATH):
            st.error(f"🚨 Model file '{MODEL_PATH}' not found. Please ensure the model file is in the correct location.")
            st.info("💡 Make sure 'blood_cancer_model.pth' is in the same directory as this app.")
            return None
            
        # Load trained weights with proper device mapping
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        try:
            # Try loading with current device
            model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
        except Exception as load_error:
            # If that fails, try CPU loading
            st.warning("⚠️ GPU loading failed, falling back to CPU...")
            model.load_state_dict(torch.load(MODEL_PATH, map_location='cpu'))
            device = torch.device('cpu')
        
        model.eval()
        model = model.to(device)
        
        logger.info(f"Model loaded successfully on {device}")
        st.success(f"✅ Model loaded successfully on {device}")
        return model
        
    except FileNotFoundError:
        st.error("🚨 Model file not found. Please check the file path.")
        return None
    except Exception as e:
        st.error(f"🚨 Error loading model: {str(e)}")
        logger.error(f"Model loading failed: {e}")
        return None

def validate_image(uploaded_file):
    """
    Validate uploaded image file.
    """
    try:
        # Check file size (limit to 10MB)
        if uploaded_file.size > 10 * 1024 * 1024:
            st.error("🚨 File size too large. Please upload an image smaller than 10MB.")
            return False
        
        # Check file type
        allowed_types = ['image/jpeg', 'image/jpg', 'image/png']
        if uploaded_file.type not in allowed_types:
            st.error("🚨 Invalid file type. Please upload a JPG, JPEG, or PNG image.")
            return False
        
        return True
    except Exception as e:
        st.error(f"🚨 Error validating image: {str(e)}")
        return False

def preprocess_image(uploaded_file):
    """
    Preprocess the uploaded image for model prediction.
    """
    try:
        # Validate image first
        if not validate_image(uploaded_file):
            return None
        
        # Define transforms
        transform = transforms.Compose([
            transforms.Resize(IMAGE_SIZE),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # ImageNet normalization
        ])
        
        # Open and process image
        img = Image.open(uploaded_file)
        
        # Convert to RGB if necessary (handles RGBA, grayscale, etc.)
        if img.mode != 'RGB':
            img = img.convert('RGB')
            logger.info(f"Converted image from {img.mode} to RGB")
        
        # Apply transformations
        img_tensor = transform(img)
        img_tensor = img_tensor.unsqueeze(0)  # Add batch dimension
        
        return img_tensor
        
    except Exception as e:
        st.error(f"🚨 Error processing image: {str(e)}")
        logger.error(f"Image preprocessing failed: {e}")
        return None

def make_prediction(model, img_tensor):
    """
    Make prediction using the loaded model.
    """
    try:
        if model is None or img_tensor is None:
            return None, None, None
        
        device = next(model.parameters()).device
        img_tensor = img_tensor.to(device)
        
        with torch.no_grad():
            output = model(img_tensor)
            probabilities = torch.nn.functional.softmax(output, dim=1)
            confidence, predicted_class = torch.max(probabilities, 1)
            
            # Get all class probabilities
            all_probabilities = probabilities[0].cpu().numpy()
            
        return predicted_class.item(), confidence.item(), all_probabilities
        
    except RuntimeError as e:
        if "CUDA" in str(e):
            st.error("🚨 CUDA/GPU error. Trying CPU prediction...")
            try:
                # Move to CPU and retry
                model = model.cpu()
                img_tensor = img_tensor.cpu()
                with torch.no_grad():
                    output = model(img_tensor)
                    probabilities = torch.nn.functional.softmax(output, dim=1)
                    confidence, predicted_class = torch.max(probabilities, 1)
                    all_probabilities = probabilities[0].cpu().numpy()
                return predicted_class.item(), confidence.item(), all_probabilities
            except Exception as cpu_error:
                st.error(f"🚨 CPU prediction also failed: {str(cpu_error)}")
                return None, None, None
        else:
            st.error(f"🚨 Runtime error during prediction: {str(e)}")
            return None, None, None
    except Exception as e:
        st.error(f"🚨 Error during prediction: {str(e)}")
        logger.error(f"Prediction failed: {e}")
        return None, None, None

def format_class_name(predicted_class):
    """
    Format the predicted class name for display.
    """
    try:
        class_name = LABELS_MAP.get(predicted_class, "Unknown")
        
        if class_name in ['Early_Pre_B', 'Pre_B', 'Pro_B']:
            return f"Malignant - {class_name}", "🔴"
        elif class_name == "Benign":
            return "Normal/Benign Forms Noted", "🟢"
        else:
            return "Unknown Classification", "❓"
    except Exception as e:
        logger.error(f"Error formatting class name: {e}")
        return "Error in Classification", "❌"

def display_results(predicted_class, confidence_score, all_probabilities):
    """
    Display prediction results in an organized manner.
    """
    try:
        if predicted_class is None or confidence_score is None or all_probabilities is None:
            st.error("🚨 Invalid prediction results")
            return
        
        formatted_class_name, status_icon = format_class_name(predicted_class)
        
        # Main result
        st.subheader("🔬 Prediction Results")
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Predicted Class", f"{status_icon} {formatted_class_name}")
        with col2:
            st.metric("Confidence Score", f"{confidence_score:.2%}")
        
        # Detailed probabilities
        with st.expander("📊 Detailed Probabilities", expanded=False):
            for i, (class_idx, class_name) in enumerate(LABELS_MAP.items()):
                if i < len(all_probabilities):
                    probability = all_probabilities[i]
                    st.write(f"**{class_name}:** {probability:.2%}")
                    st.progress(float(probability))
        
        # Warning for low confidence
        if confidence_score < 0.7:
            st.warning("⚠️ Low confidence prediction. Consider consulting a medical professional for accurate diagnosis.")
        
        # Critical warning for malignant predictions
        if predicted_class in [1, 2, 3]:  # Malignant cases
            st.error("🚨 **Important:** This prediction suggests possible malignant cells. Please consult a medical professional immediately.")
            
    except Exception as e:
        st.error(f"🚨 Error displaying results: {str(e)}")
        logger.error(f"Display results failed: {e}")

def main():
    """
    Main Streamlit application.
    """
    try:
        # Title and description
        st.title("🩸 LeukoApp - Blood Cancer Prediction")
        st.markdown("---")
        
        # Sidebar navigation
        with st.sidebar:
            try:
                selected = option_menu(
                    'Navigation',
                    ['Prediction', 'About Us'],
                    icons=['activity', 'info-circle'],
                    default_index=0,
                    styles={
                        "container": {"padding": "0!important", "background-color": "#fafafa"},
                        "icon": {"color": "orange", "font-size": "25px"},
                        "nav-link": {"font-size": "16px", "text-align": "left", "margin": "0px", "--hover-color": "#eee"},
                        "nav-link-selected": {"background-color": "#02ab21"},
                    }
                )
            except Exception as menu_error:
                st.sidebar.error("Error loading navigation menu")
                logger.error(f"Menu error: {menu_error}")
                selected = "Prediction"  # Default fallback
        
        if selected == "Prediction":
            st.subheader("📤 Upload Medical Image")
            st.write("Upload a blood cell image to predict the type of blood cancer.")
            
            # Important disclaimer
            st.info("⚠️ **Disclaimer:** This tool is for educational purposes only and should not replace professional medical diagnosis.")
            
            # File uploader
            uploaded_file = st.file_uploader(
                "Choose an image file", 
                type=["jpg", "jpeg", "png"],
                help="Supported formats: JPG, JPEG, PNG (Max size: 10MB)"
            )

            if uploaded_file is not None:
                # Display uploaded image
                col1, col2 = st.columns([1, 1])
                
                with col1:
                    try:
                        st.image(uploaded_file, caption="Uploaded Image", use_container_width=True)
                    except Exception as img_error:
                        st.error("🚨 Error displaying uploaded image")
                        logger.error(f"Image display error: {img_error}")
                
                with col2:
                    # Initialize model
                    with st.spinner("Loading model..."):
                        model = initialize_model()
                    
                    if model is None:
                        st.error("🚨 Failed to load model. Please check the console for details.")
                        st.stop()
                    
                    # Process image
                    with st.spinner("Processing image..."):
                        img_tensor = preprocess_image(uploaded_file)
                    
                    if img_tensor is None:
                        st.error("🚨 Failed to process image. Please try a different image.")
                        st.stop()
                    
                    # Make prediction
                    with st.spinner("Making prediction..."):
                        predicted_class, confidence_score, all_probabilities = make_prediction(model, img_tensor)
                    
                    if predicted_class is None:
                        st.error("🚨 Prediction failed. Please try again.")
                        st.stop()
                
                # Display results
                st.markdown("---")
                display_results(predicted_class, confidence_score, all_probabilities)
                
                # Additional information
                st.markdown("---")
                st.subheader("ℹ️ Understanding Your Results")
                
                result_info = {
                    "Benign": {
                        "description": "Normal/benign forms noted - healthy blood cells with no signs of cancer or malignancy.",
                        "details": """
                        **What this means:**
                        - No cancerous cells detected
                        - Normal/benign forms noted in cellular morphology and structure
                        - Healthy immune system function
                        
                        **Next Steps:**
                        - Continue regular health monitoring
                        - Maintain healthy lifestyle habits
                        - Follow routine medical checkups
                        """,
                        "icon": "🟢",
                        "severity": "Normal/Benign Forms Noted"
                    },
                    "Early_Pre_B": {
                        "description": "Early stage B-cell precursor acute lymphoblastic leukemia (B-ALL).",
                        "details": """
                        **What this means:**
                        - Early developmental stage of B-cell leukemia
                        - Abnormal B-cell precursors in bone marrow
                        - Most common type of childhood leukemia
                        
                        **Important Notes:**
                        - Early detection allows for better treatment outcomes
                        - Highly treatable with modern therapy protocols
                        - Requires immediate medical consultation
                        
                        **Typical Symptoms:**
                        - Fatigue, weakness, pale skin
                        - Frequent infections
                        - Easy bruising or bleeding
                        """,
                        "icon": "🟡",
                        "severity": "Early Stage"
                    },
                    "Pre_B": {
                        "description": "Pre-B cell acute lymphoblastic leukemia - intermediate development stage.",
                        "details": """
                        **What this means:**
                        - B-cell precursors at pre-B developmental stage
                        - More advanced than Early Pre-B stage
                        - Characterized by specific cellular markers
                        
                        **Clinical Significance:**
                        - Represents progression in B-cell development
                        - Treatment protocols may differ from early stage
                        - Generally good prognosis with proper treatment
                        
                        **Key Features:**
                        - Cells express specific pre-B markers
                        - May show immunoglobulin gene rearrangements
                        - Responds well to targeted therapy
                        """,
                        "icon": "🟠",
                        "severity": "Intermediate Stage"
                    },
                    "Pro_B": {
                        "description": "Pro-B cell acute lymphoblastic leukemia - earliest B-cell developmental stage.",
                        "details": """
                        **What this means:**
                        - Earliest stage of B-cell development affected
                        - Pro-B cells are the most primitive B-cell precursors
                        - Represents a specific subtype of B-ALL
                        
                        **Clinical Characteristics:**
                        - Often associated with specific genetic abnormalities
                        - May require intensive treatment protocols
                        - Prognosis varies based on specific genetic markers
                        
                        **Treatment Considerations:**
                        - May need specialized therapeutic approaches
                        - Close monitoring during treatment essential
                        - Regular genetic testing recommended
                        """,
                        "icon": "🔴",
                        "severity": "Early Developmental Stage"
                    }
                }
                
                # Create tabs for better organization
                try:
                    tab1, tab2, tab3 = st.tabs(["📋 Quick Reference", "🔬 Detailed Information", "⚠️ Important Notes"])
                    
                    with tab1:
                        # Quick reference cards
                        for condition, info in result_info.items():
                            with st.container():
                                col1, col2, col3 = st.columns([1, 6, 2])
                                with col1:
                                    st.markdown(f"## {info['icon']}")
                                with col2:
                                    st.markdown(f"**{condition}**")
                                    st.markdown(f"*{info['description']}*")
                                with col3:
                                    if condition == "Benign":
                                        st.success(info['severity'])
                                    elif "Early" in condition:
                                        st.warning(info['severity'])
                                    else:
                                        st.error(info['severity'])
                                st.markdown("---")
                    
                    with tab2:
                        # Detailed expandable sections
                        for condition, info in result_info.items():
                            with st.expander(f"{info['icon']} {condition} - {info['severity']}", expanded=False):
                                st.markdown(info['details'])
                    
                    with tab3:
                        st.markdown("""
                        ### 🩺 Medical Disclaimer
                        - **This tool is for educational purposes only**
                        - **Always consult healthcare professionals for diagnosis**
                        - **Do not use as sole basis for medical decisions**
                        
                        ### 🆘 When to Seek Immediate Medical Care
                        - Persistent fatigue or weakness
                        - Unexplained fever or infections
                        - Easy bruising or bleeding
                        - Swollen lymph nodes
                        - Bone or joint pain
                        
                        ### 📞 Emergency Contacts
                        - Contact your primary care physician
                        - Visit nearest emergency room if symptoms are severe
                        - Call emergency services if experiencing severe symptoms
                        """)
                except Exception as tab_error:
                    st.error("Error loading information tabs")
                    logger.error(f"Tab error: {tab_error}")

        elif selected == "About Us":
            st.markdown("<h2 style='text-align: center;'>📋 ABOUT</h2>", unsafe_allow_html=True)
            st.markdown("---")
            
            # Project description
            st.markdown("""
            ### 🎯 Project Overview
            LeukoApp is an AI-powered diagnostic tool designed to assist in the early detection of blood cancer types from microscopic blood cell images. 
            This application uses deep learning techniques to analyze uploaded images and provide predictions with confidence scores.
            
            ### 🔬 Technology Stack
            - **Deep Learning Framework:** PyTorch
            - **Model Architecture:** GoogLeNet (Inception v1)
            - **Web Framework:** Streamlit
            - **Image Processing:** PIL, torchvision
            
            ### ⚠️ Important Notes
            - This tool is for educational and research purposes only
            - Always consult with healthcare professionals for medical diagnosis
            - The model predictions should not be used as the sole basis for medical decisions
            """)
            
            st.markdown("---")
            
            # Team information
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("### 🏢 Ideation Partner")
                st.markdown("**Shawred Analytics Pvt Ltd**")
            
            with col2:
                st.markdown("### 👨‍💻 Development Team")
                st.markdown("- **Pavan Kumar Didde**")
                st.markdown("- **Shaik Zuber**")
            
            st.markdown("---")
            st.markdown("### 📞 Contact & Support")
            st.markdown("For questions, support, or collaboration opportunities, please reach out to the development team.")
    
    except Exception as e:
        st.error(f"🚨 Application error: {str(e)}")
        logger.error(f"Main application error: {e}")

if __name__ == '__main__':
    main()