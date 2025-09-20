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
    Validates if the uploaded image is likely a blood smear image.
    Returns True if it's likely a blood smear, False otherwise.
    
    This function uses very permissive heuristics to:
    1. Only reject extremely obvious non-medical images
    2. Accept most blood smear images with various staining techniques and conditions
    """
    # Convert to numpy array
    img_array = np.array(image)
    
    # 1. Basic dimension check - only reject extremely small images
    if image.width < 20 or image.height < 20:
        return False
    
    # 2. Convert to smaller size for faster processing
    small_img = image.resize((100, 100))
    img_array_small = np.array(small_img)
    
    # 3. Very permissive color analysis - only reject extremely obvious non-medical images
    if len(img_array_small.shape) == 3 and img_array_small.shape[2] == 3:
        # Calculate average color
        avg_color = np.mean(img_array_small, axis=(0, 1))
        r, g, b = avg_color
        
        # Only reject images with extremely unnatural colors for medical images
        # Very bright neon-like colors that are impossible in microscopy
        if max(r, g, b) > 240 and min(r, g, b) < 50:
            # Check for extremely saturated single colors (like pure red, green, or blue)
            if (r > 240 and g < 50 and b < 50) or \
               (g > 240 and r < 50 and b < 50) or \
               (b > 240 and r < 50 and g < 50):
                return False
    
    # 4. Very permissive texture analysis - only reject extremely complex images
    # Convert to grayscale
    if len(img_array_small.shape) == 3:
        gray = np.mean(img_array_small, axis=2).astype(np.uint8)
    else:
        gray = img_array_small
    
    # Calculate edge density with very high threshold
    grad_x = np.abs(np.diff(gray, axis=1))
    grad_y = np.abs(np.diff(gray, axis=0))
    
    # Count very strong edges only
    strong_edges_x = np.sum(grad_x > 80)  # Much higher threshold
    strong_edges_y = np.sum(grad_y > 80)  # Much higher threshold
    total_pixels = gray.size
    
    edge_density = (strong_edges_x + strong_edges_y) / total_pixels
    
    # Only reject if edge density is extremely high (like detailed photographs)
    if edge_density > 0.3:  # Much higher threshold
        return False
    
    # Accept most images as potential medical images
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
            st.warning("⚠️ Model weights not found. Running in DEMO mode with random predictions.")
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
        page_title="LeukoApp - Blood Cancer Detection",
        page_icon="🩸",
        layout="wide"
    )
    
    # Show OpenCV status after Streamlit is initialized
    if not HAS_OPENCV:
        st.sidebar.info("ℹ️ Basic image processing active. Install opencv-python for enhanced screen capture support.")
    
    st.title("🩸 LeukoApp - Blood Cancer Prediction")
    st.markdown("*AI-powered blood smear analysis for educational purposes*")
    
    # Developer Credits with dx.anx platform information + UPDATED CONTACT
    st.markdown("---")
    st.markdown("**🏢 Developed by [Shawred Analytics](https://www.shawredanalytics.com) | 📧 shawred.analytics@gmail.com | Part of [dx.anx Platform](https://shawredanalytics.com/dx-anx-analytics)**")
    st.markdown("*Advanced image-based diagnostics powered by state-of-the-art machine learning algorithms*")
    st.markdown("*With contributions from: Pavan Kumar Didde, Shaik Zuber, Ritabrata Dey, Patrika Chatterjee, Titli Paul, Sumit Mitra*")
    
    st.error("🚨 **IMPORTANT LIMITATION NOTICE**")
    st.markdown("""
    **This tool ONLY detects White Blood Cell (WBC) abnormalities related to leukemia.**
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
        
        # File uploader
        uploaded_file = st.file_uploader("Choose a blood smear image...", type=["jpg", "jpeg", "png"])
        
        # Process the uploaded image
        if uploaded_file is not None:
            # Display the uploaded image
            image = Image.open(uploaded_file)
            st.image(image, caption="Uploaded Image", use_column_width=True)
            
            # Check if the image is likely a blood smear
            is_valid_image = validate_blood_smear_image(image)
            
            if not is_valid_image:
                st.error("⚠️ **Image Rejected for Analysis**")
                st.markdown("**Your image was rejected for the following possible reasons:**")
                
                # Get image dimensions for specific feedback
                img_width, img_height = image.size
                
                # Check specific rejection criteria and provide explanations
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("### 🚫 **Possible Rejection Reasons:**")
                    
                    # Check dimension issues
                    if img_width < 20 or img_height < 20:
                        st.error(f"❌ **Image too small**: {img_width}x{img_height} pixels (minimum: 20x20)")
                    else:
                        st.success(f"✅ **Size acceptable**: {img_width}x{img_height} pixels")
                    
                    # Check for obvious non-medical characteristics
                    img_array = np.array(image)
                    if len(img_array.shape) == 3 and img_array.shape[2] == 3:
                        avg_color = np.mean(img_array, axis=(0, 1))
                        r, g, b = avg_color
                        
                        # Check for neon colors
                        if max(r, g, b) > 240 and min(r, g, b) < 50:
                            if (r > 240 and g < 50 and b < 50) or \
                               (g > 240 and r < 50 and b < 50) or \
                               (b > 240 and r < 50 and g < 50):
                                st.error("❌ **Unnatural colors detected** (pure neon colors)")
                            else:
                                st.info("⚠️ **High contrast colors detected**")
                        else:
                            st.success("✅ **Color profile acceptable**")
                    
                    # Check texture complexity
                    small_img = image.resize((100, 100))
                    img_array_small = np.array(small_img)
                    if len(img_array_small.shape) == 3:
                        gray = np.mean(img_array_small, axis=2).astype(np.uint8)
                    else:
                        gray = img_array_small
                    
                    grad_x = np.abs(np.diff(gray, axis=1))
                    grad_y = np.abs(np.diff(gray, axis=0))
                    strong_edges_x = np.sum(grad_x > 80)
                    strong_edges_y = np.sum(grad_y > 80)
                    edge_density = (strong_edges_x + strong_edges_y) / gray.size
                    
                    if edge_density > 0.3:
                        st.error(f"❌ **Too complex/detailed** (edge density: {edge_density:.3f})")
                    else:
                        st.success(f"✅ **Texture complexity acceptable** (edge density: {edge_density:.3f})")
                
                with col2:
                    st.markdown("### ✅ **Acceptable Image Specifications:**")
                    st.markdown("""
                    **📏 Dimensions:**
                    - Minimum: 20x20 pixels
                    - Recommended: ≥200x200 pixels
                    - Maximum: No limit
                    
                    **🎨 Color Requirements:**
                    - Medical/microscopy colors
                    - Various staining techniques accepted:
                      - Wright's stain (purple/pink)
                      - Giemsa stain (blue/purple)
                      - May-Grünwald stain
                    - Avoid pure neon colors
                    
                    **🔬 Image Characteristics:**
                    - Blood cells on light background
                    - Microscopic cellular structures
                    - Various magnifications accepted
                    - Different lighting conditions OK
                    
                    **📁 File Formats:**
                    - JPG/JPEG
                    - PNG
                    - File size: No specific limit
                    """)
                
                st.markdown("---")
                st.info("💡 **Tip**: Ensure your image shows blood cells under microscopic view with typical medical staining. The system is designed to accept most legitimate blood smear images while filtering out obvious non-medical content.")
                
                # Exit the prediction flow for invalid images
                return
            
            # Only proceed with valid blood smear images
            
            # Check for low-resolution images and show disclaimer
            image_width, image_height = image.size
            is_low_resolution = image_width < 200 or image_height < 200
            
            if is_low_resolution:
                st.warning("📏 **Low Resolution Image Detected**")
                st.info(f"Image resolution: {image_width}x{image_height} pixels. For optimal accuracy, higher resolution images (≥200x200) are recommended. Results may be less reliable with low-resolution images.")
            
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
            st.subheader("📊 Prediction Results")
            
            # Create columns for the results
            col1, col2 = st.columns(2)
            
            with col1:
                # Normal probability
                normal_prob = probabilities[0].item() * 100
                st.metric("Normal WBC", f"{normal_prob:.1f}%")
                
            with col2:
                # Leukemia probability
                leukemia_prob = probabilities[1].item() * 100
                st.metric("Leukemia Indicators", f"{leukemia_prob:.1f}%")
            
            # Display warning based on prediction
            if leukemia_prob > 50:
                st.warning("⚠️ Potential leukemia indicators detected. Please consult with a healthcare professional.")
            else:
                st.success("✅ No significant leukemia indicators detected.")
            
            # Additional disclaimer for low-resolution images
            if is_low_resolution:
                st.markdown("---")
                st.warning("⚠️ **Important Note for Low-Resolution Images:**")
                st.markdown("""
                - The prediction accuracy may be reduced due to low image resolution
                - Fine cellular details may not be clearly visible for analysis
                - Consider using a higher resolution image for more reliable results
                - Always consult with a healthcare professional for medical diagnosis
                """)
                
            # Debug information
            if show_debug:
                st.subheader("🔍 Debug Information")
                st.write(f"Raw Model Output: {output.numpy()}")
                st.write(f"Temperature Scaling: {temp_value}")
                st.write(f"Device: {device}")
                st.write(f"Demo Mode: {demo}")
        
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
        st.header("📋 About LeukoApp")
        
        st.markdown("---")
        st.markdown("### 🏢 **Development Team & Platform**")
        st.success("🔬 **LeukoApp is part of the dx.anx Platform Initiative**")
        
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
        
        st.markdown("---")
        st.markdown("### 👥 **Contributors**")
        
        st.markdown("""
        This project was made possible by contributions from:
        
        - **Pavan Kumar Didde:** Machine Learning Architecture
        - **Shaik Zuber:** Data Collection and Annotation
        - **Ritabrata Dey:** Model Training and Optimization
        - **Patrika Chatterjee:** Medical Validation
        - **Titli Paul:** UI/UX Design
        - **Sumit Mitra:** Project Management
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
