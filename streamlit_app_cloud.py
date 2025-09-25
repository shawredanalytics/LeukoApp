#!/usr/bin/env python3
"""
Streamlit Cloud Entry Point for Leuko App
Cloud-compatible version with fallback for missing model files
Updated: 2025-01-27 - Enhanced cloud deployment with original model design
"""

import streamlit as st
import os

# Set page configuration
st.set_page_config(
    page_title="Leuko - Cancer Screening Tool",
    page_icon="🩸",
    layout="wide",
    initial_sidebar_state="expanded"
)

def main():
    try:
        # Display loading message
        st.title("🩸 Leuko - Cancer Screening Tool")
        st.markdown("### AI-Powered Blood Smear Analysis")
        
        # Check if we have the required model files
        model_files = [
            "blood_smear_screening_model.pth",
            "blood_cancer_model.pth", 
            "best_binary_model.pth"
        ]
        
        missing_models = [f for f in model_files if not os.path.exists(f)]
        
        if missing_models:
            st.warning(f"⚠️ **Cloud Deployment Mode**: Some model files are not available: {', '.join(missing_models)}")
            st.info("💡 Running in demonstration mode with simplified functionality.")
            
            # Try to import PyTorch-dependent modules
            try:
                import torch
                torch_available = True
            except ImportError:
                torch_available = False
                st.warning("⚠️ **PyTorch not available in cloud environment**")
            
            # Import and run the original sophisticated app in demo mode
            if torch_available:
                try:
                    from app_binary_screening import main as app_main
                    app_main()
                except ImportError:
                    st.error("❌ Original app module not found. Running basic interface.")
                    show_basic_interface()
            else:
                # Show PyTorch-free interface
                show_pytorch_free_interface()
        else:
            # Import the main application with full functionality
            try:
                from app_binary_screening import main as app_main
                app_main()
            except ImportError as e:
                st.error(f"❌ Failed to import full application: {str(e)}")
                show_pytorch_free_interface()
        
    except ImportError as e:
        st.error(f"❌ Failed to import application: {str(e)}")
        st.info("Please ensure all required files are present and dependencies are installed.")
        show_pytorch_free_interface()
        
    except Exception as e:
        st.error(f"❌ Application error: {str(e)}")
        st.info("Please check the application logs for more details.")
        show_pytorch_free_interface()

def show_basic_interface():
    """Basic interface when original app is not available"""
    st.markdown("**Basic Interface**: Please upload an image to test the interface.")
    uploaded_file = st.file_uploader("Choose an image file", type=['png', 'jpg', 'jpeg', 'svg'])
    if uploaded_file:
        st.success("✅ File uploaded successfully!")
        st.info("In full deployment, this would be analyzed by the AI model.")

def show_pytorch_free_interface():
    """PyTorch-free interface for cloud deployment"""
    # Sidebar
    with st.sidebar:
        st.header("📋 Instructions")
        st.markdown("""
        1. **Upload** a blood smear image
        2. **View** image preview
        3. **Get** demonstration analysis
        4. **Interpret** with medical expertise
        """)
        
        st.header("⚠️ Medical Disclaimer")
        st.markdown("""
        This tool is for **research and educational purposes only**.
        Always consult qualified medical professionals for diagnosis.
        """)
        
        st.header("🔧 Cloud Mode")
        st.info("Running in cloud demonstration mode. Full AI analysis requires local deployment with PyTorch.")
    
    # Main interface
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("📤 Upload Blood Smear Image")
        uploaded_file = st.file_uploader(
            "Choose an image file",
            type=['png', 'jpg', 'jpeg', 'svg'],
            help="Upload a blood smear microscopy image"
        )
        
        if uploaded_file is not None:
            # Display uploaded image
            try:
                if uploaded_file.name.endswith('.svg'):
                    st.markdown("**SVG Preview:**")
                    st.markdown(uploaded_file.getvalue().decode(), unsafe_allow_html=True)
                else:
                    from PIL import Image
                    image = Image.open(uploaded_file)
                    st.image(image, caption="Uploaded Image", use_column_width=True)
            except Exception as e:
                st.error(f"Error displaying image: {str(e)}")
    
    with col2:
        st.subheader("🔬 Analysis Results")
        
        if uploaded_file is not None:
            with st.spinner("🧠 Analyzing image (demo mode)..."):
                try:
                    import time
                    time.sleep(2)  # Simulate processing time
                    
                    # Demo predictions without PyTorch
                    import random
                    confidence = random.uniform(0.6, 0.95)
                    prediction = random.choice(['Normal', 'Leukemic'])
                    
                    # Display results
                    if prediction == "Normal":
                        st.success(f"✅ **Classification: Normal**")
                        st.metric("Confidence", f"{confidence:.1%}")
                        st.info("The blood smear appears to show normal cellular characteristics.")
                    else:
                        st.error(f"⚠️ **Classification: Leukemic**")
                        st.metric("Confidence", f"{confidence:.1%}")
                        st.warning("The blood smear shows characteristics that may indicate leukemia. Please consult a medical professional immediately.")
                    
                    # Additional info
                    st.markdown("---")
                    st.markdown("**🔍 Analysis Details:**")
                    st.markdown("- Model: Demo version (cloud deployment)")
                    st.markdown("- Processing: Simulated analysis")
                    st.markdown("- Note: For full AI analysis, use local deployment")
                    
                except Exception as e:
                    st.error(f"❌ Analysis failed: {str(e)}")
        else:
            st.info("👆 Please upload an image to begin analysis")
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: #666;'>
        <p>🏥 <strong>Leuko App</strong> - Advanced Blood Cancer Screening Tool</p>
        <p>Powered by Deep Learning • For Research & Educational Use</p>
        <p><em>Cloud Demo Mode - Full functionality available in local deployment</em></p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()