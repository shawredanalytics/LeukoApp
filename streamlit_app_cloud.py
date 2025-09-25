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
            
            # Import and run the original sophisticated app in demo mode
            try:
                from app_binary_screening import main as app_main
                app_main()
            except ImportError:
                st.error("❌ Original app module not found. Running basic interface.")
                st.markdown("**Basic Interface**: Please upload an image to test the interface.")
                uploaded_file = st.file_uploader("Choose an image file", type=['png', 'jpg', 'jpeg', 'svg'])
                if uploaded_file:
                    st.success("✅ File uploaded successfully!")
                    st.info("In full deployment, this would be analyzed by the AI model.")
        else:
            # Import the main application with full functionality
            from app_binary_screening import main as app_main
            app_main()
        
    except ImportError as e:
        st.error(f"❌ Failed to import application: {str(e)}")
        st.info("Please ensure all required files are present and dependencies are installed.")
        
        # Fallback to basic interface
        st.title("🩸 Leuko - Cancer Screening Tool")
        st.markdown("### Application Loading Error")
        st.markdown("The application encountered an import error. Please check the deployment configuration.")
        
    except Exception as e:
        st.error(f"❌ Application error: {str(e)}")
        st.info("Please check the application logs for more details.")
        
        # Show basic error interface
        st.title("🩸 Leuko - Cancer Screening Tool")
        st.markdown("### Runtime Error")
        st.markdown("The application encountered a runtime error. Please try refreshing the page.")

if __name__ == "__main__":
    main()