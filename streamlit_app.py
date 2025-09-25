#!/usr/bin/env python3
"""
Streamlit Cloud Entry Point for Leuko App
Standard entry point that Streamlit Cloud expects
Updated: 2025-09-25 - Force redeploy
"""

import streamlit as st
import sys
import os

# Add current directory to path for imports
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

# Set page configuration
st.set_page_config(
    page_title="Leuko - Blood Cancer Screening Tool",
    page_icon="🩸",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Import and run the main binary screening app
try:
    from app_binary_screening import main
    
    # Run the main application
    if __name__ == "__main__":
        main()
    else:
        main()
        
except ImportError as e:
    st.error("❌ Failed to import main application module")
    st.error(f"Import error: {str(e)}")
    st.info("Please ensure all required files are available in the deployment.")
    
except Exception as e:
    st.error("❌ Error running the application")
    st.error(f"Runtime error: {str(e)}")
    st.info("Please check the application logs for more details.")