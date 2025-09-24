import streamlit as st
import subprocess
import sys
import os

# Redirect to the main binary screening app
st.set_page_config(
    page_title="Blood Smear Screening - Redirecting...",
    page_icon="🔬",
    layout="wide"
)

st.title("🔬 Blood Smear Screening App")
st.info("Redirecting to the main application...")

# Import and run the main app
try:
    # Import the main app module
    import importlib.util
    spec = importlib.util.spec_from_file_location("app_binary_screening", "app_binary_screening.py")
    app_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(app_module)
except Exception as e:
    st.error(f"Error loading main application: {str(e)}")
    st.info("Please ensure app_binary_screening.py is available in the deployment.")
