# Leuko App - Blood Cancer Screening Tool

🩸 **AI-Powered Blood Smear Analysis for Early Cancer Detection**

## Overview

The Leuko App is an advanced AI-powered tool designed to assist healthcare professionals in screening blood smear images for potential cancerous abnormalities. Using deep learning models, the app analyzes microscopic blood cell images to detect patterns associated with various blood cancers.

## Features

- **Binary Screening**: Classifies blood smears as Normal or Abnormal
- **High Sensitivity**: Optimized threshold (0.50) for early detection
- **Confidence Scoring**: Provides confidence levels for each prediction
- **Medical Disclaimer**: Includes appropriate medical disclaimers for clinical use
- **User-Friendly Interface**: Simple drag-and-drop image upload

## How It Works

1. Upload a blood smear microscopy image
2. The AI model analyzes cellular patterns and morphology
3. Receive classification results with confidence scores
4. Review detailed analysis and recommendations

## Medical Disclaimer

⚠️ **Important**: This tool is for screening purposes only and should not replace professional medical diagnosis. All results should be reviewed by qualified medical practitioners.

## Technology Stack

- **Frontend**: Streamlit
- **AI Model**: PyTorch-based deep learning model
- **Image Processing**: OpenCV, PIL
- **Deployment**: Streamlit Community Cloud

## Usage

Visit the live app: [https://leukoappsaplc.streamlit.app/](https://leukoappsaplc.streamlit.app/)

## Local Development

```bash
# Clone the repository
git clone <repository-url>

# Install dependencies
pip install -r requirements.txt

# Run the app
streamlit run app_binary_screening.py
```

## Model Information

The app uses a specialized binary classification model trained on blood smear images with:
- Optimized sensitivity for abnormal detection
- Balanced accuracy across different cell types
- Confidence estimation for reliable screening

---

*Developed for medical screening assistance - Always consult healthcare professionals for medical decisions.*