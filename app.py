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
import numpy as np
from collections import OrderedDict

# ----------------- LOGGING SETUP -----------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),            # Console
        logging.FileHandler("uploads.log", mode="a", encoding="utf-8")  # File
    ]
)
logger = logging.getLogger(__name__)

# ----------------- CONSTANTS -----------------
MODEL_PATH = "blood_cancer_model.pth"
IMAGE_SIZE = (224, 224)  # Set to (128,128) if your model was trained that way
LABELS_MAP = {0: "Benign", 1: "Early_Pre_B", 2: "Pre_B", 3: "Pro_B"}
MAX_FILE_SIZE = 10 * 1024 * 1024  # 10 MB
ALLOW_DEMO_MODE = False  # If True, runs demo mode if model weights missing

# Configurable heuristics
EDGE_VAR_THRESHOLD = 1500
GREEN_RATIO_THRESHOLD = 0.35
COLOR_RATIO_THRESHOLD = 0.25
DEFAULT_TEMPERATURE = 2.0

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

        state = torch.load(model_path, map_location=device)
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
    Stricter heuristic check:
    - Color dominance: red/purple relative to green.
    - Texture detail: high variance in edges (cell nuclei visible).
    Returns (bool, reason).
    """
    try:
        img_small = img.resize((128, 128))
        arr = np.array(img_small).astype(float) / 255.0

        r, g, b = arr[:, :, 0], arr[:, :, 1], arr[:, :, 2]

        # Color heuristics
        red_ratio = np.mean(r > 0.5)
        purple_ratio = np.mean((r > 0.4) & (b > 0.3))
        green_ratio = np.mean(g > 0.5)

        # Texture heuristics using numpy gradient instead of cv2
        gray = np.mean(arr, axis=2)  # grayscale approximation
        gx, gy = np.gradient(gray)
        edges = np.sqrt(gx**2 + gy**2)
        edge_var = np.var(edges)

        # Checks with reasons
        if (red_ratio + purple_ratio) <= COLOR_RATIO_THRESHOLD:
            reason = "Image colors inconsistent with stained blood smears (insufficient red/purple tones)."
            logger.warning(f"[{filename}] Blood smear check failed: {reason}")
            return False, reason
        if green_ratio >= GREEN_RATIO_THRESHOLD:
            reason = "Image background too green; not typical for blood smear slides."
            logger.warning(f"[{filename}] Blood smear check failed: {reason}")
            return False, reason
        if edge_var <= EDGE_VAR_THRESHOLD:
            reason = "Image lacks sufficient cellular texture (too smooth for blood smear)."
            logger.warning(f"[{filename}] Blood smear check failed: {reason}")
            return False, reason

        # ✅ Passed all checks
        logger.info(f"[{filename}] Blood smear check passed.")
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
        if img.mode != "RGB":
            img = img.convert("RGB")

        # ✅ Extra validation for blood smear with filename
        ok, reason = is_blood_smear(img, uploaded_file.name)
        if not ok:
            st.error(f"🚨 Image does not meet specifications of the test model\n\n**Reason:** {reason}")
            return None

        transform = transforms.Compose(
            [
                transforms.Resize(IMAGE_SIZE),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )
        tensor = transform(img).unsqueeze(0)
        return tensor
    except Exception as e:
        st.error(f"🚨 Error processing image: {e}")
        return None

# ----------------- PREDICTION -----------------
def make_prediction(model, device, img_tensor, temperature: float = DEFAULT_TEMPERATURE):
    """
    Make prediction with calibrated probabilities using temperature scaling.
    Default temperature rationalizes confidence.
    """
    try:
        img_tensor = img_tensor.to(device)
        with torch.no_grad():
            output = model(img_tensor)
            if isinstance(output, (tuple, list)):
                output = output[0]

            # 🔹 Apply temperature scaling
            probs = torch.softmax(output / temperature, dim=1)

            # Predicted class
            conf, pred = torch.max(probs, dim=1)

            # Convert to numpy
            all_probs = probs[0].cpu().numpy()

            # Normalize distribution
            all_probs = all_probs / all_probs.sum()

            # Confidence capped at 95%
            conf = float(all_probs[pred.item()])
            conf = min(conf, 0.95)

        return int(pred.item()), conf, all_probs
    except Exception as e:
        st.error(f"🚨 Prediction error: {e}")
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
    c1, c2 = st.columns(2)
    c1.metric("Class", f"{icon} {formatted}")
    c2.metric("Confidence", f"{conf:.2%}")

    with st.expander("📊 Class Probabilities"):
        for i, (idx, name) in enumerate(LABELS_MAP.items()):
            if i < len(probs):
                st.write(f"**{name}:** {probs[i]:.2%}")
                st.progress(float(probs[i]))

    if conf < 0.7:
        st.warning("⚠️ Low confidence — please consult a medical professional.")
    if pred_class in [1, 2, 3]:
        st.error("🚨 Malignant likelihood detected. Seek immediate medical advice.")

# ----------------- MAIN -----------------
def main():
    st.title("🩸 LeukoApp - Blood Cancer Prediction")
    st.markdown("---")

    with st.sidebar:
        selected = option_menu(
            "Navigation",
            ["Prediction", "About Us"],
            icons=["activity", "info-circle"],
            default_index=0,
        )

        # Allow user to tweak temperature scaling if needed
        temp_value = st.slider("Temperature Scaling", 0.5, 5.0, DEFAULT_TEMPERATURE, 0.1)

    model, device, demo = initialize_model()
    if model is None:
        st.stop()

    if selected == "Prediction":
        st.subheader("📤 Upload Blood Smear Image")
        st.info("⚠️ Educational tool only — not a medical device.")

        uploaded = st.file_uploader("Choose a blood smear image", type=["jpg", "jpeg", "png"])
        if uploaded:
            st.image(uploaded, caption="Uploaded Image", use_container_width=True)
            tensor = preprocess_image(uploaded)
            if tensor is None:
                st.stop()

            with st.spinner("Making prediction..."):
                pred, conf, probs = make_prediction(model, device, tensor, temperature=temp_value)
            if pred is not None:
                display_results(pred, conf, probs)

    else:
        st.header("📋 About")
        st.write("LeukoApp helps demonstrate AI-based analysis of blood smear images.")
        st.write("⚠️ For research & education only. Not for clinical use.")

if __name__ == "__main__":
    main()