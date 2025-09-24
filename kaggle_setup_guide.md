# 🩸 Blood Cancer Model Training with Kaggle Datasets

This guide explains how to use Kaggle datasets to train your blood cancer detection model.

## 🔧 Setup Instructions

### 1. Install Training Dependencies

```bash
pip install -r requirements_training.txt
```

### 2. Setup Kaggle API

#### Step 1: Create Kaggle Account
- Go to [kaggle.com](https://www.kaggle.com) and create an account
- Verify your phone number (required for API access)

#### Step 2: Get API Credentials
1. Go to your Kaggle account settings: https://www.kaggle.com/account
2. Scroll down to "API" section
3. Click "Create New API Token"
4. Download the `kaggle.json` file

#### Step 3: Setup API Credentials
**Windows:**
```bash
# Create .kaggle directory in your user folder
mkdir %USERPROFILE%\.kaggle

# Copy kaggle.json to the .kaggle directory
copy kaggle.json %USERPROFILE%\.kaggle\kaggle.json
```

**Linux/Mac:**
```bash
# Create .kaggle directory
mkdir ~/.kaggle

# Copy and set permissions
cp kaggle.json ~/.kaggle/kaggle.json
chmod 600 ~/.kaggle/kaggle.json
```

### 3. Verify Kaggle Setup

```python
import kaggle
kaggle.api.authenticate()
print("✅ Kaggle API authenticated successfully!")
```

## 📊 Available Datasets

The training script supports multiple Kaggle datasets:

### 1. Blood Cells Dataset
- **Dataset:** `paultimothymooney/blood-cells`
- **Size:** ~300MB
- **Classes:** EOSINOPHIL, LYMPHOCYTE, MONOCYTE, NEUTROPHIL
- **Images:** ~12,000 microscopic blood cell images

### 2. Leukemia Classification Dataset
- **Dataset:** `andrewmvd/leukemia-classification`
- **Size:** ~1GB
- **Classes:** ALL (Acute Lymphoblastic Leukemia) vs Normal
- **Images:** ~15,000 microscopic images

### 3. Blood Cancer Dataset
- **Dataset:** `mohammadamireshraghi/blood-cancer-dataset`
- **Size:** ~500MB
- **Classes:** Multiple blood cancer types
- **Images:** ~8,000 labeled images

## 🚀 Training Your Model

### Basic Training

```bash
python train_model_kaggle.py
```

### Advanced Configuration

Edit the configuration in `train_model_kaggle.py`:

```python
# Configuration
MODEL_TYPE = "googlenet"  # Options: "vit-large", "googlenet", "resnet50", "efficientnet"
NUM_CLASSES = 5
EPOCHS = 50
LEARNING_RATE = 0.001
```

### Model Architecture Options

1. **ViT-Large** (`vit-large`)
   - Vision Transformer with 307M parameters
   - Best for large datasets (>10k images)
   - Requires more GPU memory

2. **GoogLeNet** (`googlenet`)
   - Efficient CNN with Inception modules
   - Good balance of accuracy and speed
   - Enhanced with additional layers

3. **ResNet-50** (`resnet50`)
   - Deep residual network
   - Excellent for medical imaging
   - 25M parameters

4. **EfficientNet-B0** (`efficientnet`)
   - State-of-the-art efficiency
   - Compound scaling method
   - Great accuracy with fewer parameters

## 📈 Training Process

The training script will:

1. **Download Dataset** - Automatically download from Kaggle
2. **Data Preprocessing** - Resize, normalize, and augment images
3. **Model Creation** - Initialize your chosen architecture
4. **Training Loop** - Train with validation monitoring
5. **Model Saving** - Save the best performing model
6. **Visualization** - Generate training history plots

### Expected Output

```
🩸 Blood Cancer Detection Model Training
==================================================
✅ Kaggle API authenticated successfully!

📥 Downloading Blood Cells Dataset...
✅ Blood Cells Dataset downloaded successfully!

🔧 Using device: cuda
🎯 Training for 5 classes
🤖 Creating GoogLeNet model...

📊 Preparing dataset...
📈 Found 12000 images across 5 classes

🚀 Starting training for 50 epochs...
Epoch 1/50 [Train]: 100%|██████████| 300/300 [02:15<00:00]
Epoch 1/50 [Val]: 100%|██████████| 75/75 [00:30<00:00]
Epoch 1/50:
  Train Loss: 1.2345
  Val Loss: 0.9876
  Val Accuracy: 78.50%
  ✅ New best model saved! Accuracy: 78.50%
```

## 🎯 Custom Dataset Structure

If you want to use your own dataset, organize it like this:

```
dataset/
├── normal/
│   ├── image1.jpg
│   ├── image2.jpg
│   └── ...
├── all/
│   ├── image1.jpg
│   ├── image2.jpg
│   └── ...
├── aml/
│   ├── image1.jpg
│   └── ...
├── cll/
│   └── ...
└── cml/
    └── ...
```

## 🔧 Troubleshooting

### Common Issues

1. **Kaggle API Authentication Error**
   ```
   OSError: Could not find kaggle.json
   ```
   **Solution:** Ensure `kaggle.json` is in `~/.kaggle/` directory

2. **GPU Memory Error**
   ```
   RuntimeError: CUDA out of memory
   ```
   **Solution:** Reduce batch size or use a smaller model

3. **Dataset Download Fails**
   ```
   403 Forbidden
   ```
   **Solution:** Accept dataset rules on Kaggle website first

### Performance Tips

1. **Use GPU** - Training is much faster with CUDA
2. **Batch Size** - Adjust based on your GPU memory
3. **Data Augmentation** - Improves model generalization
4. **Early Stopping** - Prevents overfitting

## 📊 Model Evaluation

After training, evaluate your model:

```python
# Load trained model
model.load_state_dict(torch.load('best_blood_cancer_model.pth'))

# Generate classification report
from sklearn.metrics import classification_report
print(classification_report(y_true, y_pred))

# Create confusion matrix
import seaborn as sns
import matplotlib.pyplot as plt

cm = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title('Confusion Matrix')
plt.show()
```

## 🚀 Integration with Main App

After training, integrate your model with the main app:

1. **Copy Model File**
   ```bash
   cp best_blood_cancer_model.pth models/
   ```

2. **Update Model Loading** in `app.py`:
   ```python
   # Load your trained model
   model.load_state_dict(torch.load('models/best_blood_cancer_model.pth'))
   ```

## 📚 Additional Resources

- [Kaggle API Documentation](https://github.com/Kaggle/kaggle-api)
- [PyTorch Training Tutorial](https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html)
- [Medical Image Analysis Best Practices](https://arxiv.org/abs/2103.10292)
- [Vision Transformer Paper](https://arxiv.org/abs/2010.11929)

## ⚠️ Important Notes

1. **Medical Disclaimer** - This is for educational purposes only
2. **Data Privacy** - Ensure compliance with medical data regulations
3. **Model Validation** - Always validate on independent test sets
4. **Ethical Use** - Follow ethical guidelines for AI in healthcare

Happy training! 🎉