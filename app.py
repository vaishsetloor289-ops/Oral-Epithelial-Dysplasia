import pickle
from pathlib import Path

import numpy as np
import streamlit as st
import torch
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
from torch import nn
import ssl

ssl._create_default_https_context = ssl._create_unverified_context

# Set up page configuration
st.set_page_config(
    page_title="OED Classification",
    page_icon="🔬",
    layout="wide"
)

# Add title and description
st.title("🔬 Epithelial Dysplasia (OED) Classifier")
st.markdown("""
This application classifies Oral Epithelial Dysplasia (OED) patches as **High Risk** or **Low Risk** 
using a trained Random Forest model with ResNet50 feature extraction.
""")

# Load model artifacts
@st.cache_resource
def load_model_artifacts():
    """Load the trained model, scaler, and class map"""
    model_dir = Path.cwd() / 'viz_outputs' / 'models'
    
    with open(model_dir / 'oed_classifier.pkl', 'rb') as f:
        clf = pickle.load(f)
    
    with open(model_dir / 'scaler.pkl', 'rb') as f:
        scaler = pickle.load(f)
    
    with open(model_dir / 'class_map.pkl', 'rb') as f:
        class_map = pickle.load(f)
    
    return clf, scaler, class_map

@st.cache_resource
def load_resnet_model():
    """Load the ResNet50 model for feature extraction"""
    device = torch.device('cpu')
    model = models.resnet50(pretrained=True).to(device)
    model = nn.Sequential(*list(model.children())[:-1])  # Remove classification layer
    model.eval()
    return model, device

def extract_features_resnet(image_array, model, device):
    """Extract features from an image using ResNet50"""
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225])
    ])
    
    with torch.no_grad():
        img_tensor = transform(image_array).unsqueeze(0).to(device)
        features = model(img_tensor).cpu().numpy().flatten()
    
    return features

def classify_image(image_array, clf, scaler, class_map, resnet_model, device):
    """Classify the image and return predictions"""
    # Extract features
    features = extract_features_resnet(image_array, resnet_model, device)
    
    # Scale features
    features_scaled = scaler.transform(features.reshape(1, -1))
    
    # Make prediction
    prediction = clf.predict(features_scaled)[0]
    probabilities = clf.predict_proba(features_scaled)[0]
    
    # Get class name
    class_names = {v: k for k, v in class_map.items()}
    predicted_class = class_names[prediction]
    
    return predicted_class, probabilities, class_map

# Load models
try:
    clf, scaler, class_map = load_model_artifacts()
    resnet_model, device = load_resnet_model()
except Exception as e:
    st.error(f"Error loading models: {e}")
    st.stop()

# Create two columns for layout
col1, col2 = st.columns(2)

with col1:
    st.subheader("Upload Image")
    uploaded_file = st.file_uploader(
        "Choose an image file",
        type=['jpg', 'jpeg', 'png', 'bmp', 'tiff']
    )

with col2:
    st.subheader("Image Preview")
    if uploaded_file is not None:
        # Convert to PIL Image
        image = Image.open(uploaded_file).convert('RGB')
        st.image(image, use_column_width=True)

# Classification section
if uploaded_file is not None:
    st.divider()
    
    with st.spinner("Classifying image..."):
        # Resize image to match training size
        image_resized = image.resize((224, 224))
        image_array = np.array(image_resized)
        
        # Classify
        predicted_class, probabilities, class_map_inv = classify_image(
            image_resized, clf, scaler, class_map, resnet_model, device
        )
    
    # Display results
    st.subheader("Classification Results")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Display prediction with color coding
        if predicted_class == "High Risk OED":
            st.error(f"**Prediction: {predicted_class}**")
        else:
            st.success(f"**Prediction: {predicted_class}**")
    
    with col2:
        # Display confidence
        class_names_list = [name for name, _ in sorted(class_map.items(), key=lambda x: x[1])]
        confidence = max(probabilities)
        st.metric(label="Confidence", value=f"{confidence:.2%}")
    
    # Display detailed probabilities
    st.subheader("Class Probabilities")
    
    prob_data = {}
    for class_id, class_name in enumerate(class_names_list):
        prob_data[class_name] = probabilities[class_id]
    
    # Create bar chart
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.bar_chart(prob_data)
    
    with col2:
        st.table({
            'Class': list(prob_data.keys()),
            'Probability': [f"{v:.4f}" for v in prob_data.values()]
        })

else:
    st.info("👆 Upload an image to get started")

# Footer
st.divider()
st.markdown("""
---
**Model Information:**
- Model Type: Random Forest Classifier (100 estimators)
- Feature Extraction: ResNet50 (pretrained on ImageNet)
- Input Size: 224 × 224 pixels
- Feature Dimension: 2048
- Dataset: Training on augmented OED patches
""")
