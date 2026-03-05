import pickle
import re
import ssl
from collections import Counter
from pathlib import Path

import numpy as np
import streamlit as st
import torch
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
from scipy.special import expit
from torch import nn

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

    with open(model_dir / 'oed_rf_classifier.pkl', 'rb') as f:
        rf_clf = pickle.load(f)

    with open(model_dir / 'oed_svm_classifier.pkl', 'rb') as f:
        svm_clf = pickle.load(f)

    with open(model_dir / 'scaler.pkl', 'rb') as f:
        scaler = pickle.load(f)

    with open(model_dir / 'class_map.pkl', 'rb') as f:
        class_map = pickle.load(f)

    return rf_clf, svm_clf, scaler, class_map

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
    """Classify the image and return predictions

    Handles classifiers that lack `predict_proba` (e.g. LinearSVC) by
    converting `decision_function` output into probabilities using a sigmoid.
    """
    # Extract features
    features = extract_features_resnet(image_array, resnet_model, device)

    # Scale features
    features_scaled = scaler.transform(features.reshape(1, -1))

    # Make prediction (always available)
    prediction = clf.predict(features_scaled)[0]

    # Compute probabilities if possible, else fall back to decision_function
    if hasattr(clf, "predict_proba"):
        probabilities = clf.predict_proba(features_scaled)[0]
    else:
        # use decision_function -> sigmoid -> normalize
        decision = clf.decision_function(features_scaled)[0]
        # binary vs multiclass
        if np.ndim(decision) == 0 or len(np.unique(list(class_map.values()))) == 2:
            p = expit(decision)
            probabilities = np.array([1 - p, p])
        else:
            p = expit(decision)
            p = p / p.sum()
            probabilities = p

    # Get class name mapping
    class_names = {v: k for k, v in class_map.items()}
    predicted_class = class_names[prediction]

    return predicted_class, probabilities, class_map

def extract_magnification_from_filename(filename):
    """Extract magnification from filename using regex"""
    # Look for patterns like 4x, 10x, 20x, 40x, etc.
    match = re.search(r'(\d+)x', filename, re.IGNORECASE)
    if match:
        return f"{match.group(1)}x"
    return ""

def aggregate_predictions_majority_voting(predictions):
    """Aggregate predictions using majority voting"""
    if not predictions:
        return None, 0.0

    # Count votes for each class
    vote_counts = Counter(predictions)

    # Get the class with most votes
    majority_class = vote_counts.most_common(1)[0][0]
    confidence = vote_counts[majority_class] / len(predictions)

    return majority_class, confidence

def aggregate_predictions_average_probability(all_probabilities, class_map):
    """Aggregate predictions using average probability"""
    if not all_probabilities:
        return None, np.array([])

    # Convert to numpy array for easier manipulation
    prob_array = np.array(all_probabilities)

    # Average probabilities across all images
    avg_probabilities = np.mean(prob_array, axis=0)

    # Get class names
    class_names = [name for name, _ in sorted(class_map.items(), key=lambda x: x[1])]

    # Find the class with highest average probability
    predicted_class_idx = np.argmax(avg_probabilities)
    predicted_class = class_names[predicted_class_idx]
    confidence = avg_probabilities[predicted_class_idx]

    return predicted_class, avg_probabilities

# Load models
try:
    rf_clf, svm_clf, scaler, class_map = load_model_artifacts()
    resnet_model, device = load_resnet_model()
except Exception as e:
    st.error(f"Error loading models: {e}")
    st.stop()

# Main interface
st.subheader("📤 Upload Images")
uploaded_files = st.file_uploader(
    "Choose multiple image files for a single case",
    type=['jpg', 'jpeg', 'png', 'bmp', 'tiff'],
    accept_multiple_files=True
)

# Magnification inputs for each image
magnifications = {}
if uploaded_files:
    st.subheader("🔍 Magnification Settings")

    # Create columns for magnification inputs
    cols = st.columns(min(len(uploaded_files), 3))

    for i, uploaded_file in enumerate(uploaded_files):
        col_idx = i % 3
        with cols[col_idx]:
            # Auto-detect magnification from filename
            auto_mag = extract_magnification_from_filename(uploaded_file.name)

            # Create input field with auto-filled value
            mag_key = f"mag_{i}"
            current_value = st.text_input(
                f"Magnification for {uploaded_file.name[:20]}...",
                value=auto_mag,
                key=mag_key,
                help=f"Auto-detected: {auto_mag}" if auto_mag else "Enter magnification (e.g., 4x, 10x, 20x)"
            )
            magnifications[uploaded_file.name] = current_value

# Process images and show results
if uploaded_files and len(magnifications) == len(uploaded_files) and all(mag.strip() for mag in magnifications.values()):
    st.divider()

    # Process all images
    with st.spinner("Processing images..."):
        results = []

        for uploaded_file in uploaded_files:
            # Convert to PIL Image
            image = Image.open(uploaded_file).convert('RGB')
            image_resized = image.resize((224, 224))
            image_array = np.array(image_resized)

            # Classify with both models
            rf_pred, rf_probs, _ = classify_image(
                image_resized, rf_clf, scaler, class_map, resnet_model, device
            )
            svm_pred, svm_probs, _ = classify_image(
                image_resized, svm_clf, scaler, class_map, resnet_model, device
            )

            results.append({
                'filename': uploaded_file.name,
                'image': image,
                'predicted_class_rf': rf_pred,
                'probabilities_rf': rf_probs,
                'predicted_class_svm': svm_pred,
                'probabilities_svm': svm_probs,
                'magnification': magnifications[uploaded_file.name]
            })

    # Display individual image results
    st.subheader("🖼️ Individual Image Results")

    # Create a grid layout for images
    cols_per_row = 3
    for i in range(0, len(results), cols_per_row):
        cols = st.columns(cols_per_row)
        for j in range(cols_per_row):
            if i + j < len(results):
                result = results[i + j]
                with cols[j]:
                    st.image(result['image'], caption=f"{result['filename'][:15]}...", use_column_width=True)

                    # Color-coded prediction
                    if result['predicted_class_rf'] == "High Risk OED":
                        st.error(f"**{result['predicted_class_rf']}**")
                    else:
                        st.success(f"**{result['predicted_class_rf']}**")

                    # Show confidence and magnification
                    confidence = max(result['probabilities_rf'])
                    st.caption(f"Confidence: {confidence:.1%} | Mag: {result['magnification']}")

    # Aggregate results
    st.divider()
    st.subheader("📊 Aggregated Case Results")

    # Extract predictions and probabilities for aggregation for each classifier
    predictions_rf = [r['predicted_class_rf'] for r in results]
    all_prob_rf = [r['probabilities_rf'] for r in results]
    predictions_svm = [r['predicted_class_svm'] for r in results]
    all_prob_svm = [r['probabilities_svm'] for r in results]

    # Random Forest aggregations
    majority_class_rf, majority_confidence_rf = aggregate_predictions_majority_voting(predictions_rf)
    avg_class_rf, avg_probabilities_rf = aggregate_predictions_average_probability(all_prob_rf, class_map)

    # SVM aggregations
    majority_class_svm, majority_confidence_svm = aggregate_predictions_majority_voting(predictions_svm)
    avg_class_svm, avg_probabilities_svm = aggregate_predictions_average_probability(all_prob_svm, class_map)

    # Display aggregated results
    # Display aggregated results in two side-by-side columns per classifier
    rf_col1, rf_col2 = st.columns(2)
    with rf_col1:
        st.subheader("🅡🅕 Majority Voting")
        if majority_class_rf == "High Risk OED":
            st.error(f"**Final Prediction: {majority_class_rf}**")
        else:
            st.success(f"**Final Prediction: {majority_class_rf}**")
        st.metric(label="Agreement", value=f"{majority_confidence_rf:.1%}")
        st.caption(f"Based on {len(predictions_rf)} images")
        vote_counts_rf = Counter(predictions_rf)
        st.write("**Vote Distribution:**")
        for class_name, count in vote_counts_rf.items():
            st.write(f"- {class_name}: {count} votes")
    with rf_col2:
        st.subheader("🅡🅕 Average Probability")
        if avg_class_rf == "High Risk OED":
            st.error(f"**Final Prediction: {avg_class_rf}**")
        else:
            st.success(f"**Final Prediction: {avg_class_rf}**")
        confidence = max(avg_probabilities_rf) if len(avg_probabilities_rf) > 0 else 0
        st.metric(label="Avg Confidence", value=f"{confidence:.1%}")
        class_names_list = [name for name, _ in sorted(class_map.items(), key=lambda x: x[1])]
        st.write("**Average Probabilities:**")
        for class_name, prob in zip(class_names_list, avg_probabilities_rf):
            st.write(f"{class_name}: {prob:.1%}")

    st.markdown("---")

    svm_col1, svm_col2 = st.columns(2)
    with svm_col1:
        st.subheader("🅢🅥🅜 Majority Voting")
        if majority_class_svm == "High Risk OED":
            st.error(f"**Final Prediction: {majority_class_svm}**")
        else:
            st.success(f"**Final Prediction: {majority_class_svm}**")
        st.metric(label="Agreement", value=f"{majority_confidence_svm:.1%}")
        st.caption(f"Based on {len(predictions_svm)} images")
        vote_counts_svm = Counter(predictions_svm)
        st.write("**Vote Distribution:**")
        for class_name, count in vote_counts_svm.items():
            st.write(f"- {class_name}: {count} votes")
    with svm_col2:
        st.subheader("🅢🅥🅜 Average Probability")
        if avg_class_svm == "High Risk OED":
            st.error(f"**Final Prediction: {avg_class_svm}**")
        else:
            st.success(f"**Final Prediction: {avg_class_svm}**")
        confidence = max(avg_probabilities_svm) if len(avg_probabilities_svm) > 0 else 0
        st.metric(label="Avg Confidence", value=f"{confidence:.1%}")
        class_names_list = [name for name, _ in sorted(class_map.items(), key=lambda x: x[1])]
        st.write("**Average Probabilities:**")
        for class_name, prob in zip(class_names_list, avg_probabilities_svm):
            st.write(f"{class_name}: {prob:.1%}")

    # Visual comparison
    st.subheader("📊 Comparison Visualization")

    # Random Forest comparison
    st.markdown("### Random Forest")
    comparison_rf = {
        'Majority Voting': {'class': majority_class_rf, 'confidence': majority_confidence_rf},
        'Average Probability': {
            'class': avg_class_rf,
            'confidence': max(avg_probabilities_rf) if len(avg_probabilities_rf) > 0 else 0
        }
    }
    col1, col2 = st.columns(2)
    with col1:
        mv = comparison_rf['Majority Voting']
        if mv['class'] == "High Risk OED":
            st.error("**Majority Voting**")
        else:
            st.success("**Majority Voting**")
        st.write(f"Prediction: {mv['class']}")
        st.write(f"Confidence: {mv['confidence']:.1%}")
    with col2:
        ap = comparison_rf['Average Probability']
        if ap['class'] == "High Risk OED":
            st.error("**Average Probability**")
        else:
            st.success("**Average Probability**")
        st.write(f"Prediction: {ap['class']}")
        st.write(f"Confidence: {ap['confidence']:.1%}")
    if majority_class_rf == avg_class_rf:
        st.success("✅ RF methods agree on the final classification")
    else:
        st.warning("⚠️ RF methods disagree - consider expert review")

    st.markdown("---")

    # SVM comparison
    st.markdown("### Linear SVM")
    comparison_svm = {
        'Majority Voting': {'class': majority_class_svm, 'confidence': majority_confidence_svm},
        'Average Probability': {
            'class': avg_class_svm,
            'confidence': max(avg_probabilities_svm) if len(avg_probabilities_svm) > 0 else 0
        }
    }
    col1, col2 = st.columns(2)
    with col1:
        mv = comparison_svm['Majority Voting']
        if mv['class'] == "High Risk OED":
            st.error("**Majority Voting**")
        else:
            st.success("**Majority Voting**")
        st.write(f"Prediction: {mv['class']}")
        st.write(f"Confidence: {mv['confidence']:.1%}")
    with col2:
        ap = comparison_svm['Average Probability']
        if ap['class'] == "High Risk OED":
            st.error("**Average Probability**")
        else:
            st.success("**Average Probability**")
        st.write(f"Prediction: {ap['class']}")
        st.write(f"Confidence: {ap['confidence']:.1%}")
    if majority_class_svm == avg_class_svm:
        st.success("✅ SVM methods agree on the final classification")
    else:
        st.warning("⚠️ SVM methods disagree - consider expert review")

    # Detailed probability breakdown
    st.subheader("📋 Detailed Results")

    # Create a table with all results (RF & SVM)
    result_table = {
        'Image': [r['filename'][:20] + "..." for r in results],
        'Magnification': [r['magnification'] for r in results],
        'RF Prediction': [r['predicted_class_rf'] for r in results],
        'RF HighRisk': [f"{r['probabilities_rf'][0]:.3f}" for r in results],
        'RF LowRisk': [f"{r['probabilities_rf'][1]:.3f}" for r in results],
        'RF NoRisk': [f"{r['probabilities_rf'][2]:.3f}" for r in results],
        'SVM Prediction': [r['predicted_class_svm'] for r in results],
        'SVM HighRisk': [f"{r['probabilities_svm'][0]:.3f}" for r in results],
        'SVM LowRisk': [f"{r['probabilities_svm'][1]:.3f}" for r in results],
        'SVM NoRisk': [f"{r['probabilities_svm'][2]:.3f}" for r in results]
    }

    st.table(result_table)

else:
    if uploaded_files:
        # Check if all magnifications are filled
        empty_mags = [name for name, mag in magnifications.items() if not mag.strip()]
        if empty_mags:
            st.warning(f"⚠️ Please fill in magnification for: {', '.join(empty_mags[:3])}{'...' if len(empty_mags) > 3 else ''}")
    else:
        st.info("👆 Upload multiple images to get started")

# Footer
st.divider()
st.markdown("""
---
**Model Information:**
- Classifiers used: Random Forest & Linear SVM
- RF details: 100 estimators, max_depth=20, trained with `random_state=42`
- SVM details: LinearSVC, C=1.0, class_weight='balanced', max_iter=2000
- Feature extractor: ResNet50 (pretrained on ImageNet, final layer removed)
- Input size for each patch: 224 × 224 pixels
- Feature dimension after ResNet: 2048
- Training dataset: augmented OED patches (PIL augmentation applied)

**Aggregation Methods (per model):**
- **Majority Voting**: Count predictions across all images
- **Average Probability**: Average probability (or sigmoid‑converted decision scores) then take highest
""")
