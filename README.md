# Oral Epithelial Dysplasia Classification

This project focuses on classifying Oral Epithelial Dysplasia (OED) into High Risk and Low Risk categories using machine learning techniques. It consists of two main components: a training pipeline for model development and a Streamlit web application for interactive classification.

## Training

The training process involves preprocessing histopathological patches, training classification models, and generating visualizations to analyze the dataset and model performance.

### Key Features:
- **Data Preparation**: Patches are exported and organized into `exported_patches/High Risk OED/` and `exported_patches/Low Risk OED/` folders.
- **Model Training**: Utilize the `oed_classification.ipynb` notebook to train models on the prepared dataset. Trained models are saved in the `models/` directory.
- **Validation**: Evaluate model performance using data in the `validation/` folder.
- **Visualizations**: Generate insightful visualizations stored in `viz_outputs/`, including dataset distributions, model metrics, and patch analyses.

To run the visualization script and create images in `viz_outputs/`:


## Streamlit App

The interactive Streamlit application allows users to upload histopathological images for real-time classification. It employs a voting process where multiple trained models evaluate the input, providing a consensus classification into High Risk or Low Risk OED categories.

### Features:
- **Image Upload**: Users can upload patch images for analysis.
- **Voting Process**: The app aggregates predictions from ensemble models to determine the final classification.
- **Real-time Results**: Instant feedback on the dysplasia risk level.

Access the live application at: [https://oral-epithelial-dysplasia.streamlit.app/](https://oral-epithelial-dysplasia.streamlit.app/)

To run the app locally:

```bash
streamlit run app.py
```
