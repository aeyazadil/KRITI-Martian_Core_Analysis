# Seismic Image Clustering and Anomaly Detection

This repository contains a Python script for unsupervised clustering of seismic images. The code leverages a pre-trained ResNet50 model for feature extraction, uses UMAP for dimensionality reduction, and applies K-Means clustering to group seismic images. This process aids in identifying anomalies in the data—such as distinct fault horizons or chaotic regions—that may indicate important geological features.

## Overview

This project automates the process of:
- Loading seismic data stored in `.mat` files (organized by geological category).
- Extracting features from the seismic images using a modified ResNet50.
- Reducing the feature dimensionality with UMAP for better interpretability.
- Clustering the data using K-Means to detect anomalies and differentiate geological patterns.
- Evaluating the clustering results and visualizing the data to highlight features such as fault horizons and chaotic seismic regions.

---

## Challenges Faced

- **Data Complexity:**  
  Seismic images often contain intricate and subtle geological patterns. Extracting meaningful features required careful image normalization and pre-processing.

- **High Dimensionality:**  
  The features extracted from ResNet50 are high-dimensional. Reducing this dimensionality was essential to make clustering computationally feasible and to help reveal inherent data structures.

- **Geological Ambiguity:**  
  Differentiating between well-defined fault horizons and disordered, chaotic regions can be challenging, as the features may overlap or be subtle in some cases.

---

## Methodology

1. **Feature Extraction:**
   - **Data Loading:**  
     A custom `SeismicDataset` class loads seismic images from `.mat` files. The images are processed to ensure a 3-channel format and normalized to a consistent range.
     
   - **Pre-trained Model:**  
     A pre-trained ResNet50 model (with its final classification layer removed) is used to extract robust high-level features from the seismic images. This model operates in evaluation mode, utilizing GPU acceleration if available.

2. **Dimensionality Reduction:**
   - **UMAP:**  
     The extracted high-dimensional feature vectors are reduced to 50 dimensions using UMAP. This reduction helps maintain the essential structure of the data while facilitating efficient clustering.
   - **Visualization:**  
     A secondary reduction to 2 dimensions is performed specifically for generating visualizations. UMAP projections are saved as images to compare true labels against clustering outcomes.

3. **Clustering:**
   - **K-Means:**  
     K-Means clustering is applied to the UMAP-reduced features. The number of clusters is chosen based on geological insights and heuristic methods.  
     The resulting cluster assignments help identify regions where seismic patterns differ, allowing for the detection of anomalies that might indicate important geological features.

4. **Evaluation and Anomaly Detection:**
   - **Visualization and Mapping:**  
     UMAP plots colored by true labels (`umap_true_labels.png`) and by K-Means cluster assignments (`umap_kmeans_clusters.png`) are generated to assess clustering quality.
   - **Metric Calculation:**  
     The Adjusted Rand Index (ARI) and class-specific metrics (precision, recall, F1-score) are computed to evaluate clustering performance.
   - **Geological Insights:**  
     By examining the clustering output and visualizations, anomalies such as distinct fault horizons (linear, well-defined patterns) and chaotic regions (more scattered clusters) are identified. These anomalies can serve as indicators for further geological analysis.

---

## Results and Discussion

1. **UMAP Visualizations:**
   - **True Labels:**  
     The `umap_true_labels.png` file displays the UMAP projection colored by the actual (inferred) seismic labels.
   - **K-Means Clusters:**  
     The `umap_kmeans_clusters.png` file shows the same projection but with colors indicating the clusters formed by K-Means.
   - **Interpretation:**  
     Differences between these plots help reveal how well the clustering captures distinct geological patterns. Fault horizons tend to form distinct, compact clusters, while chaotic regions appear more dispersed.

2. **Clustering Performance:**
   - The K-Means clustering approach achieved a moderate ARI, indicating that while there is partial agreement between the clusters and true labels, some discrepancies remain.
   - Quantitative metrics (precision, recall, F1-score) vary across geological classes, suggesting that some features are more readily distinguished than others.

3. **File Outputs:**
   - **Models and Reducers:**  
     - `feature_extractor.pth` (ResNet50 feature extractor weights)  
     - `umap_reducer.joblib` and `umap_reducer_2d.joblib` (UMAP reducers for dimensionality reduction)
     - `kmeans_model.joblib` (K-Means clustering model)
   - **Clustering and Evaluation Data:**  
     - `cluster_assignments.npy` (K-Means cluster labels)
     - `cluster_to_label_mapping.npy` (Mapping of clusters to geological labels)
     - `classification_metrics.csv` (Detailed performance metrics per class)
     - `predictions.csv` (File-level predictions with true and predicted labels)
   - **Visualizations:**  
     - `umap_true_labels.png`
     - `umap_kmeans_clusters.png`

4. **Anomaly Detection and Geological Insights:**
   - The clustering approach successfully identifies anomalies in seismic data:
     - **Fault Horizons:** Regions exhibiting linear, distinct seismic patterns are clustered together.
     - **Chaotic Regions:** Areas with disordered or ambiguous seismic signals are more scattered, indicating potential geological anomalies.
   - These findings provide a valuable basis for further geophysical analysis and decision-making.

---

## Files and Outputs

- **Code File:**  
  - `main.py` (Main Python script)

- **Saved Models/Reducers:**  
  - `feature_extractor.pth`
  - `umap_reducer.joblib`
  - `umap_reducer_2d.joblib`
  - `kmeans_model.joblib`
  
- **Clustering and Evaluation Data:**  
  - `cluster_assignments.npy`
  - `cluster_to_label_mapping.npy`
  - `classification_metrics.csv`
  - `predictions.csv`

- **Visualization Files:**  
  - `umap_true_labels.png`
  - `umap_kmeans_clusters.png`

---

## Usage

1. **Environment Setup:**
   - Ensure Python 3.x is installed along with required packages (PyTorch, torchvision, scikit-learn, UMAP, joblib, pandas, etc.).
   - You can use the following line for installing the required libraries:
        pip install -r requirements.txt
    
2. **Data Preparation:**
   - Organize your seismic `.mat` files in subdirectories under the designated dataset paths.
   - Update the dataset paths (`dataset` and `test_dataset`) in the code as needed.

3. **Running the Code:**
   - Run the script from the command line:
        python main.py
   - The script will execute feature extraction, clustering, evaluation, and testing on new seismic data / or used for testing depending upon which function you choose to run in the script!.

4. **Results:**
   - Examine the generated files and visualizations to analyze clustering performance and geological anomalies.
---
