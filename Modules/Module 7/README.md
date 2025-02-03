
# Predicting Martian Core Radius Using Regression

This repository contains a Python-based implementation of a regression model to predict the Martian core radius. The model utilizes seismic wave velocities and density data to estimate the core radius, applying machine learning techniques such as **Random Forest** and **Linear Regression**. This solution also leverages hyperparameter tuning using **Grid Search** and additionally uses **LightGBM Regressor** and **AdaBoost** for better results.

## Overview

This project automates the process of:

- Loading seismic wave data containing **P-wave velocity (P_v)**, **S-wave velocity (S_v)**, and **density (ρ)**.
- Training a regression model to predict the **Martian core radius (R_c)** based on the input parameters.
- Evaluating the model’s performance using standard regression metrics such as **Mean Squared Error (MSE)** and **R² score**.

---

## Challenges Faced

### **Data Correlation & Feature Significance**
The relationship between seismic properties and core radius is nonlinear, requiring feature selection and tuning to improve prediction accuracy.

### **Model Selection & Optimization**
Choosing between models like **Random Forest** and **Linear Regression** was based on trade-offs between interpretability and performance.

### **Limited Data**
With limited seismic observations, preventing overfitting and ensuring generalization was a challenge. We found data to ensure there is efficient generalization for our model to work effectively. 

---

## Methodology

### **1. Loading Dataset**
- The dataset consists of seismic wave velocities (**P_v, S_v**) and **density (ρ)** as input features, with **core radius (R_c)** as the output.

### **2. Model Training**

#### **Regression Algorithms Used**
- **a. Random Forest Regression:**  
  A tree-based ensemble model that captures complex patterns but requires hyperparameter tuning. It aggregates multiple decision trees to reduce overfitting and improve predictive accuracy.

- **b. LightGBM Boost:**  
  A gradient boosting framework that builds trees efficiently using a leaf-wise growth strategy. LightGBM is optimized for speed and handles large datasets well, but requires careful tuning to prevent overfitting.

- **c. AdaBoost:**  
  An adaptive boosting algorithm that combines multiple weak learners (typically decision trees) to create a strong predictive model. It assigns higher weights to misclassified samples in each iteration, improving model robustness, but can be sensitive to noise in the data.

#### **Hyperparameter Tuning**
- **Grid Search** and **Cross-Validation** are applied to optimize hyperparameters, improving the model's performance.

#### **Evaluation Metrics**
- **Mean Squared Error (MSE):** Measures prediction error.
- **R² Score:** Indicates the model’s explanatory power.

---

## Results and Discussion

### **1. Model Performance**
- The **Random Forest model** achieved a lower **MSE** compared to **Linear Regression**, indicating better predictive accuracy.
- The **R² score** suggests how well seismic velocities and density explain variations in the core radius.

### **2. Feature Importance**
- **Density (ρ)** was found to be the most influential factor in predicting core radius.
- **P-wave velocity (P_v)** and **S-wave velocity (S_v)** also contributed significantly.

---

## Files and Outputs

### **1. Code File**
- `Module_7(Code).ipynb` → Jupyter Notebook containing data processing, model training, and evaluation.

### **2. Data Sources**
- https://dataverse.ipgp.fr/dataset.xhtml?persistentId=doi:10.18715/IPGP.2023.llxn7e6d



