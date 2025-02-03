
# Shadow vs Non-Shadow Classification

This repository contains Jupyter notebooks for feature extraction, synthetic dataset generation, and classification of shadow vs. non-shadow regions using machine learning models.

## Table of Contents
- [Feature Extraction](#feature-extraction)
- [Synthetic Dataset Generation](#synthetic-dataset-generation)
- [Classification Model](#classification-model)
- [Reference](#reference)
- [Installation](#installation)
- [Usage](#usage)

## Feature Extraction
The `feature_extraction.ipynb` notebook extracts relevant features from seismic waveforms for shadow classification. 

### Key Features:
- Reads seismic waveforms using Syngine API.
- Applies preprocessing techniques including baseline correction, bandpass filtering, and normalization.
- Extracts amplitude, frequency, and phase shift features.
- Computes dominant frequency using FFT.
- Saves extracted features for further processing.

## Synthetic Dataset Generation
The `Synthetic Dataset generation.ipynb` notebook generates synthetic seismic data with shadow and non-shadow event labels.

### Key Features:
- Uses TauPyModel to compute seismic wave travel times.
- Determines whether an event falls in the seismic shadow zone.
- Simulates waveform responses for different locations and magnitudes.
- Extracts key features such as amplitude, RMS amplitude, dominant frequency, and ray parameter.
- Balances the dataset by ensuring an equal number of shadow and non-shadow events.
- Saves the generated dataset for training.

## Classification Model
The `Model - Shadow vs non shadow classifier.ipynb` notebook implements a machine learning model to classify seismic events as shadow or non-shadow.

### Key Features:
- Loads preprocessed dataset containing extracted seismic features.
- Applies data augmentation techniques, including Gaussian noise, uniform noise, and salt-and-pepper noise.
- Splits the dataset into training and testing sets.
- Uses a Random Forest classifier for classification.
- Performs hyperparameter tuning using RandomizedSearchCV.
- Evaluates model performance using classification reports, confusion matrices, ROC curves, and feature importance analysis.
- Visualizes feature importance using bar plots.

## Reference
The TauPy geophysical model used in this project was downloaded from: [Dataverse IPGP](https://dataverse.ipgp.fr/dataset.xhtml?persistentId=doi:10.18715/IPGP.2022.kzwpiude)

## Usage
1. Run the `feature_extraction.ipynb` notebook to extract features from seismic waveforms.
2. Run `Synthetic Dataset generation.ipynb` to create a dataset of shadow and non-shadow seismic events.
3. Train and evaluate the classifier using `Model - Shadow vs non shadow classifier.ipynb`.
