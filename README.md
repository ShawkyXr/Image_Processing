# Image Classification and Noise Reduction GUI Application

**A Python-based GUI application for Image Noise Reduction and Image Classification (Cats vs Dogs).**

The application allows users to:

* Apply multiple noise reduction filters to images
* Classify images using multiple machine learning algorithms
* Interact with a clean GUI built using `customtkinter`

---

## How to Get Started

### Prerequisites

* Python **3.6+**
* Required libraries:
  * `numpy`
  * `tkinter`
  * `customtkinter`
  * `scikit-learn`
  * `opencv-python`
  * `matplotlib`
  * `Pillow (PIL)`
  * `pickle`

---

## Installation

### 1. Create a virtual environment

```bash
python3 -m venv env
```

### 2. Activate the virtual environment

```bash
source env/bin/activate
```

### 3. Install required packages

```bash
pip install -r requirements.txt
```

---

## Running the Application

```bash
python3 App.py
```

---

# Project Documentation

## Overview

This project implements an **Image Processing GUI Application** that supports:

* **Image Noise Reduction**
* **Image Classification (Cat vs Dog)**

The application integrates multiple classical image filters and multiple machine learning algorithms for classification.

---

## Project Structure

```
IMAGE_PROCESSING/
├── Image_Classification_App/
│   ├── models/
│   ├── data_loader.py
│   ├── decision_tree_model.py
│   ├── knn_model.py
│   ├── svm_model.py
│   ├── Prediction.py
│   └── main.py
│
├── Image_Denoising_App/
│   ├── images/
│   ├── Average_Filter.py
│   ├── Gaussian_Filter.py
│   ├── Median_Filter.py
│   └── main.py
│
├── App.py
├── README.md
├── requirements.txt
└── .gitignore

```

---

# Noise Reduction Algorithms

## Median Filter

**Purpose:**
Removes salt-and-pepper (impulse) noise.

**How it works:**

* Replaces each pixel with the median of neighboring pixels.
* Non-linear filtering approach.

**Strengths:**

* Preserves edges effectively
* Very robust against impulse noise

**Weaknesses:**

* Computationally expensive
* Less effective for Gaussian noise

---

## Average (Mean) Filter

**Purpose:**
General image smoothing.

**How it works:**

* Replaces each pixel with the average of its neighborhood.
* Linear filter.

**Strengths:**

* Simple and fast
* Reduces random noise

**Weaknesses:**

* Blurs edges and fine details
* Weak against salt-and-pepper noise

---

## Gaussian Filter

**Purpose:**
Smooths images while preserving structural details.

**How it works:**

* Uses a Gaussian-weighted average of neighboring pixels.
* Linear filter.

**Strengths:**

* Produces smoother results than the average filter
* Better edge preservation

**Weaknesses:**

* Not suitable for impulse noise
* Parameter tuning required (kernel size, sigma)

---

# Image Classification Algorithms

The application now supports **three machine learning algorithms** for image classification:

* **Support Vector Machine (SVM)**
* **K-Nearest Neighbors (KNN)**
* **Decision Tree (DTree)**

---

## Feature Extraction

All classifiers require numerical feature vectors, not raw images.

Supported feature extraction techniques include:

* Histogram of Oriented Gradients (HOG)
* SIFT (Scale-Invariant Feature Transform)
* ORB (Oriented FAST and Rotated BRIEF)
* Flattened pixel intensities (baseline approach)

---

## Dataset Preparation

* Labeled images (Cats / Dogs)
* Dataset split into:
  * Training set
  * Testing set

---

## Support Vector Machine (SVM)

**Training:**

* Uses extracted feature vectors
* Optimized using hyperparameters:

**Key Parameters:**

* **Kernel:** Linear, RBF, Polynomial
* **C:** Regularization parameter
* **Gamma:** Controls influence of data points (RBF kernel)

**Strengths:**

* High accuracy for well-separated data
* Effective in high-dimensional spaces

**Weaknesses:**

* Sensitive to parameter tuning
* Slower training for large datasets

---

## K-Nearest Neighbors (KNN) — *New Algorithm*

**How it works:**

* Classifies an image based on the majority class of its  **K nearest neighbors** .

**Key Parameters:**

* **K:** Number of neighbors
* **Distance metric:** Euclidean (default)

**Strengths:**

* Simple and intuitive
* No training phase (lazy learning)

**Weaknesses:**

* Slow prediction for large datasets
* Sensitive to noisy data and feature scaling

---

## Decision Tree (DTree) — *New Algorithm*

**How it works:**

* Builds a tree of decisions based on feature thresholds.
* Each leaf node represents a class.

**Key Parameters:**

* **Max depth**
* **Minimum samples per split**

**Strengths:**

* Easy to interpret and visualize
* Fast inference
* Handles non-linear decision boundaries

**Weaknesses:**

* Prone to overfitting
* Less stable with noisy data

---

## Testing and Prediction

* Models are evaluated on the test dataset
* The trained model predicts whether the input image is:
  * **Cat**
  * **Dog**

---

## Summary of Supported Algorithms

| Task                 | Algorithms                |
| -------------------- | ------------------------- |
| Noise Reduction      | Median, Gaussian, Average |
| Image Classification | SVM, KNN, Decision Tree   |
