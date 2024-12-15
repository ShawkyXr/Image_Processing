# Image Classification and Noise Reduction GUI APP
    Image Classification and Noise Reduction GI APP python: tknter

## How to Get Started 
### Prerequisities
- Python 3.6 or higher
- tknter
- Pillow
  
## How to Install the required packages

1. Create a virtual enviroment
   
''' pash
$ python3 -m venv env
'''

2. Activate the virtual environment

```bash

$ source env/bin/activate
```

3. Install the required packages

```bash
$ pip install -r requirements.txt
```

### How to run the app

```bash
$ python3 App.py
```


# Project Documentation: Image Compression Application

## Overview

This project implements an image Classification and Noise Reduction application with a graphical user interface (GUI) using tknter, it allows the user to view image and select the filter he want to use to Denoise it and it allow the user to insert a aphoto to predict wither it's a cat or a dog.

- **App.py**: The main application that defines the GUI and handles user interactions and contain's tha main window.
- **Image_Denoising_App/main.py**: The function witch control the whole Denoising Process.
- **Median_Filter.py**: A utility to reduce noise in images using the Median Filter.
- **Gaussian_Filter.py**: A utility to reduce noise in images using the Gaussian Filter.
- **Average_Filter.py**: A utility to reduce noise in images using the Average Filter.
- **Image_Classification_App/main.py**: The function witch control the whole Classification Process.
- **Model_train.py**: A utility to Train a Model with up to 8000+ images.
- **Prediction.py**: A utility to predicit a given images using the pretrained Model.
  
---

## `App.py`

This file implements the main application GUI using tknter.

## Main FUnctions

## **Median Filter**

**Purpose:** Primarily used to remove salt-and-pepper noise from images.

**Operation:**
- The pixel value is replaced with the median of the values in its neighborhood.
- This is a non-linear filter.

**Strengths:**
- Preserves edges effectively since it replaces outliers without blurring.
- Robust against impulse noise.

**Weaknesses:**
- Computationally intensive compared to linear filters.
- Not ideal for Gaussian noise.


## **Average Filter (Mean Filter)**

**Purpose:** Used for general smoothing of images to reduce noise.

**Operation:**
- The pixel value is replaced with the average of the values in its neighborhood.
- This is a linear filter.

**Strengths:**
- Simple and fast to compute.
- Reduces random noise effectively.

**Weaknesses:**
- Blurs edges and details.
- Not effective at removing high-intensity noise like salt-and-pepper.
- 
## **Gaussian Filter**

**Purpose:** Used for smoothing while preserving important structural details in the image.
**Operation:**
- The pixel value is computed using a weighted average where weights follow a Gaussian distribution.
- This is a linear filter.

**Strengths:**
- Provides a smoother result compared to an average filter.
- Slightly more computationally intensive than the average filter.

**Weaknesses:**
- Not effective at removing impulse noise like salt-and-pepper noise.
- Not ideal for Gaussian noise.

## **Image Classification using SVM**
**Feature Extraction:**
- SVM require vectorized data as input, not war images;
- ### **Extract features from images using techniques such as:**
  - Histogram of Oriented Gradients (HOG)
  - SIFT (Scale-Invariant Feature Transform)
  - ORB (Oriented FAST and Rotated BRIEF)
  - Flattening pixel intensities (less common now).
## **Dataset Preperation:**
- Prepare a dataset with labeled images.
- Split the data into training and testing sets.
## **Training SVM:**
- Use the extracted feature vectors as input to train the SVM model.
- ### **Optimize hyperparameters such as:**
  - **Kernel**: Linear, RBF (Radial Basis Function), or Polynomial.
  - **C** (Regularization parameter): Controls the trade-off between achieving a low error on training data and a low margin.
  - **Gamma** (for RBF kernel): Controls the influence of individual data points.

## **Testing and Classification:**
- Test the model on the test dataset.
- The SVM assigns a label to each image based on the decision boundary it learned during training.