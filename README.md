# Breast Cancer Classification Using Machine Learning

## Project Overview

The objective of this project is to classify breast cancer tumours as benign or malignant using machine learning algorithms.

I have used the Breast Cancer Wisconsin Diagnostic dataset and implemented five classification models. The performance of these models is compared using different evaluation metrics.

A Streamlit application is also developed to test the trained models using a CSV file.

## Dataset

- Dataset: Breast Cancer Wisconsin Diagnostic
- Number of samples: 569
- Number of features: 30
- Target classes: Benign and Malignant

## Machine Learning Models

1. Logistic Regression
2. Decision Tree
3. K-Nearest Neighbors (KNN)
4. Gaussian Naive Bayes
5. Support Vector Machine (SVM)

## Evaluation Metrics

The models are evaluated using Accuracy, AUC, Precision, Recall, F1 Score and Matthews Correlation Coefficient (MCC).

## Model Performance

| Model | Accuracy | AUC | F1 Score |
|---|---:|---:|---:|
| Logistic Regression | 96.49% | 99.60% | 95.12% |
| Decision Tree | 92.98% | 92.46% | 90.48% |
| KNN | 95.61% | 98.23% | 93.83% |
| Gaussian Naive Bayes | 93.86% | 99.34% | 90.91% |
| SVM | 97.37% | 99.47% | 96.30% |

SVM achieved the highest accuracy and F1 score on the test dataset, while Logistic Regression achieved the highest AUC.

## Streamlit Application

The application allows users to upload a test CSV file, select a trained model and view its performance metrics and confusion matrix.

The models are loaded from saved model files without retraining.

## How to Run

Install the required packages:

    pip install -r requirements.txt

Start the application:

    streamlit run app.py

## Project Files

- app.py
- requirements.txt
- test_data.csv
- model/

## Links

- GitHub Repository: To be added
- Streamlit Application: To be added
