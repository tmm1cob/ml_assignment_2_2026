# Breast Cancer Classification Using Machine Learning

## Project Overview

The objective of this project is to classify breast cancer tumours as benign or malignant using machine learning algorithms.

I have used the Breast Cancer Wisconsin Diagnostic dataset and implemented five classification models. The performance of these models is compared using different evaluation metrics.

A Streamlit application is also developed to test the trained models using a CSV file and compare their performance.

## Dataset

- Dataset: Breast Cancer Wisconsin Diagnostic
- Number of samples: 569
- Number of features: 30
- Target classes: Benign and Malignant
- Training and testing split: 80:20
- Missing values: None

The dataset contains 357 benign and 212 malignant samples.

StandardScaler is used for Logistic Regression, KNN and SVM. Decision Tree and Gaussian Naive Bayes use the original features.

## Machine Learning Models

1. Logistic Regression
2. Decision Tree
3. K-Nearest Neighbors (KNN)
4. Gaussian Naive Bayes
5. Support Vector Machine (SVM)

## Evaluation Metrics

The models are evaluated using Accuracy, AUC, Precision, Recall, F1 Score and Matthews Correlation Coefficient (MCC).

## Model Performance

| Model | Accuracy | AUC | Precision | Recall | F1 Score | MCC |
|---|---:|---:|---:|---:|---:|---:|
| Logistic Regression | 96.49% | 99.60% | 97.50% | 92.86% | 95.12% | 0.9245 |
| Decision Tree | 92.98% | 92.46% | 90.48% | 90.48% | 90.48% | 0.8492 |
| KNN | 95.61% | 98.23% | 97.44% | 90.48% | 93.83% | 0.9058 |
| Gaussian Naive Bayes | 93.86% | 99.34% | 100.00% | 83.33% | 90.91% | 0.8715 |
| SVM | 97.37% | 99.47% | 100.00% | 92.86% | 96.30% | 0.9442 |

## Model Observations

| Model | Observation |
|---|---|
| Logistic Regression | Achieved 96.49% accuracy and the highest AUC score of 99.60%. The model performed well overall. |
| Decision Tree | Achieved 92.98% accuracy. Its performance was lower compared to the other models. |
| KNN | Achieved 95.61% accuracy with good precision and F1 score. |
| Gaussian Naive Bayes | Achieved 100% precision, but recall was lower at 83.33%. Some malignant cases were not correctly classified. |
| SVM | Achieved the highest accuracy, F1 score and MCC among the five models. |

### Best Performing Model

SVM achieved the highest accuracy of 97.37% and F1 score of 96.30% on the test dataset, while Logistic Regression achieved the highest AUC.

Based on the overall results, SVM is the best-performing model for this dataset.

## Streamlit Application

The application allows users to upload a test CSV file, select a trained model and view its performance.

The application includes:

- CSV file upload option
- Sample test data option
- Selection of five trained models
- Display of six evaluation metrics
- Circular meters for model performance
- Confusion matrix
- Comparison of all five models
- Dataset preview

The models are loaded from saved model files without retraining.

## How to Run

Install the required packages:

```bash
pip install -r requirements.txt
```

Start the application:

```bash
streamlit run app.py
```

## Project Files

```text
ml_assignment_2_2026/
├── app.py
├── README.md
├── requirements.txt
├── test_data.csv
└── model/
    ├── decision_tree.joblib
    ├── knn.joblib
    ├── logistic_regression.joblib
    ├── naive_bayes.joblib
    ├── standard_scaler.joblib
    └── support_vector_machine.joblib
```

## Links

- GitHub Repository: https://github.com/tmm1cob/ml_assignment_2_2026
- Streamlit Application: https://mlassignment22026-cfa3wgjm7unpt9pqzswguu.streamlit.app/

## Conclusion

All five classification models were implemented and evaluated using the same dataset.

SVM achieved the best overall performance, while Logistic Regression achieved the highest AUC score.

The Streamlit application is developed to evaluate the saved models using test data without retraining.
