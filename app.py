import streamlit as st
import pandas as pd
import joblib
import numpy as np

from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    matthews_corrcoef,
    confusion_matrix,
    ConfusionMatrixDisplay
)
import matplotlib.pyplot as plt

st.set_page_config(
    page_title="Breast Cancer Classifier Comparison",
    page_icon="🔬",
    layout="wide"
)

st.title("Breast Cancer Classification")
st.write(
    "Compare five trained machine learning classifiers "
    "on an uploaded test dataset."
)

MODEL_DIR = "model"

model_files = {
    "Logistic Regression": "logistic_regression.joblib",
    "Decision Tree": "decision_tree.joblib",
    "KNN": "knn.joblib",
    "Naive Bayes": "naive_bayes.joblib",
    "Support Vector Machine": "support_vector_machine.joblib"
}

scaled_models = {
    "Logistic Regression",
    "KNN",
    "Support Vector Machine"
}

@st.cache_resource
def load_models():
    loaded = {
        name: joblib.load(f"{MODEL_DIR}/{filename}")
        for name, filename in model_files.items()
    }
    scaler = joblib.load(f"{MODEL_DIR}/standard_scaler.joblib")
    return loaded, scaler

try:
    models, scaler = load_models()
except Exception as exc:
    st.error(f"Could not load the saved models: {exc}")
    st.stop()

uploaded_file = st.file_uploader(
    "Upload test data (CSV)",
    type=["csv"]
)

if uploaded_file is not None:
    try:
        test_df = pd.read_csv(uploaded_file)

        if "diagnosis" not in test_df.columns:
            st.error("The CSV must contain a 'diagnosis' target column.")
            st.stop()

        feature_names = list(models["Logistic Regression"].feature_names_in_)
        missing = [col for col in feature_names if col not in test_df.columns]

        if missing:
            st.error(f"Missing required input features: {missing}")
            st.stop()

        if test_df.empty:
            st.error("The uploaded CSV contains no rows.")
            st.stop()

        X_test = test_df[feature_names]
        y_test = test_df["diagnosis"]

        if X_test.isnull().any().any() or y_test.isnull().any():
            st.error("Please remove or handle missing values before evaluation.")
            st.stop()

        if not set(y_test.unique()).issubset({0, 1}):
            st.error("Diagnosis must use 0 = benign and 1 = malignant.")
            st.stop()

        if y_test.nunique() < 2:
            st.error("The test data must contain both target classes to calculate AUC.")
            st.stop()

        st.subheader("Uploaded dataset")
        st.write(f"Rows: {len(test_df)} | Input features: {len(feature_names)}")
        st.dataframe(test_df.head())

        selected_model = st.selectbox(
            "Select a classification model",
            list(model_files.keys())
        )

        model = models[selected_model]

        if selected_model in scaled_models:
            X_eval = scaler.transform(X_test)
        else:
            X_eval = X_test

        y_pred = model.predict(X_eval)
        y_prob = model.predict_proba(X_eval)[:, 1]

        metrics = {
            "Accuracy": accuracy_score(y_test, y_pred),
            "AUC": roc_auc_score(y_test, y_prob),
            "Precision": precision_score(y_test, y_pred, zero_division=0),
            "Recall": recall_score(y_test, y_pred, zero_division=0),
            "F1 Score": f1_score(y_test, y_pred, zero_division=0),
            "MCC": matthews_corrcoef(y_test, y_pred)
        }

        st.subheader(f"Evaluation: {selected_model}")

        cols = st.columns(3)
        for i, (metric, value) in enumerate(metrics.items()):
            cols[i % 3].metric(metric, f"{value:.4f}")

        st.subheader("Confusion Matrix")
        cm = confusion_matrix(y_test, y_pred, labels=[0, 1])

        fig, ax = plt.subplots(figsize=(5, 4))
        display = ConfusionMatrixDisplay(
            confusion_matrix=cm,
            display_labels=["Benign", "Malignant"]
        )
        display.plot(ax=ax, cmap="Blues", values_format="d")
        st.pyplot(fig)
        plt.close(fig)

        st.caption(
            "Target encoding: 0 = benign, 1 = malignant. "
            "The app loads saved models and does not retrain them."
        )

    except Exception as exc:
        st.error(f"Unable to evaluate the uploaded CSV: {exc}")
else:
    st.info("Upload test_data.csv to evaluate a trained model.")
