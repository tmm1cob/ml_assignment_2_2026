
import streamlit as st
import pandas as pd
import joblib
import os

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


# --------------------------------------------------
# 1. Streamlit Page Configuration
# --------------------------------------------------

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


# --------------------------------------------------
# 2. Model Configuration
# --------------------------------------------------

MODEL_DIR = "model"

model_files = {
    "Logistic Regression": "logistic_regression.joblib",
    "Decision Tree": "decision_tree.joblib",
    "KNN": "knn.joblib",
    "Naive Bayes": "naive_bayes.joblib",
    "Support Vector Machine": "support_vector_machine.joblib"
}

# Models that require scaled input
scaled_models = {
    "Logistic Regression",
    "KNN",
    "Support Vector Machine"
}


# --------------------------------------------------
# 3. Load Saved Models and Scaler
# --------------------------------------------------

@st.cache_resource
def load_models():

    loaded = {
        name: joblib.load(os.path.join(MODEL_DIR, filename))
        for name, filename in model_files.items()
    }

    scaler = joblib.load(
        os.path.join(MODEL_DIR, "standard_scaler.joblib")
    )

    return loaded, scaler


try:
    models, scaler = load_models()

except Exception as exc:
    st.error(f"Could not load the saved models: {exc}")
    st.stop()


# --------------------------------------------------
# 4. Upload Test Dataset
# --------------------------------------------------

uploaded_file = st.file_uploader(
    "Upload test data (CSV)",
    type=["csv"]
)


if uploaded_file is not None:

    try:

        test_df = pd.read_csv(uploaded_file)

        # Check target column
        if "diagnosis" not in test_df.columns:
            st.error(
                "The CSV must contain a 'diagnosis' target column."
            )
            st.stop()

        # Read expected feature names from test CSV header
        feature_names = [
            col for col in pd.read_csv(
                "test_data.csv", nrows=0
            ).columns
            if col != "diagnosis"
        ]

        # Check missing features
        missing = [
            col for col in feature_names
            if col not in test_df.columns
        ]

        if missing:
            st.error(
                f"Missing required input features: {missing}"
            )
            st.stop()

        # Check empty dataset
        if test_df.empty:
            st.error("The uploaded CSV contains no rows.")
            st.stop()

        # Separate features and target
        X_test = test_df[feature_names]
        y_test = test_df["diagnosis"]

        # Check missing values
        if X_test.isnull().any().any() or y_test.isnull().any():
            st.error(
                "Please remove or handle missing values "
                "before evaluation."
            )
            st.stop()

        # Check target encoding
        if not set(y_test.unique()).issubset({0, 1}):
            st.error(
                "Diagnosis must use 0 = benign and 1 = malignant."
            )
            st.stop()

        # Both classes are required for AUC
        if y_test.nunique() < 2:
            st.error(
                "The test data must contain both target classes "
                "to calculate AUC."
            )
            st.stop()


        # --------------------------------------------------
        # 5. Display Uploaded Dataset
        # --------------------------------------------------

        st.subheader("Uploaded Dataset")

        st.write(
            f"Rows: {len(test_df)} | "
            f"Input features: {len(feature_names)}"
        )

        st.dataframe(test_df.head())


        # --------------------------------------------------
        # 6. Select Classification Model
        # --------------------------------------------------

        selected_model = st.selectbox(
            "Select a classification model",
            list(model_files.keys())
        )

        model = models[selected_model]


        # --------------------------------------------------
        # 7. Apply Preprocessing
        # --------------------------------------------------

        if selected_model in scaled_models:

            X_eval = scaler.transform(X_test)

        else:

            X_eval = X_test


        # --------------------------------------------------
        # 8. Model Prediction
        # --------------------------------------------------

        y_pred = model.predict(X_eval)

        y_prob = model.predict_proba(X_eval)[:, 1]


        # --------------------------------------------------
        # 9. Calculate Evaluation Metrics
        # --------------------------------------------------

        metrics = {

            "Accuracy": accuracy_score(y_test, y_pred),

            "AUC": roc_auc_score(y_test, y_prob),

            "Precision": precision_score(
                y_test, y_pred, zero_division=0
            ),

            "Recall": recall_score(
                y_test, y_pred, zero_division=0
            ),

            "F1 Score": f1_score(
                y_test, y_pred, zero_division=0
            ),

            "MCC": matthews_corrcoef(y_test, y_pred)
        }


        # --------------------------------------------------
        # 10. Display Evaluation Results
        # --------------------------------------------------

        st.subheader(f"Evaluation: {selected_model}")

        cols = st.columns(3)

        for i, (metric, value) in enumerate(metrics.items()):

            cols[i % 3].metric(
                metric,
                f"{value:.4f}"
            )


        # --------------------------------------------------
        # 11. Display Confusion Matrix
        # --------------------------------------------------

        st.subheader("Confusion Matrix")

        cm = confusion_matrix(
            y_test,
            y_pred,
            labels=[0, 1]
        )

        fig, ax = plt.subplots(figsize=(5, 4))

        display = ConfusionMatrixDisplay(
            confusion_matrix=cm,
            display_labels=["Benign", "Malignant"]
        )

        display.plot(
            ax=ax,
            cmap="Blues",
            values_format="d"
        )

        st.pyplot(fig)

        plt.close(fig)


        # --------------------------------------------------
        # 12. Information
        # --------------------------------------------------

        st.caption(
            "Target encoding: 0 = benign, 1 = malignant. "
            "The app loads saved models and does not retrain them."
        )


    except Exception as exc:

        st.error(
            f"Unable to evaluate the uploaded CSV: {exc}"
        )


else:

    st.info(
        "Upload test_data.csv to evaluate a trained model."
    )
