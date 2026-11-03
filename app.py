
import streamlit as st
import pandas as pd
import joblib
import os
import matplotlib.pyplot as plt

from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    matthews_corrcoef,
    confusion_matrix,
    ConfusionMatrixDisplay,
    classification_report
)


# --------------------------------------------------
# 1. Page Configuration
# --------------------------------------------------

st.set_page_config(
    page_title="Breast Cancer AI Dashboard",
    page_icon="🔬",
    layout="wide"
)


# --------------------------------------------------
# 2. Custom CSS
# --------------------------------------------------

st.markdown("""
<style>

.stApp {
    background-color: #f5f7fb;
}

.main-title {
    font-size: 32px;
    font-weight: 700;
    color: #172554;
    margin-bottom: 4px;
}

.subtitle {
    font-size: 15px;
    color: #64748b;
    margin-bottom: 25px;
}

.section-title {
    font-size: 21px;
    font-weight: 650;
    color: #1e293b;
    margin-top: 20px;
    margin-bottom: 15px;
}

.metric-card {
    background-color: white;
    padding: 18px;
    border-radius: 12px;
    border: 1px solid #e2e8f0;
    box-shadow: 0 2px 6px rgba(0,0,0,0.04);
}

.metric-label {
    font-size: 13px;
    color: #64748b;
    font-weight: 600;
}

.metric-value {
    font-size: 25px;
    font-weight: 700;
    color: #172554;
    margin-top: 5px;
}

.info-card {
    background-color: white;
    padding: 18px;
    border-radius: 12px;
    border: 1px solid #e2e8f0;
}

div[data-testid="stSidebar"] {
    background-color: #eef2ff;
}

.stButton button {
    border-radius: 8px;
}

</style>
""", unsafe_allow_html=True)


# --------------------------------------------------
# 3. Header
# --------------------------------------------------

st.markdown(
    '<div class="main-title">🔬 Breast Cancer Classification</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">'
    'Machine Learning Model Evaluation Dashboard | '
    'Breast Cancer Wisconsin Diagnostic Dataset'
    '</div>',
    unsafe_allow_html=True
)

st.divider()


# --------------------------------------------------
# 4. Model Configuration
# --------------------------------------------------

MODEL_DIR = "model"

model_files = {
    "Logistic Regression": "logistic_regression.joblib",
    "Decision Tree": "decision_tree.joblib",
    "KNN": "knn.joblib",
    "Naive Bayes": "naive_bayes.joblib",
    "Support Vector Machine": "support_vector_machine.joblib"
}

# Models requiring StandardScaler
scaled_models = {
    "Logistic Regression",
    "KNN",
    "Support Vector Machine"
}


# --------------------------------------------------
# 5. Load Saved Models and Scaler
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
# 6. Sidebar Dashboard Menu
# --------------------------------------------------

with st.sidebar:

    st.title("⚙️ Dashboard Controls")

    st.markdown("---")

    st.subheader("Dashboard Menu")

    menu_option = st.radio(
        "Select View",
        [
            "Single Model Evaluation",
            "Compare All Models"
        ]
    )

    st.markdown("---")

    selected_model = None

    if menu_option == "Single Model Evaluation":

        st.subheader("Model Selection")

        selected_model = st.selectbox(
            "Choose a classifier",
            list(model_files.keys())
        )

        preprocessing = (
            "Standard Scaling"
            if selected_model in scaled_models
            else "Original Features"
        )

        st.info(f"Preprocessing: {preprocessing}")

    st.markdown("---")

    st.subheader("Dataset")

    uploaded_file = st.file_uploader(
        "Upload test CSV",
        type=["csv"]
    )


# --------------------------------------------------
# 7. Initial Screen
# --------------------------------------------------

if uploaded_file is None:

    st.info(
        "Upload your test CSV from the sidebar to begin."
    )

    st.markdown("### About this Project")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("""
        <div class="info-card">
            <h3>5</h3>
            <b>Classification Models</b>
            <p>Five machine learning algorithms are compared.</p>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown("""
        <div class="info-card">
            <h3>30</h3>
            <b>Input Features</b>
            <p>Cell nucleus measurements are used for classification.</p>
        </div>
        """, unsafe_allow_html=True)

    with col3:
        st.markdown("""
        <div class="info-card">
            <h3>6</h3>
            <b>Evaluation Metrics</b>
            <p>Different metrics are used to evaluate model performance.</p>
        </div>
        """, unsafe_allow_html=True)

    st.stop()


# --------------------------------------------------
# 8. Read and Validate Uploaded Dataset
# --------------------------------------------------

try:

    test_df = pd.read_csv(uploaded_file)

    if "diagnosis" not in test_df.columns:
        st.error("The CSV must contain a 'diagnosis' column.")
        st.stop()

    # Get expected feature names from the reference CSV
    reference_columns = pd.read_csv(
        "test_data.csv",
        nrows=0
    ).columns

    feature_names = [
        col for col in reference_columns
        if col != "diagnosis"
    ]

    # Check missing features
    missing = [
        col for col in feature_names
        if col not in test_df.columns
    ]

    if missing:
        st.error(f"Missing required features: {missing}")
        st.stop()

    if test_df.empty:
        st.error("The uploaded CSV is empty.")
        st.stop()

    # Separate input features and target
    X_test = test_df[feature_names]
    y_test = test_df["diagnosis"]

    # Check missing values
    if X_test.isnull().any().any() or y_test.isnull().any():
        st.error("The dataset contains missing values.")
        st.stop()

    # Check numeric input features
    if not all(
        pd.api.types.is_numeric_dtype(X_test[col])
        for col in feature_names
    ):
        st.error("All input features must be numeric.")
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
            "The test dataset must contain both target classes "
            "to calculate AUC."
        )
        st.stop()


    # --------------------------------------------------
    # 9. Dataset Overview
    # --------------------------------------------------

    st.markdown(
        '<div class="section-title">📊 Dataset Overview</div>',
        unsafe_allow_html=True
    )

    c1, c2, c3, c4 = st.columns(4)

    c1.metric("Total Samples", len(test_df))
    c2.metric("Input Features", len(feature_names))
    c3.metric("Benign Samples", int((y_test == 0).sum()))
    c4.metric("Malignant Samples", int((y_test == 1).sum()))


    # --------------------------------------------------
    # 10. Compare All Models
    # --------------------------------------------------

    if menu_option == "Compare All Models":

        st.markdown(
            '<div class="section-title">'
            '📊 Comparison of All Classification Models'
            '</div>',
            unsafe_allow_html=True
        )

        comparison_results = []

        # Evaluate all five saved models
        for model_name, model in models.items():

            if model_name in scaled_models:
                X_eval = scaler.transform(X_test)
            else:
                X_eval = X_test

            y_pred = model.predict(X_eval)

            y_prob = model.predict_proba(X_eval)[:, 1]

            results = {
                "Model": model_name,
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

            comparison_results.append(results)

        comparison_df = pd.DataFrame(comparison_results)

        # Sort by accuracy
        comparison_df = comparison_df.sort_values(
            by="Accuracy",
            ascending=False
        ).reset_index(drop=True)

        # Performance table
        st.markdown("### Model Performance Summary")

        st.dataframe(
            comparison_df.style.format({
                "Accuracy": "{:.4f}",
                "AUC": "{:.4f}",
                "Precision": "{:.4f}",
                "Recall": "{:.4f}",
                "F1 Score": "{:.4f}",
                "MCC": "{:.4f}"
            }),
            use_container_width=True,
            hide_index=True
        )

        # Best model by accuracy
        best_model = comparison_df.iloc[0]

        st.success(
            f"Best model by accuracy: {best_model['Model']} "
            f"({best_model['Accuracy']:.4f})"
        )

        # Best model for individual metrics
        st.markdown("### Best Performance by Metric")

        metric_names = [
            "Accuracy",
            "AUC",
            "Precision",
            "Recall",
            "F1 Score",
            "MCC"
        ]

        best_cols = st.columns(3)

        for i, metric in enumerate(metric_names):

            best_index = comparison_df[metric].idxmax()
            best_row = comparison_df.loc[best_index]

            best_cols[i % 3].metric(
                metric,
                f"{best_row[metric]:.4f}",
                best_row["Model"]
            )

        # Performance comparison chart
        st.markdown("### Performance Comparison")

        chart_df = comparison_df.set_index("Model")[
            ["Accuracy", "AUC", "F1 Score"]
        ]

        st.bar_chart(
            chart_df,
            use_container_width=True
        )

        # Complete metrics comparison chart
        with st.expander("View all six metrics"):

            all_metrics = comparison_df.set_index("Model")[
                metric_names
            ]

            st.bar_chart(
                all_metrics,
                use_container_width=True
            )

        st.caption(
            "All five models are evaluated using the same uploaded "
            "test dataset and their respective preprocessing methods. "
            "No model retraining is performed."
        )

        st.stop()


    # --------------------------------------------------
    # 11. Single Model Evaluation
    # --------------------------------------------------

    st.markdown(
        '<div class="section-title">🎯 Model Performance</div>',
        unsafe_allow_html=True
    )

    st.caption(f"Selected classifier: {selected_model}")

    model = models[selected_model]

    # Apply model-specific preprocessing
    if selected_model in scaled_models:
        X_eval = scaler.transform(X_test)
    else:
        X_eval = X_test

    # Predictions
    y_pred = model.predict(X_eval)

    y_prob = model.predict_proba(X_eval)[:, 1]

    # Calculate metrics
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

    # Display metric cards
    metric_cols = st.columns(6)

    for i, (name, value) in enumerate(metrics.items()):

        metric_cols[i].markdown(
            f"""
            <div class="metric-card">
                <div class="metric-label">{name}</div>
                <div class="metric-value">{value:.4f}</div>
            </div>
            """,
            unsafe_allow_html=True
        )


    # --------------------------------------------------
    # 12. Confusion Matrix and Classification Report
    # --------------------------------------------------

    st.markdown(
        '<div class="section-title">'
        '📈 Model Evaluation Details'
        '</div>',
        unsafe_allow_html=True
    )

    tab1, tab2 = st.tabs([
        "Confusion Matrix",
        "Classification Report"
    ])

    cm = confusion_matrix(
        y_test,
        y_pred,
        labels=[0, 1]
    )

    with tab1:

        col1, col2 = st.columns([1.2, 1])

        with col1:

            fig, ax = plt.subplots(figsize=(6, 4))

            display = ConfusionMatrixDisplay(
                confusion_matrix=cm,
                display_labels=["Benign", "Malignant"]
            )

            display.plot(
                ax=ax,
                cmap="Blues",
                values_format="d",
                colorbar=False
            )

            ax.set_title("Confusion Matrix")

            fig.tight_layout()

            st.pyplot(fig)

            plt.close(fig)

        with col2:

            tn, fp, fn, tp = cm.ravel()

            st.markdown("#### Prediction Summary")

            st.metric("True Negatives", int(tn))
            st.metric("False Positives", int(fp))
            st.metric("False Negatives", int(fn))
            st.metric("True Positives", int(tp))


    with tab2:

        report = classification_report(
            y_test,
            y_pred,
            target_names=["Benign", "Malignant"],
            output_dict=True,
            zero_division=0
        )

        report_df = pd.DataFrame(report).transpose()

        st.dataframe(
            report_df.round(4),
            use_container_width=True
        )


    # --------------------------------------------------
    # 13. Dataset Preview
    # --------------------------------------------------

    st.markdown(
        '<div class="section-title">📄 Dataset Preview</div>',
        unsafe_allow_html=True
    )

    with st.expander("View uploaded test data"):

        st.dataframe(
            test_df,
            use_container_width=True
        )


    # --------------------------------------------------
    # 14. Footer
    # --------------------------------------------------

    st.divider()

    st.caption(
        "Target encoding: 0 = Benign | 1 = Malignant. "
        "All models are loaded from saved artifacts. "
        "No retraining is performed by this application."
    )


except Exception as exc:

    st.error(f"Unable to evaluate the uploaded CSV: {exc}")
