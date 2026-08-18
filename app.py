import os
import json
import joblib
import pandas as pd
import streamlit as st
import plotly.express as px


# =========================================================
# CONFIGURATION
# =========================================================

MODEL_PATH = "models/fraud_detection_pipeline.joblib"
METRICS_PATH = "artifacts/metrics.json"
IMPORTANCE_PATH = "artifacts/feature_importance.csv"
DATA_PATH = "data/creditcard.csv"


st.set_page_config(
    page_title="Credit Card Fraud Detection",
    page_icon="💳",
    layout="wide",
    initial_sidebar_state="expanded",
)


# =========================================================
# CUSTOM CSS
# =========================================================



st.markdown(
    """
    <style>

    [data-testid="stAppViewContainer"] {
        background: #f6f8fb;
    }

    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #081a31 0%, #102947 100%);
    }

    [data-testid="stSidebar"] * {
        color: #ffffff !important;
    }

    .block-container {
        max-width: 1500px;
        padding-top: 1.4rem;
        padding-bottom: 2rem;
    }

    div[data-testid="stNumberInput"] label,
    div[data-testid="stNumberInput"] label p {
        color: #000000 !important;
    }

    .stButton > button,
    div[data-testid="stFormSubmitButton"] button,
    div[data-testid="stDownloadButton"] button {
        background-color: #2563eb !important;
        color: #ffffff !important;
        border: none !important;
        font-weight: 600 !important;
        transition: none !important;
        box-shadow: none !important;
    }

    .stButton > button:hover,
    .stButton > button:focus,
    .stButton > button:active,
    div[data-testid="stFormSubmitButton"] button:hover,
    div[data-testid="stFormSubmitButton"] button:focus,
    div[data-testid="stFormSubmitButton"] button:active,
    div[data-testid="stDownloadButton"] button:hover,
    div[data-testid="stDownloadButton"] button:focus,
    div[data-testid="stDownloadButton"] button:active {
        background-color: #2563eb !important;
        color: #ffffff !important;
        border: none !important;
        transition: none !important;
        box-shadow: none !important;
    }

    .hero {
        background: linear-gradient(135deg, #061a38 0%, #0d3f84 60%, #1266d6 100%);
        color: #ffffff;
        border-radius: 22px;
        padding: 28px 34px;
        margin-bottom: 22px;
        box-shadow: 0 10px 30px rgba(7, 35, 75, 0.18);
    }

    .hero-title {
        color: #ffffff;
        font-size: 38px;
        font-weight: 800;
        margin: 0;
    }

    .hero-sub {
        color: #dbeafe;
        font-size: 16px;
        margin-top: 7px;
    }

    .hero-badge {
        color: #ffffff;
        float: right;
        font-size: 15px;
        font-weight: 700;
        padding-top: 8px;
    }

    .card {
        background: #87ceeb;
        color: #000000;
        border: 1px solid #e5eaf1;
        border-radius: 16px;
        padding: 18px 20px;
        box-shadow: 0 3px 12px rgba(15, 23, 42, 0.05);
    }

    .card-label {
        color: #000000;
        font-size: 14px;
        font-weight: 600;
    }

    .card-value {
        color: #000000;
        font-size: 29px;
        font-weight: 800;
        margin-top: 6px;
    }

    .card-note {
        color: #344054;
        font-size: 12px;
        margin-top: 4px;
    }

    .section-title {
        color: #000000 !important;
        font-size: 22px;
        font-weight: 800;
        margin: 22px 0 12px;
    }

    .info-box {
        background: #87ceeb;
        color: #000000;
        padding: 12px 16px;
        border-radius: 8px;
        font-size: 14px;
        border-left: 4px solid #2563eb;
        margin-bottom: 15px;
    }

    .result-fraud {
        background: #fff1f3;
        color: #000000;
        border: 1px solid #fecdca;
        border-radius: 16px;
        padding: 22px;
    }

    .result-safe {
        background: #dbeafe;
        color: #000000;
        border: 1px solid #93c5fd;
        border-radius: 16px;
        padding: 22px;
    }

    .result-fraud h2,
    .result-fraud p,
    .result-safe h2,
    .result-safe p {
        color: #000000 !important;
    }

    .about-card {
        background: #87ceeb;
        color: #000000;
        border: 1px solid #e5eaf1;
        border-radius: 16px;
        padding: 22px;
        min-height: 350px;
        box-shadow: 0 3px 12px rgba(15, 23, 42, 0.05);
    }

    .about-card h2,
    .about-card h3,
    .about-card p,
    .about-card b,
    .about-card li,
    .about-card strong {
        color: #000000 !important;
    }

    .about-card h2 {
        margin-top: 0;
    }

    .about-card h3 {
        margin-bottom: 8px;
    }

    .footer {
        color: #667085;
        font-size: 12px;
        text-align: center;
        padding: 22px 0 5px;
    }

    </style>
    """,
    unsafe_allow_html=True,
)


# =========================================================
# LOAD MODEL
# =========================================================

if not os.path.exists(MODEL_PATH):
    st.error("Model not found. Run `python src/train.py` first.")
    st.stop()

model = joblib.load(MODEL_PATH)

feature_names = list(model.feature_names_in_)


# =========================================================
# LOAD METRICS
# =========================================================

metrics = {}

if os.path.exists(METRICS_PATH):
    with open(METRICS_PATH, encoding="utf-8") as file:
        metrics = json.load(file)


# =========================================================
# DATASET SUMMARY
# =========================================================

total = 0
fraud = 0
legit = 0
fraud_pct = 0

if os.path.exists(DATA_PATH):

    try:
        data = pd.read_csv(
            DATA_PATH,
            usecols=["Class"]
        )

        total = len(data)
        fraud = int((data["Class"] == 1).sum())
        legit = int((data["Class"] == 0).sum())

        if total > 0:
            fraud_pct = fraud / total * 100

    except Exception:
        total = int(metrics.get("test_rows", 0))
        fraud = int(metrics.get("fraud_test_rows", 0))
        legit = max(total - fraud, 0)

        if total > 0:
            fraud_pct = fraud / total * 100


# =========================================================
# HELPER FUNCTION
# =========================================================

def card(label, value, note=""):
    st.markdown(
        f"""
        <div class="card">
            <div class="card-label">{label}</div>
            <div class="card-value">{value}</div>
            <div class="card-note">{note}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# =========================================================
# SIDEBAR
# =========================================================

st.sidebar.markdown("# 💳")
st.sidebar.markdown("## Credit Card Fraud Detection")
st.sidebar.caption(
    "Machine Learning Powered Fraud Detection System"
)

st.sidebar.markdown("---")

page = st.sidebar.radio(
    "NAVIGATION",
    [
        "Dashboard",
        "Predict Transaction",
        "Batch Prediction",
        "About Model",
    ],
)

st.sidebar.markdown("### DATASET SUMMARY")

st.sidebar.metric(
    "Total Transactions",
    f"{total:,}"
)

st.sidebar.metric(
    "Fraudulent Transactions",
    f"{fraud:,}"
)

st.sidebar.metric(
    "Legitimate Transactions",
    f"{legit:,}"
)

st.sidebar.metric(
    "Fraud Percentage",
    f"{fraud_pct:.3f}%"
)

st.sidebar.markdown("---")

st.sidebar.caption("💳 Credit Card Fraud Detection")
st.sidebar.caption("© 2026 ML Portfolio Project")


# =========================================================
# HERO SECTION
# =========================================================

st.markdown('''
<div class="hero">
  <div class="hero-badge">🛡️ Detect Fraud.<br/>Protect Transactions.</div>
  <div class="hero-title">CREDIT CARD FRAUD DETECTION</div>
  <div class="hero-sub">Machine Learning Powered Fraud Detection System</div>
</div>
''', unsafe_allow_html=True)



# =========================================================
# DASHBOARD
# =========================================================

if page == "Dashboard":

    st.markdown(
        '<div class="section-title">'
        'Welcome to Credit Card Fraud Detection System'
        '</div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        <div class="card">
            <b>
                This system uses Machine Learning to detect
                potentially fraudulent credit card transactions.
            </b>
            <br>
            <span class="card-note">
                The trained Random Forest model returns a fraud
                class and probability score for each transaction.
            </span>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # ---------------- Model Performance ----------------

    st.markdown(
        '<div class="section-title">Model Performance</div>',
        unsafe_allow_html=True,
    )

    cols = st.columns(5)

    accuracy = metrics.get("accuracy")

    performance = [
        (
            "Accuracy",
            f"{accuracy:.2%}"
            if accuracy is not None
            else "—",
            "Overall correctness",
        ),
        (
            "Precision",
            f"{metrics.get('precision', 0):.2f}",
            "Positive prediction quality",
        ),
        (
            "Recall",
            f"{metrics.get('recall', 0):.2f}",
            "Fraud detection rate",
        ),
        (
            "F1 Score",
            f"{metrics.get('f1', 0):.2f}",
            "Precision/recall balance",
        ),
        (
            "ROC-AUC",
            f"{metrics.get('roc_auc', 0):.2f}",
            "Ranking performance",
        ),
    ]

    for col, values in zip(cols, performance):

        with col:
            card(*values)

    # ---------------- Charts ----------------

    left, right = st.columns(2)

    with left:

        st.markdown(
            '<div class="section-title">'
            'Confusion Matrix'
            '</div>',
            unsafe_allow_html=True,
        )

        path = "artifacts/confusion_matrix.png"

        if os.path.exists(path):
            st.image(
                path,
                use_container_width=True
            )
        else:
            st.info(
                "Run training to generate the confusion matrix."
            )

    with right:

        st.markdown(
            '<div class="section-title">'
            'ROC Curve'
            '</div>',
            unsafe_allow_html=True,
        )

        path = "artifacts/roc_curve.png"

        if os.path.exists(path):
            st.image(
                path,
                use_container_width=True
            )
        else:
            st.info(
                "Run training to generate the ROC curve."
            )

    # ---------------- Feature Importance ----------------

    if os.path.exists(IMPORTANCE_PATH):

        st.markdown(
            '<div class="section-title">'
            'Feature Importance'
            '</div>',
            unsafe_allow_html=True,
        )

        importance = pd.read_csv(
            IMPORTANCE_PATH
        ).head(10)

        fig = px.bar(
            importance.sort_values("importance"),
            x="importance",
            y="feature",
            orientation="h",
            text_auto=".3f",
        )

        fig.update_traces(
            textfont=dict(
                color="black",
                size=12,
            )
        )

        fig.update_layout(
            xaxis=dict(
                title=dict(
                    text="Importance",
                    font=dict(
                        color="black",
                        size=14,
                    ),
                ),
                tickfont=dict(
                    color="black",
                    size=12,
                ),
            ),

            yaxis=dict(
                title=dict(
                    text="Feature",
                    font=dict(
                        color="black",
                        size=14,
                    ),
                ),
                tickfont=dict(
                    color="black",
                    size=12,
                ),
            ),

            title=dict(
                text="Top 10 Important Features",
                font=dict(
                    color="black",
                    size=20,
                ),
            ),

            plot_bgcolor="white",
            paper_bgcolor="white",
        )

        st.plotly_chart(
            fig,
            use_container_width=True,
        )


# =========================================================
# PREDICT TRANSACTION
# =========================================================

elif page == "Predict Transaction":

    st.markdown(
        '<div class="section-title">'
        '🔎 Predict Transaction'
        '</div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        <div style="
            background-color: skyblue;
            color: #344054;
            padding: 12px 16px;
            border-radius: 8px;
            font-size: 14px;
            border-left: 4px solid #2563EB;
        ">
            Enter transaction feature values and let the trained
            model estimate fraud probability.
        </div>
        """,
        unsafe_allow_html=True,
    )

    with st.form("prediction_form"):

        values = {}

        cols = st.columns(3)

        for i, feature in enumerate(feature_names):

            default = (
                100.0
                if feature == "Amount"
                else 0.0
            )

            with cols[i % 3]:

                values[feature] = st.number_input(
                    feature,
                    value=float(default),
                    format="%.6f",
                )

        submitted = st.form_submit_button(
            "🚨 Analyze Transaction",
            use_container_width=True,
        )

    if submitted:

        row = pd.DataFrame([values])

        prediction = int(
            model.predict(row)[0]
        )

        probability = float(
            model.predict_proba(row)[0, 1]
        )

        if prediction == 1:

            st.markdown(
                f"""
                <div class="result-fraud">
                    <h2>🚨 FRAUDULENT TRANSACTION</h2>
                    <p>
                        Fraud probability:
                        <b>{probability:.2%}</b>
                    </p>
                </div>
                """,
                unsafe_allow_html=True,
            )

        else:

            st.markdown(
                f"""
                <div class="result-safe">
                    <h2>✅ LEGITIMATE TRANSACTION</h2>
                    <p>
                        Fraud probability:
                        <b>{probability:.2%}</b>
                    </p>
                </div>
                """,
                unsafe_allow_html=True,
            )

        st.progress(
            min(
                max(probability, 0.0),
                1.0,
            )
        )


# =========================================================
# BATCH PREDICTION
# =========================================================

elif page == "Batch Prediction":

    st.markdown(
        '<div class="section-title">'
        '📁 Batch Prediction'
        '</div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        <div style="
            background-color: skyblue;
            color: #344054;
            padding: 12px 16px;
            border-radius: 8px;
            font-size: 14px;
            border-left: 4px solid #2563EB;
        ">
            Upload a CSV containing the same feature columns
            used during training.
        </div>
        """,
        unsafe_allow_html=True,
    )

    uploaded = st.file_uploader(
        "Upload CSV",
        type=["csv"],
    )

    if uploaded:

        df = pd.read_csv(uploaded)

        st.dataframe(
            df.head(10),
            use_container_width=True,
        )

        missing = [
            column
            for column in feature_names
            if column not in df.columns
        ]

        if missing:

            st.error(
                f"Missing columns: {missing}"
            )

        else:

            run_prediction = st.button(
                "🚨 Run Batch Prediction",
                use_container_width=True,
            )

            if run_prediction:

                result = df.copy()

                result["Fraud_Prediction"] = (
                    model.predict(
                        df[feature_names]
                    )
                )

                result["Fraud_Probability"] = (
                    model.predict_proba(
                        df[feature_names]
                    )[:, 1]
                )

                st.success(
                    "Prediction completed successfully."
                )

                st.dataframe(
                    result,
                    use_container_width=True,
                )

                csv_data = result.to_csv(
                    index=False
                ).encode("utf-8")

                st.download_button(
                    "⬇️ Download Results",
                    csv_data,
                    "fraud_predictions.csv",
                    "text/csv",
                    use_container_width=True,
                )

# =========================================================
# ABOUT MODEL
# =========================================================

else:

    st.markdown(
        '<div class="section-title">ℹ️ About Model</div>',
        unsafe_allow_html=True,
    )

    left, right = st.columns(2)

    with left:
        st.markdown(
            '''<div class="card"><h2>Random Forest Classifier</h2><p>This project uses a Random Forest classifier with preprocessing and class-weight balancing to handle the highly imbalanced fraud dataset.</p><h3>Evaluation</h3><ul><li>Accuracy</li><li>Precision</li><li>Recall</li><li>F1 Score</li><li>ROC-AUC</li><li>Confusion Matrix</li></ul></div>''',
            unsafe_allow_html=True,
        )

    with right:
        st.markdown(
            f'''<div class="card"><b>Algorithm</b><p>Random Forest</p><b>Features</b><p>{len(feature_names)} ({', '.join(feature_names[:3])}, ...)</p><b>Classification</b><p>Binary: Legitimate / Fraud</p><b>Dataset</b><p>{total:,} transactions when dataset is available</p></div>''',
            unsafe_allow_html=True,
        )

    st.markdown(
        """
        <div class="info-box">
            Educational portfolio project only.
            Production fraud systems require secure data handling,
            monitoring, drift detection, threshold tuning,
            and human review.
        </div>
        """,
        unsafe_allow_html=True,
    )


# =========================================================
# FOOTER
# =========================================================

st.markdown(
    """
    <div class="footer">
        Credit Card Fraud Detection •
        Machine Learning Portfolio Project
    </div>
    """,
    unsafe_allow_html=True,
)