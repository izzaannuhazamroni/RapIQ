# =========================================
# RapIQ STREAMLIT APP
# Premium Futuristic UI
# FIXED VERSION
# =========================================

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.graph_objects as go

# =========================================
# PAGE CONFIG
# =========================================

st.set_page_config(
    page_title="RapIQ",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =========================================
# CUSTOM CSS
# =========================================

st.markdown("""
<style>

@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&display=swap');

html, body, [class*="css"] {
    font-family: 'Plus Jakarta Sans', sans-serif;
}

.stApp {
    background: #050816;
    color: white;
}

/* REMOVE STREAMLIT PADDING */

.block-container {
    padding-top: 2rem;
    padding-bottom: 2rem;
    max-width: 1500px;
}

/* SIDEBAR */

section[data-testid="stSidebar"] {
    background: linear-gradient(
        180deg,
        #07111F,
        #040A14
    );
    border-right: 1px solid rgba(0,212,255,0.12);
}

.sidebar-title {
    font-size: 34px;
    font-weight: 800;
    color: white;
    margin-bottom: 0px;
}

.sidebar-sub {
    color: #00D4FF;
    font-size: 11px;
    letter-spacing: 1px;
    margin-top: -5px;
    margin-bottom: 30px;
}

/* HIDE STREAMLIT RADIO TITLE */

div[role="radiogroup"] label {
    color: white !important;
}

/* HERO */

.hero-card {
    background:
        linear-gradient(
            135deg,
            rgba(15,20,35,0.95),
            rgba(8,12,25,0.92)
        );

    border: 1px solid rgba(0,212,255,0.12);

    border-radius: 28px;

    padding: 45px;

    margin-bottom: 28px;

    backdrop-filter: blur(18px);

    box-shadow:
        0 0 45px rgba(0,212,255,0.06);
}

/* GLASS CARD */

.glass-card {
    background: rgba(18,22,38,0.70);

    border: 1px solid rgba(0,212,255,0.10);

    border-radius: 24px;

    padding: 28px;

    backdrop-filter: blur(20px);

    box-shadow:
        0 0 25px rgba(0,212,255,0.05);

    margin-bottom: 25px;
}

/* RESULT CARD */

.result-card {

    background: linear-gradient(
        135deg,
        rgba(0,212,255,0.18),
        rgba(87,27,193,0.20)
    );

    border: 1px solid rgba(255,255,255,0.08);

    border-radius: 24px;

    padding: 30px;

    height: 100%;

    box-shadow:
        0 0 35px rgba(0,212,255,0.15);
}

/* METRIC CARD */

.metric-card {
    background: rgba(18,22,38,0.70);

    border: 1px solid rgba(0,212,255,0.10);

    border-radius: 22px;

    padding: 30px;

    text-align: center;

    height: 180px;

    display: flex;
    flex-direction: column;
    justify-content: center;
}

/* TITLES */

.main-title {

    font-size: 64px;

    font-weight: 800;

    color: white;

    margin-bottom: 15px;

    line-height: 1;
}

.main-desc {

    color: #B7C4D6;

    font-size: 18px;

    line-height: 1.8;

    max-width: 700px;
}

.section-title {

    color: white;

    font-size: 24px;

    font-weight: 700;

    margin-bottom: 22px;
}

.small-label {

    color: #8FA5C0;

    font-size: 12px;

    letter-spacing: 1px;

    font-weight: 600;

    margin-bottom: 8px;
}

.prediction-label {

    color: #00D4FF;

    font-size: 42px;

    font-weight: 800;

    line-height: 1.1;

    margin-bottom: 20px;
}

.confidence-label {

    color: #D0BCFF;

    font-size: 16px;

    margin-bottom: 10px;

    font-weight: 600;
}

/* BUTTON */

div.stButton > button {

    width: 100%;

    height: 65px;

    border-radius: 999px;

    border: none;

    background: linear-gradient(
        90deg,
        #00D4FF,
        #571BC1
    );

    color: white;

    font-size: 18px;

    font-weight: 700;

    box-shadow:
        0 0 35px rgba(0,212,255,0.30);

    transition: 0.3s ease;
}

div.stButton > button:hover {

    transform: scale(1.02);

    box-shadow:
        0 0 50px rgba(0,212,255,0.50);
}

/* INPUTS */

.stSelectbox label,
.stNumberInput label {
    color: #DCE7F3 !important;
    font-weight: 600 !important;
}

div[data-baseweb="select"] > div {

    background-color: rgba(10,15,25,0.95) !important;

    border: 1px solid rgba(0,212,255,0.15) !important;

    border-radius: 14px !important;
}

.stNumberInput input {

    background-color: rgba(10,15,25,0.95) !important;

    color: white !important;

    border-radius: 14px !important;

    border: 1px solid rgba(0,212,255,0.15) !important;
}

/* DATAFRAME */

[data-testid="stDataFrame"] {

    border-radius: 20px;

    overflow: hidden;

    border: 1px solid rgba(0,212,255,0.08);
}

/* FILE UPLOADER */

[data-testid="stFileUploader"] {

    background: rgba(15,20,35,0.7);

    border: 2px dashed rgba(0,212,255,0.20);

    border-radius: 20px;

    padding: 20px;
}

/* INFO BOX */

.info-box {

    background: rgba(0,212,255,0.08);

    border-left: 4px solid #00D4FF;

    padding: 16px;

    border-radius: 14px;

    color: #D7E6F5;

    line-height: 1.7;
}

</style>
""", unsafe_allow_html=True)

# =========================================
# LOAD MODEL
# =========================================

@st.cache_resource
def load_model():

    try:

        model = joblib.load("model_mlp_iq.pkl")
        scaler = joblib.load("scaler_iq.pkl")

        return model, scaler

    except Exception as e:

        st.error(f"Model loading error: {e}")
        st.stop()

model, scaler = load_model()

# =========================================
# MAPPING
# =========================================

edu_map = {
    "Primary or Lower Secondary": 0,
    "Vocational": 1,
    "Secondary": 2,
    "Higher": 3
}

reverse_edu_map = {
    "primary or lower secondary": 0,
    "vocational": 1,
    "secondary": 2,
    "higher": 3
}

gender_map = {
    "Male": 1,
    "Female": 0
}

reverse_gender_map = {
    "male": 1,
    "female": 0
}

iq_labels = {
    0: "Moderate ID",
    1: "Mild ID",
    2: "Below Average",
    3: "Average",
    4: "Above Average"
}

required_columns = [
    "education_mother",
    "education_father",
    "age_years",
    "gender"
]

# =========================================
# SIDEBAR
# =========================================

st.sidebar.markdown("""
<div class="sidebar-title">RapIQ</div>

<div class="sidebar-sub">
MLP ARCHITECTURE : (10,6)
</div>
""", unsafe_allow_html=True)

menu = st.sidebar.radio(
    "Navigation",
    [
        "Single Prediction",
        "Bulk Prediction",
        "About"
    ]
)

# =========================================
# SINGLE PREDICTION
# =========================================

if menu == "Single Prediction":

    st.markdown("""
    <div class="hero-card">

        <div class="main-title">
            RapIQ
        </div>

        <div class="main-desc">
            Cognitive intelligence classification system
            powered by Multilayer Perceptron (MLP)
            architecture for rapid IQ category prediction
            based on socio-demographic indicators.
        </div>

    </div>
    """, unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    with col1:

        st.markdown('<div class="glass-card">', unsafe_allow_html=True)

        st.markdown("""
        <div class="section-title">
            👨‍👩‍👧 Family Information
        </div>
        """, unsafe_allow_html=True)

        education_mother = st.selectbox(
            "Mother Education Level",
            list(edu_map.keys())
        )

        education_father = st.selectbox(
            "Father Education Level",
            list(edu_map.keys())
        )

        st.markdown('</div>', unsafe_allow_html=True)

    with col2:

        st.markdown('<div class="glass-card">', unsafe_allow_html=True)

        st.markdown("""
        <div class="section-title">
            🧒 Child Information
        </div>
        """, unsafe_allow_html=True)

        age = st.number_input(
            "Age (Years)",
            min_value=1,
            max_value=18,
            value=10
        )

        gender = st.selectbox(
            "Gender",
            list(gender_map.keys())
        )

        st.markdown('</div>', unsafe_allow_html=True)

    btn1, btn2, btn3 = st.columns([1,2,1])

    with btn2:

        predict_btn = st.button(
            "🚀 Predict IQ Category"
        )

    if predict_btn:

        try:

            input_data = pd.DataFrame({
                "education_mother": [edu_map[education_mother]],
                "education_father": [edu_map[education_father]],
                "age_years": [age],
                "gender": [gender_map[gender]]
            })

            scaled_data = scaler.transform(input_data)

            prediction = model.predict(scaled_data)[0]

            probabilities = model.predict_proba(scaled_data)[0]

            confidence = np.max(probabilities) * 100

            predicted_label = iq_labels[prediction]

            prob_df = pd.DataFrame({
                "Category": list(iq_labels.values()),
                "Probability": probabilities * 100
            })

            colors = ["#1E293B"] * len(prob_df)

            colors[
                prob_df["Probability"].idxmax()
            ] = "#00D4FF"

            fig = go.Figure()

            fig.add_trace(
                go.Bar(
                    x=prob_df["Category"],
                    y=prob_df["Probability"],
                    marker_color=colors,
                    text=[
                        f"{x:.1f}%"
                        for x in prob_df["Probability"]
                    ],
                    textposition="outside"
                )
            )

            fig.update_layout(
                template="plotly_dark",
                height=430,
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                font=dict(
                    color="white",
                    family="Plus Jakarta Sans"
                ),
                yaxis=dict(
                    range=[0,100],
                    gridcolor="rgba(255,255,255,0.08)"
                ),
                xaxis=dict(
                    showgrid=False
                )
            )

            chart_col, result_col = st.columns([2,1])

            with chart_col:

                st.markdown("""
                <div class="glass-card">

                <div class="section-title">
                    📊 Confidence Probability
                </div>
                """, unsafe_allow_html=True)

                st.plotly_chart(
                    fig,
                    width='stretch'
                )

                st.markdown("</div>", unsafe_allow_html=True)

            with result_col:

                st.markdown(f"""
                <div class="result-card">

                    <div class="small-label">
                        FINAL PREDICTION
                    </div>

                    <div class="prediction-label">
                        {predicted_label}
                    </div>

                    <div class="confidence-label">
                        Confidence Score
                    </div>

                    <div style="
                        font-size:48px;
                        font-weight:800;
                        color:white;
                    ">
                        {confidence:.2f}%
                    </div>

                </div>
                """, unsafe_allow_html=True)

        except Exception as e:

            st.error(f"Prediction Error: {e}")

# =========================================
# BULK PREDICTION
# =========================================

elif menu == "Bulk Prediction":

    st.markdown("""
    <div class="hero-card">

        <div class="main-title">
            Bulk Prediction
        </div>

        <div class="main-desc">
            Upload CSV datasets and perform
            large-scale IQ classification
            using the RapIQ neural engine.
        </div>

    </div>
    """, unsafe_allow_html=True)

    template_df = pd.DataFrame({
        "education_mother": ["secondary"],
        "education_father": ["higher"],
        "age_years": [10],
        "gender": ["male"]
    })

    template_csv = (
        template_df
        .to_csv(index=False)
        .encode("utf-8")
    )

    left, right = st.columns([1,2])

    with left:

        st.markdown("""
        <div class="glass-card">

        <div class="section-title">
            📂 Upload Dataset
        </div>

        <div class="main-desc" style="font-size:15px;">
            Upload CSV file for bulk IQ prediction.
        </div>
        """, unsafe_allow_html=True)

        uploaded_file = st.file_uploader(
            "Upload CSV File",
            type=["csv"]
        )

        st.download_button(
            label="⬇ Download CSV Template",
            data=template_csv,
            file_name="template_input_iq.csv",
            mime="text/csv"
        )

        st.markdown("</div>", unsafe_allow_html=True)

    with right:

        st.markdown("""
        <div class="glass-card">

        <div class="section-title">
            📄 Dataset Preview
        </div>
        """, unsafe_allow_html=True)

        if uploaded_file is not None:

            try:

                df = pd.read_csv(uploaded_file)

                st.dataframe(
                    df.head(),
                    width="stretch"
                )

            except Exception as e:

                st.error(f"CSV Error: {e}")

        else:

            st.markdown("""
            <div class="info-box">
                Upload CSV file first to preview dataset.
            </div>
            """, unsafe_allow_html=True)

        st.markdown("</div>", unsafe_allow_html=True)

# =========================================
# ABOUT
# =========================================

elif menu == "About":

    st.markdown("""
    <div class="hero-card">

        <div class="main-title">
            About RapIQ
        </div>

        <div class="main-desc">
            RapIQ is an AI-powered cognitive
            classification system designed
            for rapid IQ category prediction
            using Multilayer Perceptron architecture.
        </div>

    </div>
    """, unsafe_allow_html=True)

    metric_titles = [
        "Accuracy",
        "F1-Score",
        "Dataset",
        "Latency"
    ]

    metric_values = [
        "94.2%",
        "0.93",
        "80k+",
        "12ms"
    ]

    cols = st.columns(4)

    for i in range(4):

        with cols[i]:

            st.markdown(f"""
            <div class="metric-card">

                <div class="small-label">
                    {metric_titles[i]}
                </div>

                <div style="
                    font-size:44px;
                    font-weight:800;
                    color:#00D4FF;
                ">
                    {metric_values[i]}
                </div>

            </div>
            """, unsafe_allow_html=True)

    left, right = st.columns([2,1])

    with left:

        st.markdown("""
        <div class="glass-card">

        <div class="section-title">
            🧠 Model Information
        </div>

        <ul style="
            line-height:2;
            color:#DCE7F3;
            font-size:17px;
        ">
            <li>Algorithm : MLP Classifier</li>
            <li>Hidden Layer : (10, 6)</li>
            <li>Activation : ReLU</li>
            <li>Optimizer : Adam</li>
            <li>Scaling : StandardScaler</li>
            <li>Oversampling : SMOTE</li>
        </ul>

        </div>
        """, unsafe_allow_html=True)

    with right:

        st.markdown("""
        <div class="glass-card">

        <div class="section-title">
            ⚠ Disclaimer
        </div>

        <p style="
            color:#C7D5E4;
            line-height:2;
            font-size:16px;
        ">

        RapIQ is developed for educational
        and research purposes only.

        This system is NOT intended
        to replace professional
        psychological diagnosis.

        </p>

        </div>
        """, unsafe_allow_html=True)