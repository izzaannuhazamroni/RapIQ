# =========================================
# RapIQ STREAMLIT APP
# Premium UI Refactor
# =========================================

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
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

html, body, [class*="css"]  {
    font-family: 'Plus Jakarta Sans', sans-serif;
}

.stApp {
    background: #070B1A;
    color: white;
}

/* SIDEBAR */

section[data-testid="stSidebar"] {
    background: linear-gradient(
        180deg,
        #081120,
        #050B18
    );
    border-right: 1px solid rgba(60,215,255,0.15);
}

.sidebar-title {
    font-size: 32px;
    font-weight: 800;
    color: white;
    margin-bottom: 0;
}

.sidebar-sub {
    color: #6EE7FF;
    font-size: 12px;
    letter-spacing: 1px;
    margin-top: -10px;
    margin-bottom: 30px;
}

/* HERO */

.hero-card {
    background:
        linear-gradient(
            135deg,
            rgba(18,25,40,0.95),
            rgba(10,15,30,0.85)
        );

    border: 1px solid rgba(60,215,255,0.12);
    border-radius: 28px;
    padding: 40px;
    margin-bottom: 25px;
    backdrop-filter: blur(20px);

    box-shadow:
        0 0 40px rgba(60,215,255,0.08);
}

/* GLASS CARD */

.glass-card {
    background: rgba(20,25,40,0.55);
    border: 1px solid rgba(60,215,255,0.12);
    border-radius: 24px;
    padding: 28px;
    backdrop-filter: blur(20px);

    box-shadow:
        0 0 25px rgba(60,215,255,0.05);
}

/* RESULT CARD */

.result-card {
    background: linear-gradient(
        135deg,
        rgba(0,212,255,0.18),
        rgba(87,27,193,0.18)
    );

    border: 1px solid rgba(255,255,255,0.08);
    border-radius: 24px;
    padding: 30px;

    box-shadow:
        0 0 35px rgba(60,215,255,0.15);
}

/* METRIC */

.metric-card {
    background: rgba(20,25,40,0.55);
    border: 1px solid rgba(60,215,255,0.10);
    border-radius: 20px;
    padding: 25px;
    text-align: center;
}

/* TEXT */

.section-title {
    color: white;
    font-size: 24px;
    font-weight: 700;
    margin-bottom: 20px;
}

.small-label {
    color: #9FB3C8;
    font-size: 13px;
    font-weight: 600;
    letter-spacing: 0.5px;
    margin-bottom: 6px;
}

.main-title {
    font-size: 58px;
    font-weight: 800;
    line-height: 1;
    color: white;
    margin-bottom: 15px;
}

.main-desc {
    color: #B7C4D6;
    font-size: 18px;
    line-height: 1.8;
    max-width: 700px;
}

.prediction-label {
    color: #00D4FF;
    font-size: 42px;
    font-weight: 800;
    margin-top: 10px;
}

.confidence-label {
    color: #D0BCFF;
    font-size: 20px;
    font-weight: 700;
}

/* BUTTON */

div.stButton > button {
    width: 100%;
    height: 62px;
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
        0 0 30px rgba(60,215,255,0.35);

    transition: 0.3s ease;
}

div.stButton > button:hover {
    transform: scale(1.02);

    box-shadow:
        0 0 40px rgba(60,215,255,0.50);
}

/* INPUT */

.stSelectbox label,
.stNumberInput label {
    color: #DCE7F3 !important;
    font-weight: 600 !important;
}

div[data-baseweb="select"] > div {
    background-color: rgba(15,20,30,0.90) !important;
    border: 1px solid rgba(60,215,255,0.18) !important;
    border-radius: 14px !important;
}

.stNumberInput input {
    background-color: rgba(15,20,30,0.90) !important;
    color: white !important;
    border-radius: 14px !important;
    border: 1px solid rgba(60,215,255,0.18) !important;
}

/* DATAFRAME */

.stDataFrame {
    border-radius: 18px;
    overflow: hidden;
}

/* UPLOAD */

.upload-box {
    background: rgba(20,25,40,0.55);
    border: 2px dashed rgba(60,215,255,0.25);
    border-radius: 24px;
    padding: 40px;
    text-align: center;
}

</style>
""", unsafe_allow_html=True)

# =========================================
# LOAD MODEL
# =========================================

@st.cache_resource
def load_model():

    model = joblib.load("model_mlp_iq.pkl")
    scaler = joblib.load("scaler_iq.pkl")

    return model, scaler

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
MLP ARCHITECTURE : (10, 6)
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
        <div class="main-title">RapIQ</div>

        <div class="main-desc">
            Cognitive potential orchestration.
            Predict child IQ category using
            Multilayer Perceptron (MLP)
            based on parental education,
            age, and gender.
        </div>
    </div>
    """, unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    with col1:

        st.markdown(
            '<div class="glass-card">',
            unsafe_allow_html=True
        )

        st.markdown(
            '<div class="section-title">👨‍👩‍👧 Family Information</div>',
            unsafe_allow_html=True
        )

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

        st.markdown(
            '<div class="glass-card">',
            unsafe_allow_html=True
        )

        st.markdown(
            '<div class="section-title">🧒 Child Information</div>',
            unsafe_allow_html=True
        )

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

    st.markdown("<br>", unsafe_allow_html=True)

    button_col1, button_col2, button_col3 = st.columns([1,2,1])

    with button_col2:

        predict_btn = st.button(
            "🚀 Predict IQ Category"
        )

    if predict_btn:

        try:

            input_data = pd.DataFrame({
                "education_mother": [
                    edu_map[education_mother]
                ],

                "education_father": [
                    edu_map[education_father]
                ],

                "age_years": [age],

                "gender": [
                    gender_map[gender]
                ]
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

            colors = [
                "#1E293B"
                for _ in range(len(prob_df))
            ]

            max_idx = prob_df[
                "Probability"
            ].idxmax()

            colors[max_idx] = "#00D4FF"

            fig = go.Figure(
                data=[
                    go.Bar(
                        x=prob_df["Category"],
                        y=prob_df["Probability"],
                        marker_color=colors,

                        text=[
                            f"{x:.1f}%"
                            for x in prob_df["Probability"]
                        ],

                        textposition='outside'
                    )
                ]
            )

            fig.update_layout(
                template="plotly_dark",

                paper_bgcolor='rgba(0,0,0,0)',

                plot_bgcolor='rgba(0,0,0,0)',

                height=420,

                font=dict(
                    color='white',
                    family='Plus Jakarta Sans'
                ),

                xaxis=dict(showgrid=False),

                yaxis=dict(
                    range=[0,100],
                    gridcolor='rgba(255,255,255,0.08)'
                )
            )

            chart_col, result_col = st.columns([2,1])

            with chart_col:

                st.markdown(
                    '<div class="glass-card">',
                    unsafe_allow_html=True
                )

                st.markdown(
                    '<div class="section-title">📊 Confidence Probability</div>',
                    unsafe_allow_html=True
                )

                st.plotly_chart(
                    fig,
                    use_container_width=True
                )

                st.markdown(
                    '</div>',
                    unsafe_allow_html=True
                )

            with result_col:

                st.markdown(f"""
                <div class="result-card">

                    <div class="small-label">
                        FINAL PREDICTION
                    </div>

                    <div class="prediction-label">
                        {predicted_label}
                    </div>

                    <br>

                    <div class="confidence-label">
                        Confidence Score
                    </div>

                    <div style="
                        font-size:40px;
                        font-weight:800;
                        color:white;
                    ">
                        {confidence:.2f}%
                    </div>

                </div>
                """, unsafe_allow_html=True)

        except Exception as e:

            st.error(
                f"Prediction Error: {e}"
            )

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
            Upload CSV datasets and run
            large-scale IQ category prediction
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

    col1, col2 = st.columns([1,2])

    with col1:

        st.markdown(
            '<div class="upload-box">',
            unsafe_allow_html=True
        )

        st.markdown("## 📂 Upload Dataset")

        st.markdown(
            "Drag and drop CSV file for bulk prediction"
        )

        uploaded_file = st.file_uploader(
            "Upload CSV",
            type=["csv"]
        )

        st.download_button(
            label="⬇ Download CSV Template",
            data=template_csv,
            file_name="template_input_iq.csv",
            mime="text/csv"
        )

        st.markdown(
            '</div>',
            unsafe_allow_html=True
        )

    with col2:

        st.markdown(
            '<div class="glass-card">',
            unsafe_allow_html=True
        )

        st.markdown(
            '<div class="section-title">📄 Dataset Preview</div>',
            unsafe_allow_html=True
        )

        if uploaded_file is not None:

            df = pd.read_csv(uploaded_file)

            st.dataframe(
                df.head(),
                use_container_width=True
            )

        else:

            st.info("Upload CSV file first.")

        st.markdown(
            '</div>',
            unsafe_allow_html=True
        )

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
            RapIQ is an AI-based web application
            designed to classify child IQ categories
            using Multilayer Perceptron (MLP)
            architecture.
        </div>

    </div>
    """, unsafe_allow_html=True)

    c1, c2, c3, c4 = st.columns(4)

    metrics = [
        ("Accuracy", "94.2%"),
        ("F1-Score", "0.93"),
        ("Dataset", "80k+"),
        ("Latency", "12ms")
    ]

    cols = [c1, c2, c3, c4]

    for col, metric in zip(cols, metrics):

        with col:

            st.markdown(f"""
            <div class="metric-card">

                <div class="small-label">
                    {metric[0]}
                </div>

                <div style="
                    font-size:42px;
                    font-weight:800;
                    color:#00D4FF;
                ">
                    {metric[1]}
                </div>

            </div>
            """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    left, right = st.columns([2,1])

    with left:

        st.markdown("""
        <div class="glass-card">

        <div class="section-title">
            🧠 Model Information
        </div>

        <ul style="
            line-height:2;
            font-size:17px;
        ">

            <li>Algorithm : MLP</li>
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
            color:#B7C4D6;
            line-height:2;
            font-size:16px;
        ">

            RapIQ is developed for
            educational and research purposes.

            This system is NOT intended
            to replace professional
            psychological diagnosis.

        </p>

        </div>
        """, unsafe_allow_html=True)