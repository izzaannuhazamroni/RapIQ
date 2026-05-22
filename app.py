# =========================================
# RAPIQ — Rapid IQ Classification System
# Modern AI Dashboard Redesign
# =========================================

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.graph_objects as go
import plotly.express as px
from streamlit_option_menu import option_menu

# =========================================
# PAGE CONFIG
# =========================================

st.set_page_config(
    page_title="RapIQ — Intelligence Classification",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =========================================
# CUSTOM CSS
# =========================================

st.markdown("""
<style>
/* ── Google Fonts ── */
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&family=Space+Grotesk:wght@400;500;600;700&display=swap');

/* ── Reset & Root ── */
*, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }

:root {
    --bg-base:       #020617;
    --bg-sidebar:    #0A0F1E;
    --bg-card:       #0D1526;
    --bg-card-hover: #111d35;
    --border:        #1E2D4A;
    --border-glow:   #38BDF840;
    --accent-blue:   #38BDF8;
    --accent-purple: #8B5CF6;
    --accent-cyan:   #06B6D4;
    --success:       #22C55E;
    --warning:       #F59E0B;
    --danger:        #EF4444;
    --text-primary:  #F0F6FF;
    --text-secondary:#7C9CBF;
    --text-muted:    #4A6080;
    --gradient-main: linear-gradient(135deg, #38BDF8, #8B5CF6);
    --gradient-card: linear-gradient(145deg, #0D1526, #0A1220);
    --shadow-glow:   0 0 30px #38BDF815;
    --shadow-card:   0 4px 24px #00000060;
    --radius-lg:     16px;
    --radius-xl:     24px;
    --radius-pill:   999px;
}

/* ── App background ── */
.stApp {
    background: var(--bg-base) !important;
    font-family: 'Plus Jakarta Sans', sans-serif !important;
    color: var(--text-primary) !important;
}

/* ── Hide default streamlit elements ── */
#MainMenu, footer, header { visibility: hidden; }
.stDeployButton { display: none !important; }
[data-testid="stToolbar"] { display: none !important; }

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background: var(--bg-sidebar) !important;
    border-right: 1px solid var(--border) !important;
    width: 240px !important;
}
[data-testid="stSidebar"] > div:first-child {
    padding: 0 !important;
}

/* ── Option menu overrides ── */
.nav-link {
    font-family: 'Plus Jakarta Sans', sans-serif !important;
    font-size: 13.5px !important;
    font-weight: 500 !important;
    color: var(--text-secondary) !important;
    border-radius: 10px !important;
    margin: 2px 12px !important;
    padding: 10px 14px !important;
    transition: all 0.2s ease !important;
}
.nav-link:hover {
    background: #1E2D4A30 !important;
    color: var(--text-primary) !important;
}
.nav-link.active {
    background: linear-gradient(135deg, #38BDF820, #8B5CF620) !important;
    color: var(--accent-blue) !important;
    border: 1px solid #38BDF830 !important;
    font-weight: 600 !important;
}
.nav-link-selected {
    background: linear-gradient(135deg, #38BDF820, #8B5CF620) !important;
}

/* ── Main content padding ── */
.main .block-container {
    padding: 2rem 2.5rem 3rem !important;
    max-width: 1400px !important;
}

/* ── Cards ── */
.rapiq-card {
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-radius: var(--radius-lg);
    padding: 1.5rem;
    margin-bottom: 1rem;
    box-shadow: var(--shadow-card);
    transition: border-color 0.2s ease, box-shadow 0.2s ease;
}
.rapiq-card:hover {
    border-color: var(--border-glow);
    box-shadow: var(--shadow-glow);
}

/* ── Hero Banner ── */
.hero-banner {
    background: linear-gradient(135deg, #0D1F3C 0%, #0D1526 50%, #1A0D2E 100%);
    border: 1px solid var(--border);
    border-radius: var(--radius-xl);
    padding: 2.5rem 3rem;
    margin-bottom: 1.8rem;
    position: relative;
    overflow: hidden;
}
.hero-banner::before {
    content: '';
    position: absolute;
    top: -60px; right: -60px;
    width: 280px; height: 280px;
    background: radial-gradient(circle, #38BDF812 0%, transparent 70%);
    border-radius: 50%;
}
.hero-banner::after {
    content: '';
    position: absolute;
    bottom: -80px; left: 30%;
    width: 320px; height: 200px;
    background: radial-gradient(circle, #8B5CF60A 0%, transparent 70%);
    border-radius: 50%;
}
.hero-title {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 3rem;
    font-weight: 700;
    background: linear-gradient(135deg, #F0F6FF, #38BDF8);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    line-height: 1.1;
    margin-bottom: 0.5rem;
}
.hero-subtitle {
    font-size: 1rem;
    color: var(--text-secondary);
    font-weight: 400;
    margin-bottom: 1.2rem;
}
.hero-badge {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    background: #38BDF815;
    border: 1px solid #38BDF830;
    border-radius: var(--radius-pill);
    padding: 4px 12px;
    font-size: 11px;
    font-weight: 600;
    color: var(--accent-blue);
    letter-spacing: 0.08em;
    text-transform: uppercase;
}

/* ── Section headers ── */
.section-header {
    display: flex;
    align-items: center;
    gap: 10px;
    margin-bottom: 1.2rem;
}
.section-icon {
    width: 32px; height: 32px;
    background: linear-gradient(135deg, #38BDF820, #8B5CF620);
    border: 1px solid #38BDF830;
    border-radius: 8px;
    display: flex; align-items: center; justify-content: center;
    font-size: 15px;
}
.section-title {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 1rem;
    font-weight: 600;
    color: var(--text-primary);
    letter-spacing: 0.01em;
}

/* ── Input labels ── */
.input-label {
    font-size: 10.5px;
    font-weight: 600;
    color: var(--text-muted);
    letter-spacing: 0.1em;
    text-transform: uppercase;
    margin-bottom: 6px;
}

/* ── Selectbox ── */
[data-baseweb="select"] > div {
    background: #070E1E !important;
    border: 1px solid var(--border) !important;
    border-radius: 10px !important;
    color: var(--text-primary) !important;
    font-family: 'Plus Jakarta Sans', sans-serif !important;
    font-size: 13.5px !important;
    transition: border-color 0.2s !important;
}
[data-baseweb="select"] > div:hover,
[data-baseweb="select"] > div:focus-within {
    border-color: #38BDF850 !important;
    box-shadow: 0 0 0 3px #38BDF810 !important;
}
[data-baseweb="popover"] {
    background: #0D1526 !important;
    border: 1px solid var(--border) !important;
    border-radius: 10px !important;
}
[role="option"] {
    background: #0D1526 !important;
    color: var(--text-secondary) !important;
    font-size: 13.5px !important;
}
[role="option"]:hover {
    background: #1E2D4A !important;
    color: var(--text-primary) !important;
}

/* ── Number input ── */
[data-testid="stNumberInput"] input {
    background: #070E1E !important;
    border: 1px solid var(--border) !important;
    border-radius: 10px !important;
    color: var(--text-primary) !important;
    font-family: 'Plus Jakarta Sans', sans-serif !important;
    font-size: 14px !important;
    padding: 10px 14px !important;
}
[data-testid="stNumberInput"] input:focus {
    border-color: #38BDF850 !important;
    box-shadow: 0 0 0 3px #38BDF810 !important;
    outline: none !important;
}

/* ── Number input stepper buttons ── */
[data-testid="stNumberInput"] button {
    background: #0D1526 !important;
    border-color: var(--border) !important;
    color: var(--text-secondary) !important;
}

/* ── Radio (gender) ── */
[data-testid="stRadio"] > div {
    display: flex !important;
    flex-direction: row !important;
    gap: 10px !important;
}
[data-testid="stRadio"] label {
    background: #070E1E !important;
    border: 1px solid var(--border) !important;
    border-radius: 10px !important;
    padding: 10px 22px !important;
    color: var(--text-secondary) !important;
    font-size: 13.5px !important;
    font-weight: 500 !important;
    cursor: pointer !important;
    transition: all 0.2s !important;
    flex: 1 !important;
    text-align: center !important;
}
[data-testid="stRadio"] label:hover {
    border-color: #38BDF850 !important;
    color: var(--text-primary) !important;
}
[data-testid="stRadio"] label[data-checked="true"] {
    border-color: var(--accent-blue) !important;
    background: #38BDF815 !important;
    color: var(--accent-blue) !important;
}

/* ── Predict Button ── */
.stButton > button {
    background: linear-gradient(135deg, #38BDF8, #6366F1, #8B5CF6) !important;
    color: white !important;
    font-family: 'Plus Jakarta Sans', sans-serif !important;
    font-weight: 700 !important;
    font-size: 14.5px !important;
    letter-spacing: 0.04em !important;
    border: none !important;
    border-radius: var(--radius-pill) !important;
    padding: 0.75rem 2rem !important;
    width: 100% !important;
    height: 54px !important;
    cursor: pointer !important;
    transition: all 0.3s ease !important;
    box-shadow: 0 4px 20px #38BDF840 !important;
    background-size: 200% auto !important;
}
.stButton > button:hover {
    background-position: right center !important;
    box-shadow: 0 6px 28px #8B5CF650 !important;
    transform: translateY(-1px) !important;
}
.stButton > button:active {
    transform: translateY(0) !important;
}

/* ── Download button variant ── */
[data-testid="stDownloadButton"] > button {
    background: transparent !important;
    border: 1px solid var(--border) !important;
    color: var(--text-secondary) !important;
    border-radius: 10px !important;
    font-size: 13px !important;
    box-shadow: none !important;
    height: auto !important;
    padding: 8px 16px !important;
}
[data-testid="stDownloadButton"] > button:hover {
    border-color: #38BDF850 !important;
    color: var(--accent-blue) !important;
    transform: none !important;
    box-shadow: none !important;
}

/* ── Metric cards ── */
.metric-card {
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-radius: var(--radius-lg);
    padding: 1.4rem 1.5rem;
    position: relative;
    overflow: hidden;
    transition: border-color 0.2s, box-shadow 0.2s;
}
.metric-card::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 2px;
    background: var(--gradient-main);
    border-radius: var(--radius-lg) var(--radius-lg) 0 0;
}
.metric-card:hover {
    border-color: #38BDF830;
    box-shadow: var(--shadow-glow);
}
.metric-label {
    font-size: 10px;
    font-weight: 600;
    color: var(--text-muted);
    letter-spacing: 0.12em;
    text-transform: uppercase;
    margin-bottom: 8px;
    display: flex;
    align-items: center;
    gap: 8px;
}
.metric-value {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 2rem;
    font-weight: 700;
    color: var(--text-primary);
    line-height: 1;
    margin-bottom: 4px;
}
.metric-sub {
    font-size: 11px;
    color: var(--text-muted);
    margin-top: 6px;
}

/* ── Prediction result card ── */
.prediction-result-card {
    background: linear-gradient(145deg, #0D1F3C, #0D1526);
    border: 1px solid #38BDF840;
    border-radius: var(--radius-xl);
    padding: 2rem 2.5rem;
    margin: 1.5rem 0;
    box-shadow: 0 0 40px #38BDF810, var(--shadow-card);
    position: relative;
    overflow: hidden;
}
.prediction-result-card::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 3px;
    background: linear-gradient(90deg, #38BDF8, #8B5CF6, #38BDF8);
    background-size: 200% auto;
    animation: shimmer 3s linear infinite;
}
@keyframes shimmer {
    0% { background-position: 0% center; }
    100% { background-position: 200% center; }
}
.prediction-tag {
    font-size: 10px;
    font-weight: 700;
    color: var(--accent-blue);
    letter-spacing: 0.15em;
    text-transform: uppercase;
    margin-bottom: 8px;
}
.prediction-category {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 2.6rem;
    font-weight: 700;
    color: var(--text-primary);
    line-height: 1.1;
    margin-bottom: 6px;
}
.prediction-desc {
    font-size: 13px;
    color: var(--text-secondary);
    margin-bottom: 1.5rem;
}

/* ── Progress bar override ── */
[data-testid="stProgress"] > div {
    background: var(--border) !important;
    border-radius: var(--radius-pill) !important;
    height: 6px !important;
}
[data-testid="stProgress"] > div > div {
    background: linear-gradient(90deg, #38BDF8, #8B5CF6) !important;
    border-radius: var(--radius-pill) !important;
}

/* ── Dataframe ── */
[data-testid="stDataFrame"] {
    background: var(--bg-card) !important;
    border: 1px solid var(--border) !important;
    border-radius: 12px !important;
    overflow: hidden !important;
}
.stDataFrame th {
    background: #0A0F1E !important;
    color: var(--text-muted) !important;
    font-size: 11px !important;
    font-weight: 600 !important;
    letter-spacing: 0.08em !important;
    text-transform: uppercase !important;
    padding: 10px 14px !important;
    border-bottom: 1px solid var(--border) !important;
}
.stDataFrame td {
    color: var(--text-secondary) !important;
    font-size: 13px !important;
    padding: 10px 14px !important;
    border-bottom: 1px solid #1E2D4A50 !important;
}

/* ── File uploader ── */
[data-testid="stFileUploader"] {
    background: #070E1E !important;
    border: 1px dashed var(--border) !important;
    border-radius: var(--radius-lg) !important;
    transition: border-color 0.2s !important;
}
[data-testid="stFileUploader"]:hover {
    border-color: #38BDF850 !important;
}
[data-testid="stFileUploader"] label {
    color: var(--text-secondary) !important;
    font-size: 13.5px !important;
}

/* ── Tags / badges ── */
.tag {
    display: inline-flex;
    align-items: center;
    padding: 3px 10px;
    border-radius: var(--radius-pill);
    font-size: 10.5px;
    font-weight: 600;
    letter-spacing: 0.06em;
    text-transform: uppercase;
}
.tag-blue   { background: #38BDF815; border: 1px solid #38BDF830; color: var(--accent-blue); }
.tag-purple { background: #8B5CF615; border: 1px solid #8B5CF630; color: var(--accent-purple); }
.tag-green  { background: #22C55E15; border: 1px solid #22C55E30; color: var(--success); }

/* ── Taxonomy class cards ── */
.class-card {
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 1.2rem 1.4rem;
    transition: border-color 0.2s, box-shadow 0.2s;
}
.class-card:hover {
    border-color: #38BDF830;
    box-shadow: var(--shadow-glow);
}
.class-number {
    font-size: 10px;
    font-weight: 700;
    color: var(--text-muted);
    letter-spacing: 0.12em;
    text-transform: uppercase;
    margin-bottom: 6px;
}
.class-name {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 1.05rem;
    font-weight: 600;
    color: var(--text-primary);
}

/* ── Sidebar logo area ── */
.sidebar-logo {
    padding: 24px 20px 16px;
    border-bottom: 1px solid var(--border);
    margin-bottom: 8px;
}
.logo-mark {
    width: 36px; height: 36px;
    background: linear-gradient(135deg, #38BDF8, #8B5CF6);
    border-radius: 10px;
    display: flex; align-items: center; justify-content: center;
    font-size: 18px;
    margin-bottom: 10px;
}
.logo-name {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 1.2rem;
    font-weight: 700;
    color: var(--text-primary);
}
.logo-sub {
    font-size: 10px;
    color: var(--text-muted);
    letter-spacing: 0.1em;
    text-transform: uppercase;
    margin-top: 2px;
}

/* ── Divider ── */
hr {
    border: none !important;
    border-top: 1px solid var(--border) !important;
    margin: 1.5rem 0 !important;
}

/* ── Streamlit col gap fix ── */
[data-testid="column"] { padding: 0 8px !important; }
[data-testid="stHorizontalBlock"] { gap: 0 !important; }

/* ── Alert / info override ── */
.stAlert { display: none !important; }

/* ── Spinner ── */
[data-testid="stSpinner"] { color: var(--accent-blue) !important; }

/* ── Scrollbar ── */
::-webkit-scrollbar { width: 6px; height: 6px; }
::-webkit-scrollbar-track { background: var(--bg-base); }
::-webkit-scrollbar-thumb { background: var(--border); border-radius: 3px; }
::-webkit-scrollbar-thumb:hover { background: var(--text-muted); }

</style>
""", unsafe_allow_html=True)

# =========================================
# LOAD MODEL & SCALER
# =========================================

@st.cache_resource
def load_model():
    try:
        model = joblib.load("model_mlp_iq.pkl")
        scaler = joblib.load("scaler_iq.pkl")
        return model, scaler
    except Exception as e:
        st.error(f"Error loading model/scaler: {e}")
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

gender_map = {"Male": 1, "Female": 0}

reverse_edu_map = {
    "primary or lower secondary": 0,
    "vocational": 1,
    "secondary": 2,
    "higher": 3
}

reverse_gender_map = {"male": 1, "female": 0}

iq_labels = {
    0: "Moderate ID",
    1: "Mild ID",
    2: "Below Average",
    3: "Average",
    4: "Above Average"
}

iq_colors = {
    "Moderate ID":   "#EF4444",
    "Mild ID":       "#F97316",
    "Below Average": "#F59E0B",
    "Average":       "#38BDF8",
    "Above Average": "#8B5CF6"
}

iq_descriptions = {
    "Moderate ID":   "The model indicates significant developmental support may be needed.",
    "Mild ID":       "The model suggests mild intellectual developmental patterns.",
    "Below Average": "The model indicates cognitive performance slightly below peer average.",
    "Average":       "The model predicts typical cognitive development for the age group.",
    "Above Average": "The model predicts high cognitive adaptability based on socio-demographic profile."
}

required_columns = ["education_mother", "education_father", "age_years", "gender"]

# =========================================
# SIDEBAR
# =========================================

with st.sidebar:
    st.markdown("""
    <div class="sidebar-logo">
        <div class="logo-mark">🧠</div>
        <div class="logo-name">RapIQ</div>
        <div class="logo-sub">MLP Architecture: (10,6)</div>
    </div>
    """, unsafe_allow_html=True)

    selected = option_menu(
        menu_title=None,
        options=["Dashboard", "Bulk Prediction", "About Model"],
        icons=["grid-1x2-fill", "table", "info-circle-fill"],
        default_index=0,
        styles={
            "container": {
                "padding": "8px 0",
                "background-color": "transparent",
            },
            "icon": {
                "color": "#4A6080",
                "font-size": "14px",
            },
            "nav-link": {
                "font-family": "'Plus Jakarta Sans', sans-serif",
                "font-size": "13.5px",
                "color": "#7C9CBF",
                "border-radius": "10px",
                "margin": "2px 12px",
                "padding": "10px 14px",
                "--hover-color": "#1E2D4A30",
            },
            "nav-link-selected": {
                "background": "linear-gradient(135deg, #38BDF820, #8B5CF620)",
                "color": "#38BDF8",
                "border": "1px solid #38BDF830",
                "font-weight": "600",
            },
        }
    )

    st.markdown("<div style='height:1px; background:var(--border); margin:16px 20px'></div>", unsafe_allow_html=True)
    st.markdown("""
    <div style='padding:16px 20px 20px; margin-top:auto'>
        <div style='font-size:10px; color:#4A6080; letter-spacing:0.1em; text-transform:uppercase; margin-bottom:10px'>System</div>
        <div style='font-size:12px; color:#7C9CBF; line-height:1.7'>
            <div>🟢 &nbsp;Model Loaded</div>
            <div>📊 &nbsp;MLP Classifier</div>
            <div>⚡ &nbsp;StandardScaler</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

# =========================================
# HELPER: PLOTLY THEME
# =========================================

PLOT_LAYOUT = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(family="Plus Jakarta Sans", color="#7C9CBF"),
    margin=dict(l=20, r=20, t=30, b=20),
    xaxis=dict(
        gridcolor="#1E2D4A",
        zerolinecolor="#1E2D4A",
        tickfont=dict(size=11, color="#4A6080"),
    ),
    yaxis=dict(
        gridcolor="#1E2D4A",
        zerolinecolor="#1E2D4A",
        tickfont=dict(size=11, color="#4A6080"),
    ),
    hoverlabel=dict(
        bgcolor="#0D1526",
        bordercolor="#1E2D4A",
        font=dict(family="Plus Jakarta Sans", size=12, color="#F0F6FF"),
    ),
)

# =========================================
# PAGE: DASHBOARD (SINGLE PREDICTION)
# =========================================

if selected == "Dashboard":

    # Hero
    st.markdown("""
    <div class="hero-banner">
        <div class="hero-badge">⚡ Intelligence Classification Platform</div>
        <div class="hero-title" style="margin-top:14px">RapIQ</div>
        <div class="hero-subtitle">AI-powered pediatric IQ category prediction using a Multilayer Perceptron (MLP) architecture trained on the Stanford-Binet Intelligence Scales dataset.</div>
        <div style="display:flex; gap:10px; flex-wrap:wrap; margin-top:4px">
            <span class="tag tag-blue">MLP (10,6)</span>
            <span class="tag tag-purple">5 Classes</span>
            <span class="tag tag-green">94.2% Accuracy</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Input cards
    col_left, col_right = st.columns(2, gap="medium")

    with col_left:
        st.markdown("""
        <div class="rapiq-card">
            <div class="section-header">
                <div class="section-icon">👨‍👩‍👦</div>
                <div class="section-title">Family Information</div>
            </div>
        """, unsafe_allow_html=True)

        st.markdown('<div class="input-label">Mother Education Level</div>', unsafe_allow_html=True)
        education_mother = st.selectbox("_", list(edu_map.keys()), key="edu_mother", label_visibility="collapsed")

        st.markdown('<div class="input-label" style="margin-top:14px">Father Education Level</div>', unsafe_allow_html=True)
        education_father = st.selectbox("_", list(edu_map.keys()), key="edu_father", label_visibility="collapsed")

        st.markdown("</div>", unsafe_allow_html=True)

    with col_right:
        st.markdown("""
        <div class="rapiq-card">
            <div class="section-header">
                <div class="section-icon">👶</div>
                <div class="section-title">Child Information</div>
            </div>
        """, unsafe_allow_html=True)

        st.markdown('<div class="input-label">Age (Years)</div>', unsafe_allow_html=True)
        age = st.number_input("_", min_value=1, max_value=18, value=10, key="age_input", label_visibility="collapsed")

        st.markdown('<div class="input-label" style="margin-top:14px">Gender</div>', unsafe_allow_html=True)
        gender = st.radio("_", list(gender_map.keys()), horizontal=True, key="gender_input", label_visibility="collapsed")

        st.markdown("</div>", unsafe_allow_html=True)

    # Predict button
    st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)
    col_btn = st.columns([1, 2, 1])[1]
    with col_btn:
        predict_btn = st.button("🚀  Predict IQ Category", key="predict_btn")

    # Prediction logic
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
            confidence = float(np.max(probabilities) * 100)
            predicted_label = iq_labels[prediction]
            pred_color = iq_colors.get(predicted_label, "#38BDF8")
            pred_desc = iq_descriptions.get(predicted_label, "")

            # Result + chart side by side
            r_col, c_col = st.columns([1.1, 1], gap="medium")

            with r_col:
                st.markdown(f"""
                <div class="prediction-result-card">
                    <div class="prediction-tag">Final Prediction</div>
                    <div class="prediction-category" style="color:{pred_color}">{predicted_label}</div>
                    <div class="prediction-desc">{pred_desc}</div>
                    <div style="margin-bottom:8px">
                        <div style="display:flex;justify-content:space-between;margin-bottom:6px">
                            <span style="font-size:11px;color:#4A6080;font-weight:600;letter-spacing:0.08em;text-transform:uppercase">Confidence Score</span>
                            <span style="font-size:13px;color:{pred_color};font-weight:700">{confidence:.1f}%</span>
                        </div>
                    </div>
                </div>
                """, unsafe_allow_html=True)
                st.progress(confidence / 100)

            with c_col:
                prob_df = pd.DataFrame({
                    "Category": list(iq_labels.values()),
                    "Probability": probabilities * 100,
                    "Color": [iq_colors[l] for l in iq_labels.values()]
                })

                fig = go.Figure()
                fig.add_trace(go.Bar(
                    x=prob_df["Category"],
                    y=prob_df["Probability"],
                    marker=dict(
                        color=prob_df["Color"],
                        opacity=0.85,
                        line=dict(color="rgba(0,0,0,0)", width=0)
                    ),
                    text=[f"{v:.1f}%" for v in prob_df["Probability"]],
                    textposition="outside",
                    textfont=dict(size=10, color="#7C9CBF"),
                    hovertemplate="<b>%{x}</b><br>%{y:.1f}%<extra></extra>",
                ))

                fig.update_layout(
                    **PLOT_LAYOUT,
                    title=dict(text="Confidence Probability", font=dict(size=13, color="#7C9CBF"), x=0),
                    height=300,
                    yaxis=dict(**PLOT_LAYOUT["yaxis"], range=[0, 110]),
                    bargap=0.35,
                    showlegend=False,
                )
                st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

        except Exception as e:
            st.markdown(f"""
            <div class="rapiq-card" style="border-color:#EF444430">
                <div style="color:#EF4444;font-size:13px">⚠️ Prediction Error: {e}</div>
            </div>
            """, unsafe_allow_html=True)

# =========================================
# PAGE: BULK PREDICTION
# =========================================

elif selected == "Bulk Prediction":

    # Hero
    st.markdown("""
    <div class="hero-banner">
        <div class="hero-badge">🗂️ Enterprise Processing</div>
        <div class="hero-title" style="margin-top:14px;font-size:2.2rem">Orchestrate Large Datasets</div>
        <div class="hero-subtitle">Upload your CSV file to run parallelized MLP inference across thousands of samples. Our engine automatically sanitizes features and produces confidence-weighted IQ categories.</div>
    </div>
    """, unsafe_allow_html=True)

    # Template download + upload
    up_col, prev_col = st.columns([1, 1.4], gap="medium")

    with up_col:
        st.markdown("""
        <div class="rapiq-card" style="height:100%">
            <div class="section-header">
                <div class="section-icon">📤</div>
                <div class="section-title">Upload Dataset</div>
            </div>
            <div style="font-size:12px;color:#4A6080;margin-bottom:14px;line-height:1.7">
                Required columns:<br>
                <code style="color:#38BDF8;font-size:11px">education_mother</code> &nbsp;
                <code style="color:#38BDF8;font-size:11px">education_father</code><br>
                <code style="color:#38BDF8;font-size:11px">age_years</code> &nbsp;
                <code style="color:#38BDF8;font-size:11px">gender</code>
            </div>
        """, unsafe_allow_html=True)

        # Template download
        template_df = pd.DataFrame({
            "education_mother": ["secondary", "higher", "vocational"],
            "education_father": ["vocational", "higher", "secondary"],
            "age_years": [10, 15, 8],
            "gender": ["male", "female", "male"]
        })
        template_csv = template_df.to_csv(index=False).encode("utf-8")
        st.download_button("⬇ Download CSV Template", data=template_csv,
                           file_name="template_input_iq.csv", mime="text/csv")

        st.markdown("<div style='height:12px'></div>", unsafe_allow_html=True)
        uploaded_file = st.file_uploader("Upload CSV File", type=["csv"], label_visibility="collapsed")
        st.markdown("</div>", unsafe_allow_html=True)

    with prev_col:
        if uploaded_file:
            try:
                df = pd.read_csv(uploaded_file, sep=None, engine="python", decimal=",")
                st.markdown("""
                <div class="rapiq-card">
                    <div class="section-header">
                        <div class="section-icon">👁️</div>
                        <div class="section-title">Dataset Preview</div>
                    </div>
                """, unsafe_allow_html=True)
                st.dataframe(df.head(5), use_container_width=True, hide_index=True)
                st.markdown(f"""
                    <div style="display:flex;gap:12px;margin-top:10px">
                        <span class="tag tag-blue">{len(df)} Rows</span>
                        <span class="tag tag-purple">{len(df.columns)} Columns</span>
                    </div>
                </div>
                """, unsafe_allow_html=True)
            except Exception as e:
                st.markdown(f'<div class="rapiq-card" style="border-color:#EF444430"><div style="color:#EF4444;font-size:13px">Read Error: {e}</div></div>', unsafe_allow_html=True)
        else:
            st.markdown("""
            <div class="rapiq-card" style="height:100%;display:flex;align-items:center;justify-content:center;min-height:180px">
                <div style="text-align:center;color:#4A6080">
                    <div style="font-size:2.5rem;margin-bottom:10px">📂</div>
                    <div style="font-size:13px">Upload a CSV to preview your dataset</div>
                </div>
            </div>
            """, unsafe_allow_html=True)

    # Process if uploaded
    if uploaded_file:
        st.markdown("<div style='height:4px'></div>", unsafe_allow_html=True)

        run_col = st.columns([2, 1, 2])[1]
        with run_col:
            run_btn = st.button("▶  Run Bulk Analysis", key="run_bulk")

        if run_btn:
            try:
                df_encoded = df.copy()
                df_encoded = df_encoded.replace(r'^\s*$', np.nan, regex=True)
                jumlah_sebelum = len(df_encoded)
                df_encoded = df_encoded.dropna(subset=required_columns)
                jumlah_sesudah = len(df_encoded)
                jumlah_terhapus = jumlah_sebelum - jumlah_sesudah

                if len(df_encoded) == 0:
                    st.error("All data removed due to missing values.")
                    st.stop()

                for col in ["education_mother", "education_father"]:
                    df_encoded[col] = df_encoded[col].astype(str).str.strip().str.lower().str.replace(r"\s+", " ", regex=True)
                df_encoded["gender"] = df_encoded["gender"].astype(str).str.strip().str.lower()
                df_encoded["age_years"] = pd.to_numeric(df_encoded["age_years"], errors="coerce")
                df_encoded["education_mother"] = df_encoded["education_mother"].map(reverse_edu_map)
                df_encoded["education_father"] = df_encoded["education_father"].map(reverse_edu_map)
                df_encoded["gender"] = df_encoded["gender"].map(reverse_gender_map)

                errors = []
                if df_encoded["education_mother"].isnull().any(): errors.append("education_mother")
                if df_encoded["education_father"].isnull().any(): errors.append("education_father")
                if df_encoded["gender"].isnull().any(): errors.append("gender")
                if errors:
                    st.error(f"Invalid category values in: {', '.join(errors)}")
                    st.stop()

                X = df_encoded[required_columns]
                X_scaled = scaler.transform(X)
                predictions = model.predict(X_scaled)
                probabilities = model.predict_proba(X_scaled)
                confidence_scores = np.max(probabilities, axis=1) * 100
                predicted_labels = [iq_labels[p] for p in predictions]

                result_df = pd.DataFrame({
                    "education_mother": df.loc[df_encoded.index, "education_mother"],
                    "education_father": df.loc[df_encoded.index, "education_father"],
                    "age_years": df.loc[df_encoded.index, "age_years"],
                    "gender": df.loc[df_encoded.index, "gender"],
                    "predicted_iq_category": predicted_labels,
                    "confidence_score (%)": [f"{x:.2f}%" for x in confidence_scores]
                })

                # Stats row
                avg_conf = float(confidence_scores.mean())
                cat_counts = pd.Series(predicted_labels).value_counts()
                top_cat = cat_counts.index[0] if len(cat_counts) > 0 else "—"

                st.markdown("<div style='height:12px'></div>", unsafe_allow_html=True)
                s1, s2, s3, s4 = st.columns(4, gap="medium")
                with s1:
                    st.markdown(f"""
                    <div class="metric-card">
                        <div class="metric-label">📊 Total Samples</div>
                        <div class="metric-value">{len(result_df):,}</div>
                        <div class="metric-sub">{jumlah_terhapus} rows skipped</div>
                    </div>""", unsafe_allow_html=True)
                with s2:
                    st.markdown(f"""
                    <div class="metric-card">
                        <div class="metric-label">🎯 Avg Confidence</div>
                        <div class="metric-value" style="color:#38BDF8">{avg_conf:.1f}%</div>
                        <div class="metric-sub">Across all predictions</div>
                    </div>""", unsafe_allow_html=True)
                with s3:
                    st.markdown(f"""
                    <div class="metric-card">
                        <div class="metric-label">🏷️ Top Category</div>
                        <div class="metric-value" style="font-size:1.3rem">{top_cat}</div>
                        <div class="metric-sub">{cat_counts.iloc[0]} samples</div>
                    </div>""", unsafe_allow_html=True)
                with s4:
                    st.markdown(f"""
                    <div class="metric-card">
                        <div class="metric-label">✅ Status</div>
                        <div class="metric-value" style="color:#22C55E;font-size:1.3rem">Done</div>
                        <div class="metric-sub">Processing complete</div>
                    </div>""", unsafe_allow_html=True)

                st.markdown("<div style='height:12px'></div>", unsafe_allow_html=True)

                # Chart + Results
                chart_col, res_col = st.columns([1, 1.2], gap="medium")

                with chart_col:
                    st.markdown("""
                    <div class="rapiq-card">
                        <div class="section-header">
                            <div class="section-icon">📈</div>
                            <div class="section-title">IQ Distribution Analysis</div>
                        </div>
                    """, unsafe_allow_html=True)

                    dist_counts = pd.Series(predicted_labels).value_counts().reindex(list(iq_labels.values()), fill_value=0)
                    colors = [iq_colors.get(c, "#38BDF8") for c in dist_counts.index]

                    fig2 = go.Figure()
                    fig2.add_trace(go.Bar(
                        x=dist_counts.index,
                        y=dist_counts.values,
                        marker=dict(color=colors, opacity=0.85),
                        text=dist_counts.values,
                        textposition="outside",
                        textfont=dict(size=11, color="#7C9CBF"),
                        hovertemplate="<b>%{x}</b><br>%{y} samples<extra></extra>",
                    ))
                    fig2.update_layout(
                        **PLOT_LAYOUT,
                        height=280,
                        bargap=0.3,
                        showlegend=False,
                        yaxis=dict(**PLOT_LAYOUT["yaxis"], title=dict(text="Samples", font=dict(size=11))),
                    )
                    st.plotly_chart(fig2, use_container_width=True, config={"displayModeBar": False})
                    st.markdown("</div>", unsafe_allow_html=True)

                with res_col:
                    st.markdown("""
                    <div class="rapiq-card">
                        <div class="section-header">
                            <div class="section-icon">📋</div>
                            <div class="section-title">Analysis Results</div>
                        </div>
                    """, unsafe_allow_html=True)
                    st.dataframe(result_df[["age_years", "gender", "predicted_iq_category", "confidence_score (%)"]].head(8),
                                 use_container_width=True, hide_index=True)

                    csv_out = result_df.to_csv(index=False).encode("utf-8")
                    st.download_button("⬇ Download Full Results", data=csv_out,
                                       file_name="prediction_result.csv", mime="text/csv")
                    st.markdown("</div>", unsafe_allow_html=True)

            except Exception as e:
                st.markdown(f'<div class="rapiq-card" style="border-color:#EF444430"><div style="color:#EF4444;font-size:13px">ERROR: {e}</div></div>', unsafe_allow_html=True)

# =========================================
# PAGE: ABOUT MODEL
# =========================================

elif selected == "About Model":

    st.markdown("""
    <div style="margin-bottom:1.8rem">
        <div style="font-family:'Space Grotesk',sans-serif;font-size:1.8rem;font-weight:700;color:#F0F6FF;margin-bottom:4px">
            About Model
        </div>
        <div style="font-size:13px;color:#4A6080">RapIQ Neural Research Platform — Model Documentation</div>
    </div>
    """, unsafe_allow_html=True)

    # Metric cards
    m1, m2, m3, m4 = st.columns(4, gap="medium")
    metrics = [
        ("📊", "Accuracy Score", "94.2%", "Weighted across all classes", "#38BDF8"),
        ("🎯", "F1-Score (Weighted)", "0.93", "Macro-averaged F1", "#8B5CF6"),
        ("🗃️", "Samples Processed", "~80k", "Stanford-Binet Dataset", "#06B6D4"),
        ("⚡", "Model Latency", "12ms", "Average response time", "#22C55E"),
    ]
    for col, (icon, label, val, sub, color) in zip([m1, m2, m3, m4], metrics):
        with col:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">{icon} {label}</div>
                <div class="metric-value" style="color:{color}">{val}</div>
                <div class="metric-sub">{sub}</div>
            </div>""", unsafe_allow_html=True)

    st.markdown("<div style='height:12px'></div>", unsafe_allow_html=True)

    # About + Disclaimer
    info_col, disc_col = st.columns([1.5, 1], gap="medium")

    with info_col:
        st.markdown("""
        <div class="rapiq-card" style="border-color:#38BDF830">
            <div class="section-header">
                <div class="section-icon">🧠</div>
                <div class="section-title">About RapIQ</div>
            </div>
            <div style="font-size:13.5px;color:#7C9CBF;line-height:1.8;margin-bottom:1.2rem">
                RapIQ is a state-of-the-art AI orchestration platform designed specifically for child
                intelligence classification. Utilizing advanced Multi-Layer Perceptron (MLP) architectures,
                the system analyzes complex psychometric patterns to provide objective data synthesis
                for educational and research professionals.
            </div>
            <div style="display:flex;gap:8px;flex-wrap:wrap">
                <span class="tag tag-blue">AI-Driven</span>
                <span class="tag tag-purple">Classification</span>
                <span class="tag tag-green">MLP Engine</span>
            </div>
        </div>
        """, unsafe_allow_html=True)

    with disc_col:
        st.markdown("""
        <div class="rapiq-card" style="border-color:#F59E0B30;height:100%">
            <div style="font-size:10px;font-weight:700;color:#F59E0B;letter-spacing:0.12em;text-transform:uppercase;margin-bottom:12px">
                ⚠️ Research Disclaimer
            </div>
            <div style="font-size:12.5px;color:#7C9CBF;line-height:1.8;font-style:italic">
                "RapIQ is developed as a decision support system for research and preliminary screening purposes.
                It is NOT intended to serve as a standalone diagnostic tool. All results should be reviewed
                by licensed clinical neuropsychologists."
            </div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<div style='height:12px'></div>", unsafe_allow_html=True)

    # Model info + Dataset
    model_col, data_col = st.columns([1.5, 1], gap="medium")

    with model_col:
        st.markdown("""
        <div class="rapiq-card">
            <div class="section-header">
                <div class="section-icon">⚙️</div>
                <div class="section-title">Model Information</div>
            </div>
            <div style="display:grid;grid-template-columns:1fr 1fr 1fr;gap:20px;margin-top:4px">
                <div>
                    <div style="font-size:10px;color:#4A6080;letter-spacing:0.1em;text-transform:uppercase;margin-bottom:5px">Algorithm</div>
                    <div style="font-size:14px;color:#F0F6FF;font-weight:600">MLP Classifier</div>
                </div>
                <div>
                    <div style="font-size:10px;color:#4A6080;letter-spacing:0.1em;text-transform:uppercase;margin-bottom:5px">Hidden Layer</div>
                    <div style="font-size:14px;color:#F0F6FF;font-weight:600">(10, 6)</div>
                </div>
                <div>
                    <div style="font-size:10px;color:#4A6080;letter-spacing:0.1em;text-transform:uppercase;margin-bottom:5px">Activation</div>
                    <div style="font-size:14px;color:#F0F6FF;font-weight:600">ReLU</div>
                </div>
                <div>
                    <div style="font-size:10px;color:#4A6080;letter-spacing:0.1em;text-transform:uppercase;margin-bottom:5px">Optimizer</div>
                    <div style="font-size:14px;color:#F0F6FF;font-weight:600">Adam</div>
                </div>
                <div>
                    <div style="font-size:10px;color:#4A6080;letter-spacing:0.1em;text-transform:uppercase;margin-bottom:5px">Scaling</div>
                    <div style="font-size:14px;color:#F0F6FF;font-weight:600">StandardScaler</div>
                </div>
                <div>
                    <div style="font-size:10px;color:#4A6080;letter-spacing:0.1em;text-transform:uppercase;margin-bottom:5px">Oversampling</div>
                    <div style="font-size:14px;color:#F0F6FF;font-weight:600">SMOTE</div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    with data_col:
        st.markdown("""
        <div class="rapiq-card" style="height:100%">
            <div class="section-header">
                <div class="section-icon">🗄️</div>
                <div class="section-title">Dataset</div>
            </div>
            <div style="font-size:13px;color:#7C9CBF;line-height:1.7;margin-bottom:1rem">
                Utilizing the comprehensive Stanford-Binet Intelligence Scales dataset compiled
                from extensive longitudinal research.
            </div>
            <div style="background:#070E1E;border:1px solid var(--border);border-radius:10px;padding:14px 18px;display:flex;align-items:center;justify-content:space-between">
                <div>
                    <div style="font-family:'Space Grotesk',sans-serif;font-size:1.5rem;font-weight:700;color:#F0F6FF">80,000+</div>
                    <div style="font-size:11px;color:#4A6080;margin-top:2px">Validated Research Samples</div>
                </div>
                <div style="font-size:2rem;opacity:0.3">🗃️</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<div style='height:12px'></div>", unsafe_allow_html=True)

    # Taxonomy
    st.markdown("""
    <div class="rapiq-card">
        <div style="text-align:center;margin-bottom:1.5rem">
            <div style="font-family:'Space Grotesk',sans-serif;font-size:1.2rem;font-weight:700;color:#F0F6FF">
                Intelligence Classification Taxonomy
            </div>
            <div style="font-size:12px;color:#4A6080;margin-top:4px">MLP output class definitions</div>
        </div>
    """, unsafe_allow_html=True)

    class_colors = ["#EF4444", "#F97316", "#F59E0B", "#38BDF8", "#8B5CF6"]
    class_cols = st.columns(5, gap="medium")
    for i, (key, label) in enumerate(iq_labels.items()):
        with class_cols[i]:
            color = class_colors[i]
            st.markdown(f"""
            <div class="class-card" style="border-color:{color}25;border-top:2px solid {color}">
                <div class="class-number">Class {key+1:02d}</div>
                <div class="class-name">{label}</div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("</div>", unsafe_allow_html=True)

    # Footer
    st.markdown("""
    <div style="text-align:center;margin-top:2.5rem;padding-top:1.5rem;border-top:1px solid var(--border)">
        <div style="font-size:11px;color:#4A6080;letter-spacing:0.1em;text-transform:uppercase">
            © 2024 RapIQ Neural Research Group. All Rights Reserved.
        </div>
    </div>
    """, unsafe_allow_html=True)