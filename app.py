# =========================================
# STREAMLIT APP
# IQ Classification using MLP
# =========================================

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt

# =========================================
# PAGE CONFIG
# =========================================

st.set_page_config(
    page_title="Rapid IQ Classification",
    page_icon="🧠",
    layout="wide"
)

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

gender_map = {
    "Male": 1,
    "Female": 0
}

reverse_edu_map = {
    "primary or lower secondary": 0,
    "vocational": 1,
    "secondary": 2,
    "higher": 3
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

# =========================================
# REQUIRED COLUMNS
# =========================================

required_columns = [
    "education_mother",
    "education_father",
    "age_years",
    "gender"
]

# =========================================
# TITLE
# =========================================

st.title("🧠 Rapid IQ Classification System")

st.markdown(
    """
    This application predicts child IQ category 
    based on parental education, age, and gender 
    using a Multilayer Perceptron (MLP) model.
    """
)

# =========================================
# SIDEBAR
# =========================================

menu = st.sidebar.selectbox(
    "Select Menu",
    [
        "Single Prediction",
        "Bulk Prediction (CSV)",
        "About Model"
    ]
)

# =========================================
# SINGLE PREDICTION
# =========================================

if menu == "Single Prediction":

    st.subheader("📌 Single Prediction")

    col1, col2 = st.columns(2)

    with col1:

        education_mother = st.selectbox(
            "Mother Education",
            list(edu_map.keys())
        )

        education_father = st.selectbox(
            "Father Education",
            list(edu_map.keys())
        )

    with col2:

        age = st.number_input(
            "Age",
            min_value=1,
            max_value=18,
            value=10
        )

        gender = st.selectbox(
            "Gender",
            list(gender_map.keys())
        )

    # =========================================
    # PREDICT BUTTON
    # =========================================

    if st.button("Predict IQ Category"):

        try:

            # Input dataframe
            input_data = pd.DataFrame({
                "education_mother": [edu_map[education_mother]],
                "education_father": [edu_map[education_father]],
                "age_years": [age],
                "gender": [gender_map[gender]]
            })

            # Scaling
            scaled_data = scaler.transform(input_data)

            # Prediction
            prediction = model.predict(scaled_data)[0]

            # Probability
            probabilities = model.predict_proba(scaled_data)[0]

            confidence = np.max(probabilities) * 100

            predicted_label = iq_labels[prediction]

            # =========================================
            # RESULT
            # =========================================

            st.success(
                f"Predicted IQ Category: {predicted_label}"
            )

            st.info(
                f"Confidence Score: {confidence:.2f}%"
            )

            # =========================================
            # PROBABILITY CHART
            # =========================================

            st.subheader("📊 Prediction Probability")

            prob_df = pd.DataFrame({
                "Category": list(iq_labels.values()),
                "Probability": probabilities * 100
            })

            fig, ax = plt.subplots(figsize=(8, 4))

            bars = ax.bar(
                prob_df["Category"],
                prob_df["Probability"]
            )

            # Add labels
            for bar in bars:

                height = bar.get_height()

                ax.text(
                    bar.get_x() + bar.get_width()/2,
                    height + 1,
                    f"{height:.1f}%",
                    ha='center'
                )

            ax.set_ylabel("Probability (%)")
            ax.set_ylim(0, 100)

            st.pyplot(fig)

        except Exception as e:

            st.error(f"Prediction Error: {e}")

# =========================================
# BULK PREDICTION
# =========================================

elif menu == "Bulk Prediction (CSV)":

    st.subheader("📂 Bulk Prediction Using CSV")

    st.markdown("""
    ### Required CSV Columns:
    - education_mother
    - education_father
    - age_years
    - gender
    """)

    # =========================================
    # TEMPLATE CSV
    # =========================================

    template_df = pd.DataFrame({

        "education_mother": [
            "secondary",
            "higher"
        ],

        "education_father": [
            "vocational",
            "higher"
        ],

        "age_years": [
            10,
            15
        ],

        "gender": [
            "male",
            "female"
        ]

    })

    template_csv = (
        template_df
        .to_csv(index=False)
        .encode("utf-8")
    )

    st.download_button(
        label="⬇ Download CSV Template",
        data=template_csv,
        file_name="template_input_iq.csv",
        mime="text/csv"
    )

    # =========================================
    # FILE UPLOADER
    # =========================================

    uploaded_file = st.file_uploader(
        "Upload CSV File",
        type=["csv"]
    )

    if uploaded_file is not None:

        try:

            # =========================================
            # READ CSV
            # =========================================

            df = pd.read_csv(
                uploaded_file,
                sep=None,
                engine="python",
                decimal=","
            )

            st.subheader("📄 Dataset Preview")
            st.dataframe(df.head())

            # =========================================
            # VALIDASI KOLOM
            # =========================================

            missing_cols = [
                col for col in required_columns
                if col not in df.columns
            ]

            if len(missing_cols) > 0:

                st.error(
                    f"""
Missing columns:

{missing_cols}

Required columns:
- education_mother
- education_father
- age_years
- gender
                    """
                )

                st.stop()

            # =========================================
            # PREPROCESSING
            # =========================================

            df_encoded = df.copy()

            # =========================================
            # GANTI STRING KOSONG JADI NaN
            # =========================================

            df_encoded = df_encoded.replace(
                r'^\s*$',
                np.nan,
                regex=True
            )

            # =========================================
            # HAPUS MISSING VALUE
            # =========================================

            jumlah_sebelum = len(df_encoded)

            df_encoded = df_encoded.dropna(
                subset=required_columns
            )

            jumlah_sesudah = len(df_encoded)

            jumlah_terhapus = (
                jumlah_sebelum - jumlah_sesudah
            )

            if len(df_encoded) == 0:

                st.error(
                    "Semua data terhapus karena missing value."
                )

                st.stop()

            # =========================================
            # CLEANING STRING
            # =========================================

            df_encoded["education_mother"] = (
                df_encoded["education_mother"]
                .astype(str)
                .str.strip()
                .str.lower()
                .str.replace(r"\s+", " ", regex=True)
            )

            df_encoded["education_father"] = (
                df_encoded["education_father"]
                .astype(str)
                .str.strip()
                .str.lower()
                .str.replace(r"\s+", " ", regex=True)
            )

            df_encoded["gender"] = (
                df_encoded["gender"]
                .astype(str)
                .str.strip()
                .str.lower()
            )

            # =========================================
            # AGE CONVERSION
            # =========================================

            df_encoded["age_years"] = pd.to_numeric(
                df_encoded["age_years"],
                errors="coerce"
            )

            # =========================================
            # ENCODING
            # =========================================

            df_encoded["education_mother"] = (
                df_encoded["education_mother"]
                .map(reverse_edu_map)
            )

            df_encoded["education_father"] = (
                df_encoded["education_father"]
                .map(reverse_edu_map)
            )

            df_encoded["gender"] = (
                df_encoded["gender"]
                .map(reverse_gender_map)
            )

            # =========================================
            # VALIDASI KATEGORI
            # =========================================

            invalid_edu_mother = df_encoded[
                df_encoded["education_mother"].isnull()
            ]

            invalid_edu_father = df_encoded[
                df_encoded["education_father"].isnull()
            ]

            invalid_gender = df_encoded[
                df_encoded["gender"].isnull()
            ]

            error_message = ""

            if len(invalid_edu_mother) > 0:

                error_message += (
                    "\n• Education Mother tidak valid"
                )

            if len(invalid_edu_father) > 0:

                error_message += (
                    "\n• Education Father tidak valid"
                )

            if len(invalid_gender) > 0:

                error_message += (
                    "\n• Gender tidak valid"
                )

            if error_message != "":

                st.error(
                    f"""
Terdapat kategori yang tidak sesuai.

{error_message}

Gunakan format berikut:

Education:
- primary or lower secondary
- vocational
- secondary
- higher

Gender:
- male
- female
"""
                )

                st.stop()

            # =========================================
            # FEATURE SELECTION
            # =========================================

            X = df_encoded[[
                "education_mother",
                "education_father",
                "age_years",
                "gender"
            ]]

            # =========================================
            # SCALING
            # =========================================

            X_scaled = scaler.transform(X)

            # =========================================
            # PREDICTION
            # =========================================

            predictions = model.predict(X_scaled)

            # =========================================
            # PROBABILITY
            # =========================================

            probabilities = model.predict_proba(X_scaled)

            confidence_scores = (
                np.max(probabilities, axis=1) * 100
            )

            # =========================================
            # LABEL MAPPING
            # =========================================

            predicted_labels = [
                iq_labels[p]
                for p in predictions
            ]

            # =========================================
            # OUTPUT
            # =========================================

            result_df = pd.DataFrame({

                "education_mother":
                    df.loc[
                        df_encoded.index,
                        "education_mother"
                    ],

                "education_father":
                    df.loc[
                        df_encoded.index,
                        "education_father"
                    ],

                "age_years":
                    df.loc[
                        df_encoded.index,
                        "age_years"
                    ],

                "gender":
                    df.loc[
                        df_encoded.index,
                        "gender"
                    ],

                "predicted_iq_category":
                    predicted_labels,

                "confidence_score (%)": [
                    f"{x:.2f}%"
                    for x in confidence_scores
                ]

            })

            # =========================================
            # STATUS
            # =========================================

            st.success(
                "Prediction completed successfully."
            )

            st.info(
                f"""
Jumlah data diproses: {len(result_df)}

Baris dihapus: {jumlah_terhapus}
                """
            )

            # =========================================
            # DISPLAY RESULT
            # =========================================

            st.subheader("✅ Prediction Results")

            display_columns = [
                "education_mother",
                "education_father",
                "age_years",
                "gender",
                "predicted_iq_category",
                "confidence_score (%)"
            ]

            st.dataframe(
                result_df[display_columns],
                use_container_width=True
            )

            # =========================================
            # DISTRIBUTION CHART
            # =========================================

            st.subheader("📈 Prediction Distribution")

            category_counts = (
                result_df["predicted_iq_category"]
                .value_counts()
            )

            fig2, ax2 = plt.subplots(
                figsize=(10, 5)
            )

            bars = ax2.bar(
                category_counts.index,
                category_counts.values
            )

            # Add labels above bars
            for bar in bars:

                height = bar.get_height()

                ax2.text(
                    bar.get_x() + bar.get_width()/2,
                    height + 0.2,
                    str(height),
                    ha='center'
                )

            ax2.set_ylabel("Number of Data")
            ax2.set_xlabel("IQ Category")

            plt.xticks(rotation=10)

            st.pyplot(fig2)

            # =========================================
            # DOWNLOAD
            # =========================================

            csv = (
                result_df
                .to_csv(index=False)
                .encode("utf-8")
            )

            st.download_button(
                label="⬇ Download Prediction Result",
                data=csv,
                file_name="prediction_result.csv",
                mime="text/csv"
            )

        except Exception as e:

            st.error(f"ERROR:\n\n{str(e)}")

# =========================================
# ABOUT MODEL
# =========================================

elif menu == "About Model":

    st.subheader("📘 About Model")

    st.markdown("""
    ### Model Information

    - Algorithm : Multilayer Perceptron (MLP)
    - Hidden Layer : (10, 6)
    - Activation : ReLU
    - Optimizer : Adam
    - Feature Scaling : StandardScaler
    - Oversampling : SMOTE

    ### Input Features

    - Mother Education
    - Father Education
    - Age
    - Gender

    ### Output Classes

    - Moderate ID
    - Mild ID
    - Below Average
    - Average
    - Above Average
    """)

    st.info(
        "Model trained using Stanford-Binet Intelligence Scales dataset."
    )

    st.warning(
        """
Disclaimer:

Model dibuat menggunakan dataset penelitian terbatas
dan tidak dapat digunakan sebagai alat diagnosis klinis.

Confidence score merupakan probabilitas model,
bukan probabilitas klinis nyata.
        """
    )