import streamlit as st
import pandas as pd
import joblib

# Load model and feature names
model = joblib.load("patient_readmission_xgb.pkl")
feature_columns = joblib.load("feature_columns.pkl")

st.set_page_config(
    page_title="Patient Readmission Risk Scorer",
    page_icon="🏥",
    layout="wide"
)

st.title("🏥 Patient Readmission Risk Scorer")

st.markdown("""
Upload a CSV file containing patient records and the model will predict
the probability of hospital readmission.
""")

uploaded_file = st.file_uploader(
    "Upload Patient CSV File",
    type=["csv"]
)

if uploaded_file is not None:

    try:

        df = pd.read_csv(uploaded_file)

        st.subheader("Uploaded Data")
        st.dataframe(df.head())

        # Check columns
        missing_cols = set(feature_columns) - set(df.columns)

        if len(missing_cols) > 0:

            st.error(
                f"Missing Columns:\n\n{list(missing_cols)}"
            )

        else:

            df = df[feature_columns]

            predictions = model.predict(df)

            probabilities = model.predict_proba(df)[:,1]

            result_df = df.copy()

            result_df["Prediction"] = predictions

            result_df["Readmission Risk (%)"] = (
                probabilities * 100
            ).round(2)

            result_df["Risk Category"] = result_df[
                "Readmission Risk (%)"
            ].apply(
                lambda x:
                "Low Risk" if x < 30 else
                "Medium Risk" if x < 70 else
                "High Risk"
            )

            st.subheader("Prediction Results")

            st.dataframe(result_df)

            csv = result_df.to_csv(index=False)

            st.download_button(
                label="Download Results CSV",
                data=csv,
                file_name="patient_predictions.csv",
                mime="text/csv"
            )

            st.success(
                f"Successfully predicted {len(result_df)} patients."
            )

    except Exception as e:

        st.error(str(e))