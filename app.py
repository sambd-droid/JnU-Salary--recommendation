import streamlit as st
import pandas as pd
import joblib

st.set_page_config(
    page_title="Salary Prediction App",
    page_icon="💼",
    layout="centered"
)

@st.cache_resource
def load_artifact():
    return joblib.load("salary_random_forest.joblib")

artifact = load_artifact()
pipeline = artifact["pipeline"]
metadata = artifact["metadata"]

st.title("💼 Employee Salary Prediction")
st.write("Enter employee information to estimate salary with Random Forest Regression.")

age = st.number_input(
    "Age",
    min_value=float(metadata["age_min"]),
    max_value=float(metadata["age_max"]),
    value=float((metadata["age_min"] + metadata["age_max"]) / 2),
    step=1.0
)

gender = st.selectbox(
    "Gender",
    metadata["gender_options"]
)

education_level = st.selectbox(
    "Education Level",
    metadata["education_options"]
)

job_title = st.selectbox(
    "Job Title",
    metadata["job_title_options"]
)

years_experience = st.number_input(
    "Years of Experience",
    min_value=float(metadata["experience_min"]),
    max_value=float(metadata["experience_max"]),
    value=float(
        (metadata["experience_min"] + metadata["experience_max"]) / 2
    ),
    step=0.5
)

if st.button("Predict Salary"):
    input_data = pd.DataFrame(
        {
            "Age": [age],
            "Gender": [gender],
            "Education Level": [education_level],
            "Job Title": [job_title],
            "Years of Experience": [years_experience]
        }
    )

    prediction = pipeline.predict(input_data)[0]

    st.success(f"Predicted Salary: ${prediction:,.2f}")

st.caption(
    "This academic project uses a Random Forest Regression pipeline. "
    "Predictions are estimates and should not be used as the sole basis for "
    "employment or compensation decisions."
)
