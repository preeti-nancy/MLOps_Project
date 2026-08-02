from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

BASE_DIR = Path(__file__).resolve().parent

st.set_page_config(page_title="Wellness Tourism Predictor", page_icon="✈️")
st.title("Wellness Tourism Package Purchase Predictor")
st.caption("Predict whether a customer is likely to purchase the Wellness Tourism Package.")


@st.cache_resource
def load_artifacts():
    model = joblib.load(BASE_DIR / "model.pkl")
    feature_columns = joblib.load(BASE_DIR / "feature_columns.pkl")
    return model, feature_columns


def build_input_frame(raw_input, feature_columns):
    encoded = pd.get_dummies(pd.DataFrame([raw_input]))
    for column in feature_columns:
        if column not in encoded.columns:
            encoded[column] = 0
    return encoded[feature_columns]


model, feature_columns = load_artifacts()

with st.form("prediction_form"):
    st.subheader("Customer Profile")
    col1, col2 = st.columns(2)

    with col1:
        age = st.number_input("Age", min_value=18, max_value=70, value=36)
        typeof_contact = st.selectbox("Type of Contact", ["Self Enquiry", "Company Invited"])
        city_tier = st.selectbox("City Tier", [1, 2, 3], index=0)
        occupation = st.selectbox("Occupation", ["Salaried", "Small Business", "Free Lancer", "Large Business"])
        gender = st.selectbox("Gender", ["Male", "Female"])
        num_persons = st.number_input("Number of Persons Visiting", min_value=1, max_value=5, value=2)
        num_followups = st.number_input("Number of Followups", min_value=1, max_value=6, value=3)

    with col2:
        product_pitched = st.selectbox("Product Pitched", ["Basic", "Standard", "Deluxe", "Super Deluxe", "King"])
        preferred_star = st.number_input("Preferred Property Star", min_value=1, max_value=5, value=3)
        marital_status = st.selectbox("Marital Status", ["Single", "Married", "Divorced", "Unmarried"])
        num_trips = st.number_input("Number of Trips (annual)", min_value=1, max_value=10, value=3)
        passport = st.selectbox("Passport Holder", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
        pitch_score = st.slider("Pitch Satisfaction Score", 1, 5, 3)
        own_car = st.selectbox("Own Car", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No")

    col3, col4 = st.columns(2)
    with col3:
        num_children = st.number_input("Number of Children Visiting", min_value=0, max_value=3, value=0)
        designation = st.selectbox("Designation", ["Executive", "Manager", "Senior Manager", "AVP", "VP"])
    with col4:
        monthly_income = st.number_input("Monthly Income", min_value=1000, max_value=100000, value=22000)
        duration_pitch = st.number_input("Duration of Pitch (minutes)", min_value=5.0, max_value=60.0, value=15.0)

    submitted = st.form_submit_button("Predict Purchase Likelihood")

if submitted:
    raw_input = {
        "Age": age,
        "CityTier": city_tier,
        "DurationOfPitch": duration_pitch,
        "NumberOfPersonVisiting": num_persons,
        "NumberOfFollowups": num_followups,
        "PreferredPropertyStar": preferred_star,
        "NumberOfTrips": num_trips,
        "Passport": passport,
        "PitchSatisfactionScore": pitch_score,
        "OwnCar": own_car,
        "NumberOfChildrenVisiting": num_children,
        "MonthlyIncome": monthly_income,
        "TypeofContact": typeof_contact,
        "Occupation": occupation,
        "Gender": gender,
        "ProductPitched": product_pitched,
        "MaritalStatus": marital_status,
        "Designation": designation,
    }

    input_df = build_input_frame(raw_input, feature_columns)
    prediction = model.predict(input_df)[0]
    probability = model.predict_proba(input_df)[0][1]

    st.subheader("Prediction Result")
    if prediction == 1:
        st.success(f"Likely to purchase (probability: {probability:.1%})")
    else:
        st.error(f"Unlikely to purchase (probability: {probability:.1%})")
