# Streamlit app: collects customer details and predicts whether the customer
# is likely to buy the Wellness Tourism Package.
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

MODEL_PATH = Path(__file__).parent / "best_model.joblib"

st.set_page_config(page_title="Wellness Package Predictor")


@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)


model = load_model()

st.title("Wellness Tourism Package - Purchase Predictor")
st.write(
    "Enter the customer's details and the pitch information to estimate how "
    "likely they are to buy the package before the next follow-up call."
)

st.subheader("Customer details")
c1, c2, c3 = st.columns(3)
with c1:
    age = st.number_input("Age", 18, 80, 35)
    gender = st.selectbox("Gender", ["Male", "Female"])
    marital = st.selectbox("Marital status", ["Single", "Married", "Divorced", "Unmarried"])
    occupation = st.selectbox("Occupation", ["Salaried", "Small Business", "Large Business", "Free Lancer"])
with c2:
    designation = st.selectbox("Designation", ["Executive", "Manager", "Senior Manager", "AVP", "VP"])
    income = st.number_input("Monthly income", 1000, 150000, 20000, step=500)
    city_tier = st.selectbox("City tier", [1, 2, 3])
    trips = st.number_input("Trips per year", 0, 25, 3)
with c3:
    passport = st.selectbox("Holds a passport", ["Yes", "No"])
    own_car = st.selectbox("Owns a car", ["Yes", "No"])
    persons = st.number_input("People visiting", 1, 10, 3)
    children = st.number_input("Children visiting (under 5)", 0, 5, 0)

st.subheader("Pitch details")
c4, c5, c6 = st.columns(3)
with c4:
    contact = st.selectbox("Type of contact", ["Self Enquiry", "Company Invited"])
    product = st.selectbox("Product pitched", ["Basic", "Standard", "Deluxe", "Super Deluxe", "King"])
with c5:
    pitch_minutes = st.number_input("Pitch duration (minutes)", 1, 150, 15)
    followups = st.number_input("Number of follow-ups", 0, 10, 3)
with c6:
    star = st.selectbox("Preferred property star", [3, 4, 5])
    satisfaction = st.slider("Pitch satisfaction score", 1, 5, 3)

# one row, same column names the model was trained on
input_df = pd.DataFrame([{
    "Age": age,
    "TypeofContact": contact,
    "CityTier": city_tier,
    "DurationOfPitch": pitch_minutes,
    "Occupation": occupation,
    "Gender": gender,
    "NumberOfPersonVisiting": persons,
    "NumberOfFollowups": followups,
    "ProductPitched": product,
    "PreferredPropertyStar": star,
    "MaritalStatus": marital,
    "NumberOfTrips": trips,
    "Passport": 1 if passport == "Yes" else 0,
    "PitchSatisfactionScore": satisfaction,
    "OwnCar": 1 if own_car == "Yes" else 0,
    "NumberOfChildrenVisiting": children,
    "Designation": designation,
    "MonthlyIncome": income,
}])

with st.expander("Input sent to the model"):
    st.dataframe(input_df)

if st.button("Predict"):
    prob = model.predict_proba(input_df)[0, 1]
    if prob >= 0.5:
        st.success(f"Likely to purchase (probability {prob:.1%})")
    else:
        st.warning(f"Unlikely to purchase (probability {prob:.1%})")
