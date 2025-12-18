# 1 Good (Lower Risk) 0 Bad Higher risk
import streamlit as st
import pandas as pd
import joblib 
import os

# make model/encoder loading robust and show helpful error in the app
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_CANDIDATES = ["credit_risk_model.pkl", "best_model.pkl"]

# --- safe model loader (replace existing model-path / joblib.load lines) ---
candidate_paths = [os.path.join(BASE_DIR, name) for name in MODEL_CANDIDATES]
model_path = next((p for p in candidate_paths if os.path.exists(p)), None)

if model_path is None:
    st.error(f"Model file not found. Expected one of: {', '.join(MODEL_CANDIDATES)} in {BASE_DIR}")
    st.info("Generate the model by running in a terminal from the project folder:\n  py -3 train.py")
    st.stop()

try:
    model = joblib.load(model_path)
except Exception as e:
    st.error(f"Failed to load model at {model_path}: {e}")
    st.stop()

# load encoders, fail with clear message if missing
ENCODER_COLS = ["Sex", "Housing", "Saving accounts", "Checking account"]
encoders = {}
missing = []
for col in ENCODER_COLS:
    path = os.path.join(BASE_DIR, f"label_encoder_{col}.pkl")
    if os.path.exists(path):
        encoders[col] = joblib.load(path)
    else:
        missing.append(col)

if missing:
    st.error(f"Missing label encoder files for: {', '.join(missing)}")
    st.info("Run the training script to produce label_encoder_<col>.pkl files: py -3 train.py")
    st.stop()

st.title("Credit Risk Prediction")
st.write("Enter the details below to predict credit risk.")

# Input fields
age = st.number_input("Age", min_value=18, max_value=80, value=30)
sex = st.selectbox("Sex", ["male", "female"])
job = st.number_input("Job (0-3)", min_value=0, max_value=3, value=1)
housing = st.selectbox("Housing", ["own", "rent", "free"])
saving_accounts = st.selectbox("Saving accounts", ["little", "moderate", "rich", "quite rich", "unknown"])
checking_account = st.selectbox("Checking account", ["little", "moderate", "rich", "unknown"])
credit_amount = st.number_input("Credit Amount", min_value=100.0, max_value=20000.0, value=1000.0)
duration = st.number_input("Duration (months)", min_value=4, max_value=72, value=12)

input_df = pd.DataFrame({
    "Age": [age],
    "Sex": [ encoders["Sex"].transform([sex]) [0]],
    "Job": [job],
    "Housing": [ encoders["Housing"].transform([housing]) [0]],
    "Saving accounts": [ encoders["Saving accounts"].transform([saving_accounts]) [0]],
    "Checking account": [ encoders["Checking account"].transform([checking_account]) [0]],
    "Credit amount": [credit_amount],
    "Duration": [duration]
})

if st.button("Predict Risk"):
    pred = model.predict(input_df)[0]


    if pred == 1:
        st.success("The predicted credit risk is: Good (Lower Risk)")
    else:
        st.error("The predicted credit risk is: Bad (Higher Risk)")