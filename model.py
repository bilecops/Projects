import streamlit as st
!pip install joblib
from joblib import load
import numpy as np

# Load the trained Random Forest model
model_path = "C:\\Users\\iqbal\\Downloads\\MDS UM\\Model\\RandomForest_Highrise.joblib"  # Adjust the path as needed
model = load(model_path)

# Title of the app
st.title("Random Forest Prediction for Highrise Prices")

# Input features
feature1 = st.number_input("Log Land/Parcel Area", value=7.5, step=0.1)
feature2 = st.number_input("Log Main Floor Area", value=8.0, step=0.1)
feature3 = st.number_input("Transaction Date (Ordinal)", value=738611, step=1)
feature4 = st.number_input("Property Type", value=1, step=1)
feature5 = st.number_input("Mukim", value=2, step=1)
feature6 = st.number_input("Tenure", value=0, step=1)

# Predict button
if st.button("Predict Price"):
    # Prepare input features as a NumPy array
    input_features = np.array([[feature1, feature2, feature3, feature4, feature5, feature6]])

    # Perform prediction
    predicted_price = model.predict(input_features)[0]

    # Display the predicted price
    st.subheader(f"Predicted Price (RM): {predicted_price:,.2f}")


import os

# Replace with your Streamlit app's file path
streamlit_app_path = "C:\\Users\\iqbal\\Downloads\\MDS UM\\Model\\model.py"

# Run Streamlit without opening a terminal
os.system(f"streamlit run {streamlit_app_path}")
