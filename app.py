import streamlit as st
import pickle
import numpy as np
import pandas as pd

# Load the trained logistic regression model
@st.cache_resource
def load_model():
    with open("linkedin_model.pkl", "rb") as file:
        model = pickle.load(file)
    return model

# Load the model
model = load_model()

# Prediction function
def make_prediction(features):
    prediction = model.predict([features])
    probability = model.predict_proba([features])[0][1]  # Probability of being a LinkedIn user
    return prediction[0], probability

# App Title
st.title("LinkedIn Usage Predictor")
st.write(
    """
    This app predicts if a person is a LinkedIn user based on demographic and lifestyle features.
    Provide your inputs below to see the prediction and the probability!
    """
)

# Input fields
income = st.slider("Income Level (1-9)", min_value=1, max_value=9, step=1)
education = st.slider("Education Level (1=Less than High School, 8=Postgraduate)", min_value=1, max_value=8, step=1)
parent = st.selectbox("Are you a parent?", ["No", "Yes"])
married = st.selectbox("Are you married?", ["No", "Yes"])
female = st.selectbox("Are you female?", ["No", "Yes"])
age = st.slider("Age", min_value=18, max_value=98, step=1)

# Map categorical variables
parent_val = 1 if parent == "Yes" else 0
married_val = 1 if married == "Yes" else 0
female_val = 1 if female == "Yes" else 0

# Prediction button
if st.button("Predict"):
    # Prepare the feature array
    features = [income, education, parent_val, married_val, female_val, age]
    
    # Get prediction and probability
    prediction, probability = make_prediction(features)
    
    # Display results
    if prediction == 1:
        st.success("Prediction: You are classified as a LinkedIn user!")
    else:
        st.warning("Prediction: You are NOT classified as a LinkedIn user.")
    st.info(f"Probability of being a LinkedIn user: {probability:.2f}")

