import streamlit as st
import pandas as pd
import joblib
import numpy as np

# Set page title and layout
st.set_page_config(page_title="Sales Predictor App", layout="centered")

# App title and description
st.title("–️ Advertising Sales Predictor")
st.write("Enter the advertising budgets below to predict estimated Sales.")

# Load the pre-trained linear regression model
@st.cache_resource
def load_model():
    return joblib.load('model.sav')

try:
    model = load_model()
    model_loaded = True
except Exception as e:
    st.error(f"Error loading the model 'model.sav': {e}")
    model_loaded = False

if model_loaded:
    # Create inputs for features
    st.subheader("Input Ad Spend (in thousands of dollars)")
    
    tv = st.number_input("TV Advertising Budget", min_value=0.0, max_value=1000.0, value=20.0, step=1.0)
    radio = st.number_input("Radio Advertising Budget", min_value=0.0, max_value=1000.0, value=12.0, step=1.0)
    newspaper = st.number_input("Newspaper Advertising Budget", min_value=0.0, max_value=1000.0, value=5.0, step=1.0)
    
    # Predict button
    if st.button("Predict Sales"):
        # Construct feature DataFrame matching expected feature names
        input_data = pd.DataFrame([[tv, radio, newspaper]], columns=['TV', 'Radio', 'Newspaper'])
        
        # Get model prediction
        prediction = model.predict(input_data)[0]
        
        # Display result
        st.success(f"### Estimated Sales: {prediction:.4f}")
