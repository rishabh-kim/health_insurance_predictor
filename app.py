import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

# Get the directory where the script is located
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# Load models with proper path handling
@st.cache_resource
def load_models():
    scaler = joblib.load(os.path.join(SCRIPT_DIR, 'scaler.pkl'))
    le_gender = joblib.load(os.path.join(SCRIPT_DIR, 'gender_label_encoder.pkl'))
    le_diabetic = joblib.load(os.path.join(SCRIPT_DIR, 'diabetic_label_encoder.pkl'))
    le_smoker = joblib.load(os.path.join(SCRIPT_DIR, 'smoker_label_encoder.pkl'))
    model = joblib.load(os.path.join(SCRIPT_DIR, 'best_model.pkl'))
    return scaler, le_gender, le_diabetic, le_smoker, model

scaler, le_gender, le_diabetic, le_smoker, model = load_models()

st.set_page_config(page_title='Health Insurance Cost Prediction', layout='wide', page_icon='💰')
st.title('Health Insurance Cost Prediction')

st.write('Enter the details below to predict your health insurance cost:')

with st.form('input_form'):
    col1, col2 = st.columns(2)
    with col1:
        age = st.number_input('Age', min_value=0, max_value=120, value=30)
        bmi = st.number_input('BMI', min_value=0.0, max_value=100.0, value=25.0)
        children = st.number_input('Number of Children', min_value=0, max_value=10, value=0)
    with col2:
        blood_pressure = st.number_input('Blood Pressure', min_value=0.0, max_value=300.0, value=120.0)
        gender = st.selectbox('Gender', options = le_gender.classes_)
        diabetic = st.selectbox('Diabetic', options = le_diabetic.classes_)
        smoker = st.selectbox('Smoker', options = le_smoker.classes_)

    submitted = st.form_submit_button(label='Predict Payment')

if submitted:
    input_data = pd.DataFrame({
        'age': [age],
        'gender': [gender],
        'bmi': [bmi],
        'bloodpressure': [blood_pressure],
        'diabetic': [diabetic],
        'children': [children],
        'smoker': [smoker]
    })

    input_data['gender'] = le_gender.transform(input_data['gender'])
    input_data['diabetic'] = le_diabetic.transform(input_data['diabetic'])
    input_data['smoker'] = le_smoker.transform(input_data['smoker'])

    numerical_cols = ['age', 'bmi', 'bloodpressure', 'children']
    input_data[numerical_cols] = scaler.transform(input_data[numerical_cols])

    prediction = model.predict(input_data)[0]

    st.success(f'Estimated Annual Health Insurance Cost: ${prediction:,.2f}')