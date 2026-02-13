from flask import Flask, render_template, request, jsonify
import pandas as pd
import numpy as np
import joblib
import os

app = Flask(__name__)

# Get the directory where the script is located
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# Load models
try:
    scaler = joblib.load(os.path.join(SCRIPT_DIR, 'scaler.pkl'))
    le_gender = joblib.load(os.path.join(SCRIPT_DIR, 'gender_label_encoder.pkl'))
    le_diabetic = joblib.load(os.path.join(SCRIPT_DIR, 'diabetic_label_encoder.pkl'))
    le_smoker = joblib.load(os.path.join(SCRIPT_DIR, 'smoker_label_encoder.pkl'))
    model = joblib.load(os.path.join(SCRIPT_DIR, 'best_model.pkl'))
    models_loaded = True
except Exception as e:
    print(f"Error loading models: {e}")
    models_loaded = False


@app.route('/')
def index():
    """Render the main page"""
    return render_template('index.html')


@app.route('/api/predict', methods=['POST'])
def predict():
    """API endpoint for prediction"""
    if not models_loaded:
        return jsonify({'error': 'Models not loaded'}), 500
    
    try:
        data = request.json
        
        # Create input dataframe
        input_data = pd.DataFrame({
            'age': [float(data['age'])],
            'gender': [data['gender']],
            'bmi': [float(data['bmi'])],
            'bloodpressure': [float(data['bloodpressure'])],
            'diabetic': [data['diabetic']],
            'children': [int(data['children'])],
            'smoker': [data['smoker']]
        })
        
        # Encode categorical variables
        input_data['gender'] = le_gender.transform(input_data['gender'])
        input_data['diabetic'] = le_diabetic.transform(input_data['diabetic'])
        input_data['smoker'] = le_smoker.transform(input_data['smoker'])
        
        # Scale numerical columns
        numerical_cols = ['age', 'bmi', 'bloodpressure', 'children']
        input_data[numerical_cols] = scaler.transform(input_data[numerical_cols])
        
        # Make prediction
        prediction = model.predict(input_data)[0]
        
        return jsonify({
            'success': True,
            'prediction': f"${prediction:,.2f}",
            'prediction_value': float(prediction)
        })
    
    except Exception as e:
        return jsonify({'error': str(e), 'success': False}), 400


@app.route('/api/categories', methods=['GET'])
def get_categories():
    """Get available categories for dropdowns"""
    if not models_loaded:
        return jsonify({'error': 'Models not loaded'}), 500
    
    return jsonify({
        'gender': list(le_gender.classes_),
        'diabetic': list(le_diabetic.classes_),
        'smoker': list(le_smoker.classes_)
    })


@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({'status': 'ok', 'models_loaded': models_loaded})


if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(debug=False, host='0.0.0.0', port=port)