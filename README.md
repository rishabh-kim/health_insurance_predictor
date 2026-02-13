# Health Insurance Cost Prediction

A Streamlit web application that predicts health insurance costs based on personal health information.

## Features

- **Interactive Web Interface**: User-friendly form to input health metrics
- **Machine Learning Model**: Pre-trained model for accurate cost predictions
- **Real-time Predictions**: Get instant insurance cost estimates
- **Scalable Deployment**: Hosted on Render for easy access

## Project Structure

```
.
├── app.py                              # Main Streamlit application
├── insurance.csv                       # Training dataset
├── best_model.pkl                      # Pre-trained ML model
├── scaler.pkl                          # Feature scaler
├── gender_label_encoder.pkl            # Gender encoder
├── diabetic_label_encoder.pkl          # Diabetic status encoder
├── smoker_label_encoder.pkl            # Smoking status encoder
├── exploratory_data_analysis.ipynb     # EDA notebook
├── requirements.txt                    # Python dependencies
├── render.yaml                         # Render deployment config
└── README.md                           # This file
```

## Installation

### Local Development

1. Clone the repository:
```bash
git clone <repository-url>
cd health-insurance-cost-prediction
```

2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Run the application:
```bash
streamlit run app.py
```

The app will be available at `http://localhost:8501`

## Deployment on Render

### Prerequisites

- GitHub account
- Render account (https://render.com)

### Steps to Deploy

1. **Push to GitHub**
   ```bash
   git add .
   git commit -m "Initial commit"
   git push origin main
   ```

2. **Connect to Render**
   - Go to https://render.com/dashboard
   - Click "New +" → "Web Service"
   - Connect your GitHub repository
   - Select the branch to deploy

3. **Configure Settings**
   - **Name**: health-insurance-prediction
   - **Environment**: Python 3
   - **Build Command**: `pip install -r requirements.txt`
   - **Start Command**: `streamlit run app.py --server.port=10000 --server.address=0.0.0.0`
   - **Free Plan**: Select for testing

4. **Environment Variables** (if needed)
   - `STREAMLIT_SERVER_HEADLESS=true`

5. **Deploy**
   - Click "Create Web Service"
   - Wait for build to complete

Your app will be live at: `https://health-insurance-prediction.onrender.com`

## How to Use

1. Enter your health information:
   - **Age**: Your current age
   - **BMI**: Body Mass Index
   - **Blood Pressure**: Your blood pressure reading
   - **Gender**: Select your gender
   - **Diabetic Status**: Are you diabetic?
   - **Number of Children**: How many dependents
   - **Smoker Status**: Do you smoke?

2. Click **"Predict Payment"**

3. View your estimated annual health insurance cost

## Model Details

The predictive model uses:
- **Algorithm**: Trained machine learning model
- **Features**: Age, Gender, BMI, Blood Pressure, Diabetic Status, Children, Smoker Status
- **Output**: Estimated annual health insurance cost (USD)

## Technologies Used

- **Streamlit**: Web framework
- **Pandas**: Data manipulation
- **NumPy**: Numerical computations
- **Scikit-learn**: Machine learning
- **Joblib**: Model serialization

## Troubleshooting

### App runs locally but not on Render

Common issues and solutions:

1. **Missing requirements.txt**
   - Ensure all Python packages are listed
   - Run: `pip freeze > requirements.txt`

2. **File Path Issues**
   - Use relative paths for model files
   - All .pkl files must be in the same directory as app.py

3. **Port Configuration**
   - Render uses port 10000
   - Start command should include: `--server.port=10000 --server.address=0.0.0.0`

4. **Streamlit Server Settings**
   - Add environment variables for headless mode
   - Create `.streamlit/config.toml` for additional config

5. **Build Failures**
   - Check Render deployment logs
   - Verify all dependencies are in requirements.txt
   - Ensure Python version compatibility

## Contributing

Feel free to fork this project and submit pull requests.

## License

This project is open source and available under the MIT License.

## Support

For issues and questions, please open a GitHub issue or contact the maintainer.
