#### Road Accident Risk Prediction

## About the Project
This project predicts the risk level of road accidents using historical accident data along with weather, road and time-related information.

# The project classifies road locations into three risk levels:
- Low Risk
- Medium Risk
- High Risk
    
## A Streamlit web application is created to visualize accident risk on an interactive map and predict risk based on different road and environmental conditions.

## Technologies Used
- Python
- Pandas
- Scikit-learn
- XGBoost
- Streamlit
- Folium
- Matplotlib
  
## Machine Learning Models

The following models were compared:
- Logistic Regression
- Decision Tree
- KNN
- Random Forest
- XGBoost
  
XGBoost was selected based on the test Macro F1 score.

### Features
- Interactive accident risk map
- Low, Medium and High risk classification
- Road risk prediction
- Weather and road condition based prediction
- Model performance comparison
 
### Live Demo

[👉 Click here to open the Live Demo](https://road-accident-risk-prediction-system.streamlit.app/)

## How to Run

1. Clone the repository
git clone https://github.com/nirjarakhante/Road-Accident-Risk-Prediction.git
2. Open the project folder
cd Road-Accident-Risk-Prediction
3. Install the required libraries
pip install -r requirements.txt
4. Run the Streamlit application
streamlit run road-accident-risk-predictor/app/streamlit_app.py
The application will open in your browser.


#### Project By
Nirjara Khante
