from fastapi import FastAPI
from pydantic import BaseModel
from typing import List, Union
import joblib
import pandas as pd
import numpy as np
import tensorflow as tf


# Update these labels if your training label encoding changes.
RISK_LABELS = {
    0: "Low Risk",
    1: "High Risk",
}

print("Loading unified Keras model and preprocessing pipeline...")

# Load the unified Keras model
try:
    keras_model = tf.keras.models.load_model("app/keras_model.keras")
    print("✓ Keras model loaded")
except Exception as e:
    print(f"Error loading Keras model: {e}")
    keras_model = None

# Load the StandardScaler
try:
    scaler = joblib.load("app/scaler.pkl")
    print("✓ StandardScaler loaded")
except Exception as e:
    print(f"Error loading scaler: {e}")
    scaler = None

# Load the stage models for feature engineering
try:
    model1 = joblib.load("app/stage1_model.pkl")  # Previous Complications, etc.
    model2 = joblib.load("app/stage2_model.pkl")  # Abnormality flags
    model3 = joblib.load("app/stage3_model.pkl")  # Final ensemble
    print("✓ Stage models loaded")
except Exception as e:
    print(f"Error loading stage models: {e}")
    model1 = model2 = model3 = None


class PatientParams(BaseModel):
    Age: int
    Systolic_BP: float = None
    Diastolic: float = None
    BS: float = None
    Body_Temp: float = None
    BMI: float
    Previous_Complications: int = None
    Preexisting_Diabetes: int = None
    Gestational_Diabetes: int = None
    Mental_Health: int = None
    Heart_Rate: float = None


app = FastAPI()


@app.get("/")
async def root():
    return {"message": "Hello World"}


@app.post("/predict")
async def predict(data: Union[PatientParams, List[PatientParams]]):
    if isinstance(data, PatientParams):
        data = [data]

    # Convert input to DataFrame
    df = pd.DataFrame([item.model_dump() for item in data])
    df.columns = [col.replace('_', ' ').strip() for col in df.columns]
    
    # Handle missing values
    df = df.fillna(-999)
    
    predictions = []
    prediction_labels = []
    
    for idx, row in df.iterrows():
        # ===== STAGE 1: Risk Score from historical factors =====
        col1 = ['Previous Complications', 'Preexisting Diabetes', 'Gestational Diabetes', 'Mental Health']
        stage1_input = row[col1].values.reshape(1, -1)
        risk_score = model1.predict_proba(stage1_input)[0, 1]
        
        # ===== STAGE 2: Risk Score Abn from abnormality flags =====
        bs_val = row['BS'] if row['BS'] != -999 else 5.0
        bmi_val = row['BMI'] if row['BMI'] != -999 else 25.0
        systolic_val = row['Systolic BP'] if row['Systolic BP'] != -999 else 120.0
        diastolic_val = row['Diastolic'] if row['Diastolic'] != -999 else 80.0
        hr_val = row['Heart Rate'] if row['Heart Rate'] != -999 else 75.0
        
        is_low_bmi = 1 if bmi_val < 18.5 else 0
        is_high_bmi = 1 if bmi_val > 30 else 0
        is_low_bp = 1 if (systolic_val < 90 or diastolic_val < 60) else 0
        is_high_bp = 1 if (systolic_val > 140 or diastolic_val > 90) else 0
        is_high_bs = 1 if bs_val > 7.8 else 0
        is_high_hr = 1 if hr_val > 100 else 0
        is_low_hr = 1 if hr_val < 60 else 0
        
        stage2_input = np.array([[is_low_bmi, is_high_bmi, is_low_bp, is_high_bp, is_high_bs, is_high_hr, is_low_hr]])
        risk_score_abn = model2.predict_proba(stage2_input)[0, 1]
        
        # ===== STAGE 3: Final risk score =====
        stage3_input = np.array([[risk_score, risk_score_abn]])
        final_risk_score = model3.predict_proba(stage3_input)[0, 1]
        
        # ===== KERAS NEURAL NETWORK =====
        # Scale the vitals
        vitals = np.array([[bs_val, bmi_val, row['Age'], hr_val, systolic_val, diastolic_val]])
        vitals_scaled = scaler.transform(vitals)
        
        # Create input for Keras model (6 scaled vitals + final_risk_score)
        keras_input = np.hstack([vitals_scaled, np.array([[final_risk_score]])])
        
        # Get prediction
        prob = keras_model.predict(keras_input, verbose=0)[0, 0]
        pred = 1 if prob > 0.5 else 0
        
        predictions.append(pred)
        prediction_labels.append(RISK_LABELS.get(pred, "Unknown"))
    
    return {
        "prediction": predictions,
        "prediction_label": prediction_labels,
    }

