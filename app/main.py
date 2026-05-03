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

print("Loading IMPROVED Keras model (98.73% accuracy)...")

# Load the improved Keras model (98.73% accuracy)
try:
    keras_model = tf.keras.models.load_model("app/keras_model_improved.keras")
    print("[OK] Improved Keras model loaded (98.73% accuracy)")
except Exception as e:
    print(f"[ERROR] Error loading improved Keras model: {e}")
    keras_model = None

# Load the improved StandardScaler (for 10 features)
try:
    scaler = joblib.load("app/scaler_improved.pkl")
    print("[OK] Improved StandardScaler loaded (10 features)")
except Exception as e:
    print(f"[ERROR] Error loading improved scaler: {e}")
    scaler = None


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

    predictions = []
    prediction_labels = []
    
    for item in data:
        # Create feature vector with 10 features (Body Temp removed)
        # Order: Age, Systolic BP, Diastolic, BS, BMI, Previous Complications,
        #        Preexisting Diabetes, Gestational Diabetes, Mental Health, Heart Rate
        
        features = [
            item.Age,
            item.Systolic_BP if item.Systolic_BP is not None else 120.0,
            item.Diastolic if item.Diastolic is not None else 80.0,
            item.BS if item.BS is not None else 5.0,
            item.BMI,
            item.Previous_Complications if item.Previous_Complications is not None else 0,
            item.Preexisting_Diabetes if item.Preexisting_Diabetes is not None else 0,
            item.Gestational_Diabetes if item.Gestational_Diabetes is not None else 0,
            item.Mental_Health if item.Mental_Health is not None else 0,
            item.Heart_Rate if item.Heart_Rate is not None else 75.0,
        ]
        
        # Scale features using the improved scaler
        features_array = np.array(features).reshape(1, -1)
        features_scaled = scaler.transform(features_array)
        
        # Get prediction from improved Keras model
        prob = keras_model.predict(features_scaled, verbose=0)[0, 0]
        pred = 1 if prob > 0.5 else 0
        
        predictions.append(pred)
        prediction_labels.append(RISK_LABELS.get(pred, "Unknown"))
    
    return {
        "prediction": predictions,
        "prediction_label": prediction_labels,
    }

