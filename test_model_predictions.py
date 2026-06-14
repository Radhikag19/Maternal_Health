"""
Test script to validate ML model predictions for various risk scenarios
"""
import joblib
import numpy as np
import tensorflow as tf
from pathlib import Path

# Load model and scaler
print("=" * 80)
print("LOADING MODEL & SCALER")
print("=" * 80)

keras_model = tf.keras.models.load_model("app/keras_model_improved.keras")
scaler = joblib.load("app/scaler_improved.pkl")
print(f"✓ Model loaded: {keras_model}")
print(f"✓ Scaler loaded: {scaler}")
print(f"✓ Scaler params: Mean={scaler.mean_}, Scale={scaler.scale_}")

RISK_LABELS = {0: "Low Risk", 1: "High Risk"}

def predict_risk(age, systolic_bp, diastolic, bs, bmi, prev_comp, preex_diab, 
                 gest_diab, mental_health, heart_rate):
    """Make prediction with 10 features"""
    features = [age, systolic_bp, diastolic, bs, bmi, prev_comp, preex_diab, 
                gest_diab, mental_health, heart_rate]
    features_array = np.array(features).reshape(1, -1)
    features_scaled = scaler.transform(features_array)
    prob = keras_model.predict(features_scaled, verbose=0)[0, 0]
    pred = 1 if prob > 0.5 else 0
    label = RISK_LABELS[pred]
    return pred, prob, label

# Test cases: (description, age, sys_bp, dia_bp, bs, bmi, prev_comp, preex_diab, gest_diab, mental, hr)
test_cases = [
    ("✓ HEALTHY (Low Risk Expected)", 25, 120, 80, 100, 22, 0, 0, 0, 0, 72),
    ("✓ HEALTHY (No Complications)", 30, 118, 78, 105, 23, 0, 0, 0, 0, 75),
    
    ("⚠ HIGH BP (Risk Indicator)", 28, 160, 95, 100, 22, 0, 0, 0, 0, 80),
    ("⚠ HIGH HEART RATE", 26, 125, 82, 100, 22, 0, 0, 0, 0, 120),
    ("⚠ HIGH BMI", 32, 125, 82, 110, 32, 0, 0, 0, 0, 80),
    ("⚠ ELEVATED BS", 28, 120, 80, 150, 22, 0, 0, 0, 0, 75),
    
    ("🔴 WITH DIABETES", 30, 125, 85, 130, 26, 1, 1, 0, 0, 85),
    ("🔴 WITH ALL COMPLICATIONS", 35, 140, 90, 140, 28, 1, 1, 1, 1, 95),
    ("🔴 SEVERE: Old Age + High BP + Diabetes", 45, 170, 100, 150, 30, 1, 1, 1, 0, 105),
    ("🔴 SEVERE: Multiple Risk Factors", 38, 165, 95, 145, 32, 1, 1, 1, 1, 110),
]

print("\n" + "=" * 80)
print("TESTING MODEL WITH VARIOUS RISK SCENARIOS")
print("=" * 80)

for desc, age, sys, dia, bs, bmi, pc, pd, gd, mh, hr in test_cases:
    pred, prob, label = predict_risk(age, sys, dia, bs, bmi, pc, pd, gd, mh, hr)
    print(f"\n{desc}")
    print(f"  Input: Age={age}, BP={sys}/{dia}, BS={bs}, BMI={bmi}, HR={hr}")
    print(f"         Complications: Prev={pc}, PreexDiab={pd}, GestDiab={gd}, Mental={mh}")
    print(f"  → Prediction: {label} (Probability: {prob:.4f})")

# Analyze actual data from CSV
print("\n" + "=" * 80)
print("ANALYZING ACTUAL APP SUBMISSIONS")
print("=" * 80)

import pandas as pd
csv_path = Path("data/captured_vitals.csv")
if csv_path.exists():
    df = pd.read_csv(csv_path)
    print(f"\nTotal submissions: {len(df)}")
    print(f"Low Risk: {len(df[df['prediction'] == 0])}")
    print(f"High Risk: {len(df[df['prediction'] == 1])}")
    
    print("\nDetailed Predictions:")
    for idx, row in df.iterrows():
        print(f"\n  Submission {idx+1}:")
        print(f"    Patient: Age={int(row['Age'])}, BP={row['Systolic_BP']}/{row['Diastolic']}, BS={row['BS']}")
        print(f"    BMI={row['BMI']}, HR={row['Heart_Rate']}, State={row['state']}")
        print(f"    Complications: Prev={row['Previous_Complications']}, PreexDiab={row['Preexisting_Diabetes']}, GestDiab={row['Gestational_Diabetes']}, Mental={row['Mental_Health']}")
        print(f"    → Predicted: {row['prediction_label']}")
else:
    print("No captured_vitals.csv found yet")

print("\n" + "=" * 80)
print("MODEL VALIDATION COMPLETE")
print("=" * 80)
