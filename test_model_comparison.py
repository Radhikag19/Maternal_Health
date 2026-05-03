"""
Comparison script: Test improved model vs original model
"""
import pandas as pd
import numpy as np
import tensorflow as tf
import joblib
from sklearn.metrics import accuracy_score

print("=" * 70)
print("TESTING IMPROVED MODEL")
print("=" * 70)

# Load dataset
df = pd.read_csv('data/Dataset - Updated.csv')
df_clean = df.dropna(subset=['Heart Rate', 'BS', 'Previous Complications', 'Preexisting Diabetes', 'Risk Level'])
df_clean['Risk Level'] = pd.factorize(df_clean['Risk Level'])[0]
df_clean = df_clean.copy()
df_clean['Risk Level'] = df_clean['Risk Level'].apply(lambda col: 1 if col == 0 else 0)

# Take 10 samples for comparison
df_test = df_clean.sample(n=10, random_state=42)
print(f"\nTest samples:")
print(df_test[['Age', 'BS', 'BMI', 'Heart Rate', 'Risk Level']].to_string())

# Prepare features (10 features, Body Temp removed)
X_test = df_test.drop(['Risk Level', 'Body Temp'], axis=1).fillna(df_clean.drop(['Risk Level', 'Body Temp'], axis=1).median())
y_test = df_test['Risk Level'].values

# Load improved model and scaler
print("\n[1] Loading improved model and scaler...")
improved_model = tf.keras.models.load_model('app/keras_model_improved.keras')
improved_scaler = joblib.load('app/scaler_improved.pkl')

# Load original model and scaler
print("[2] Loading original model and scaler...")
original_model = tf.keras.models.load_model('app/keras_model.keras')
original_scaler = joblib.load('app/scaler.pkl')
stage1_model = joblib.load('app/stage1_model.pkl')
stage2_model = joblib.load('app/stage2_model.pkl')
stage3_model = joblib.load('app/stage3_model.pkl')

print("\n" + "=" * 70)
print("PREDICTIONS COMPARISON")
print("=" * 70)

improved_preds = []
original_preds = []

for idx, (i, row) in enumerate(X_test.iterrows()):
    print(f"\n--- Sample {idx+1} ---")
    print(f"Age={row['Age']}, BS={row['BS']:.1f}, BMI={row['BMI']:.1f}, HR={row['Heart Rate']:.1f}")
    print(f"Actual: {'Low Risk' if y_test[idx] == 1 else 'High Risk'}")
    
    # IMPROVED MODEL (10 features directly)
    X_scaled = improved_scaler.transform(row.values.reshape(1, -1))
    improved_prob = improved_model.predict(X_scaled, verbose=0)[0, 0]
    improved_pred = 1 if improved_prob > 0.5 else 0
    improved_preds.append(improved_pred)
    print(f"Improved: {'Low Risk' if improved_pred == 1 else 'High Risk'} (prob={improved_prob:.4f})")
    
    # ORIGINAL MODEL (3-stage preprocessing + 7 features)
    bs_val = row['BS']
    bmi_val = row['BMI']
    systolic_val = row['Systolic BP'] if row['Systolic BP'] != -999 else 120.0
    diastolic_val = row['Diastolic'] if row['Diastolic'] != -999 else 80.0
    hr_val = row['Heart Rate']
    
    is_low_bmi = 1 if bmi_val < 18.5 else 0
    is_high_bmi = 1 if bmi_val > 30 else 0
    is_low_bp = 1 if (systolic_val < 90 or diastolic_val < 60) else 0
    is_high_bp = 1 if (systolic_val > 140 or diastolic_val > 90) else 0
    is_high_bs = 1 if bs_val > 7.8 else 0
    is_high_hr = 1 if hr_val > 100 else 0
    is_low_hr = 1 if hr_val < 60 else 0
    
    stage1_input = row[['Previous Complications', 'Preexisting Diabetes', 'Gestational Diabetes', 'Mental Health']].values.reshape(1, -1)
    risk_score = stage1_model.predict_proba(stage1_input)[0, 1]
    
    stage2_input = np.array([[is_low_bmi, is_high_bmi, is_low_bp, is_high_bp, is_high_bs, is_high_hr, is_low_hr]])
    risk_score_abn = stage2_model.predict_proba(stage2_input)[0, 1]
    
    stage3_input = np.array([[risk_score, risk_score_abn]])
    final_risk_score = stage3_model.predict_proba(stage3_input)[0, 1]
    
    vitals = np.array([[bs_val, bmi_val, row['Age'], hr_val, systolic_val, diastolic_val]])
    vitals_scaled = original_scaler.transform(vitals)
    
    keras_input = np.hstack([vitals_scaled, np.array([[final_risk_score]])])
    original_prob = original_model.predict(keras_input, verbose=0)[0, 0]
    original_pred = 1 if original_prob > 0.5 else 0
    original_preds.append(original_pred)
    print(f"Original:  {'Low Risk' if original_pred == 1 else 'High Risk'} (prob={original_prob:.4f})")

print("\n" + "=" * 70)
print("ACCURACY SUMMARY")
print("=" * 70)

improved_acc = accuracy_score(y_test, improved_preds)
original_acc = accuracy_score(y_test, original_preds)

print(f"Improved Model Accuracy: {improved_acc*100:.1f}%")
print(f"Original Model Accuracy: {original_acc*100:.1f}%")
print(f"Improvement: {(improved_acc - original_acc)*100:+.1f}%")

if improved_acc == original_acc and improved_acc == 1.0:
    print("\n✓ Both models are highly accurate on this sample")
elif improved_acc > original_acc:
    print(f"\n✓ Improved model is better by {(improved_acc - original_acc)*100:.1f}%")
else:
    print(f"\n⚠ Original model better by {(original_acc - improved_acc)*100:.1f}% (sample variance)")
