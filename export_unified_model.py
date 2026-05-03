"""
Export the unified Keras model and preprocessing pipeline for backend use.
This script replicates the ondevice-maternalrisk.ipynb pipeline to ensure
both online and offline predictions use the same model.
"""

import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
import tensorflow as tf
from tensorflow.keras import layers, models
import os

# Load the dataset
print("Loading dataset...")
df = pd.read_csv('data/Dataset - Updated.csv')

# Prepare data (same as notebook)
print("Preparing data...")
df2 = df.dropna(subset=['Heart Rate', 'BS', 'Previous Complications', 'Preexisting Diabetes', 'Risk Level'])
df2['Risk Level'] = pd.factorize(df2['Risk Level'])[0]

# Split into train/test
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(df2.drop('Risk Level', axis=1), df2['Risk Level'], test_size=0.2, random_state=42)

X_train_imputed = X_train.copy()
X_test_imputed = X_test.copy()
y_train_imputed = y_train.copy()
y_test_imputed = y_test.copy()

# ===== STAGE 1: First LogisticRegression (Previous Complications, Preexisting Diabetes, Gestational Diabetes, Mental Health) =====
print("Training Stage 1 model...")
col = ['Previous Complications', 'Preexisting Diabetes', 'Gestational Diabetes', 'Mental Health']
X_flag_test_1 = X_train_imputed[col]
model1 = LogisticRegression()
model1.fit(X_flag_test_1, y_train_imputed)
X_train_imputed['Risk Score'] = model1.predict_proba(X_flag_test_1)[:, 1]
X_train_imputed = X_train_imputed.drop(col, axis=1)

# Apply to test
X_test_flag_1 = X_test_imputed[col]
X_test_imputed['Risk Score'] = model1.predict_proba(X_test_flag_1)[:, 1]
X_test_imputed = X_test_imputed.drop(col, axis=1)

# ===== STAGE 2: Second LogisticRegression (Abnormality flags) =====
print("Training Stage 2 model...")
cols = ['is_low_bmi', 'is_high_bmi', 'is_low_bp', 'is_high_bp', 'is_high_bs', 'is_high_hr', 'is_low_hr']

# Create abnormality flags for train
X_train_imputed['is_low_bmi'] = (X_train_imputed['BMI'] < 18.5).astype(int)
X_train_imputed['is_high_bmi'] = (X_train_imputed['BMI'] > 30).astype(int)
X_train_imputed['is_low_bp'] = ((X_train_imputed['Systolic BP'] < 90) | (X_train_imputed['Diastolic'] < 60)).astype(int)
X_train_imputed['is_high_bp'] = ((X_train_imputed['Systolic BP'] > 140) | (X_train_imputed['Diastolic'] > 90)).astype(int)
X_train_imputed['is_high_bs'] = (X_train_imputed['BS'] > 7.8).astype(int)  # Use 7.8 as threshold
X_train_imputed['is_high_hr'] = (X_train_imputed['Heart Rate'] > 100).astype(int)
X_train_imputed['is_low_hr'] = (X_train_imputed['Heart Rate'] < 60).astype(int)

X_flags_2 = X_train_imputed[cols]
model2 = LogisticRegression()
model2.fit(X_flags_2, y_train_imputed)
X_train_imputed['Risk Score Abn'] = model2.predict_proba(X_flags_2)[:, 1]
X_train_imputed = X_train_imputed.drop(cols, axis=1)

# Create abnormality flags for test
X_test_imputed['is_low_bmi'] = (X_test_imputed['BMI'] < 18.5).astype(int)
X_test_imputed['is_high_bmi'] = (X_test_imputed['BMI'] > 30).astype(int)
X_test_imputed['is_low_bp'] = ((X_test_imputed['Systolic BP'] < 90) | (X_test_imputed['Diastolic'] < 60)).astype(int)
X_test_imputed['is_high_bp'] = ((X_test_imputed['Systolic BP'] > 140) | (X_test_imputed['Diastolic'] > 90)).astype(int)
X_test_imputed['is_high_bs'] = (X_test_imputed['BS'] > 7.8).astype(int)
X_test_imputed['is_high_hr'] = (X_test_imputed['Heart Rate'] > 100).astype(int)
X_test_imputed['is_low_hr'] = (X_test_imputed['Heart Rate'] < 60).astype(int)

X_test_flags_2 = X_test_imputed[cols]
X_test_imputed['Risk Score Abn'] = model2.predict_proba(X_test_flags_2)[:, 1]
X_test_imputed = X_test_imputed.drop(cols, axis=1)

# ===== STAGE 3: Third LogisticRegression (Final ensemble) =====
print("Training Stage 3 model...")
X_ult_flag = X_train_imputed[['Risk Score', 'Risk Score Abn']]
model3 = LogisticRegression()
model3.fit(X_ult_flag, y_train_imputed)
X_train_imputed['Final_Risk_Score'] = model3.predict_proba(X_ult_flag)[:, 1]

X_test_flag_final = X_test_imputed[['Risk Score', 'Risk Score Abn']]
X_test_imputed['Final_Risk_Score'] = model3.predict_proba(X_test_flag_final)[:, 1]

# ===== KERAS NEURAL NETWORK =====
print("Training Keras Sequential model...")

# Scale the vitals (only 6 features for the NN)
features_to_scale = ['BS', 'BMI', 'Age', 'Heart Rate', 'Systolic BP', 'Diastolic']
scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train_imputed[features_to_scale])
X_test_scaled = scaler.transform(X_test_imputed[features_to_scale])

# Create input with 7 features (6 scaled vitals + Final_Risk_Score)
X_train_final = np.hstack([
    X_train_scaled,
    X_train_imputed['Final_Risk_Score'].values.reshape(-1, 1)
])

X_test_final = np.hstack([
    X_test_scaled,
    X_test_imputed['Final_Risk_Score'].values.reshape(-1, 1)
])

# Build and train the Keras model
keras_model = models.Sequential([
    layers.Input(shape=(7,)),
    layers.Dense(16, activation='relu'),
    layers.Dense(8, activation='relu'),
    layers.Dense(1, activation='sigmoid')
])

keras_model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy', tf.keras.metrics.AUC(name='auc')]
)

print("Fitting Keras model...")
keras_model.fit(X_train_final, y_train_imputed, epochs=50, batch_size=32, validation_split=0.2, verbose=0)

# Evaluate
y_pred_prob = keras_model.predict(X_test_final, verbose=0)
y_pred = (y_pred_prob >= 0.5).astype(int).flatten()

from sklearn.metrics import accuracy_score, roc_auc_score
print(f"\nKeras Model Performance:")
print(f"  Accuracy: {accuracy_score(y_test_imputed, y_pred):.4f}")
print(f"  AUC: {roc_auc_score(y_test_imputed, y_pred):.4f}")

# ===== EXPORT MODELS =====
print("\nExporting models...")

# Save Keras model in .keras format (native Keras 3)
keras_model.save('app/keras_model.keras')
print("  ✓ Keras model (.keras) -> app/keras_model.keras")

# Also save as H5 for compatibility
keras_model.save('app/keras_model.h5')
print("  ✓ Keras model (H5) -> app/keras_model.h5")

# Save StandardScaler
joblib.dump(scaler, 'app/scaler.pkl')
print("  ✓ StandardScaler -> app/scaler.pkl")

# Save as TFLite for Android
converter = tf.lite.TFLiteConverter.from_keras_model(keras_model)
tflite_model = converter.convert()
with open('AndroidApp/sanraksha/assets/finalmodel.tflite', 'wb') as f:
    f.write(tflite_model)
print("  ✓ TFLite model -> AndroidApp/sanraksha/assets/finalmodel.tflite")

# Save stage models for reference
joblib.dump(model1, 'app/stage1_model.pkl')
joblib.dump(model2, 'app/stage2_model.pkl')
joblib.dump(model3, 'app/stage3_model.pkl')
print("  ✓ Stage models (for reference)")

print("\n✅ All models exported successfully!")
print("\nNow update app/main.py to use the Keras model instead of XGBoost.")
