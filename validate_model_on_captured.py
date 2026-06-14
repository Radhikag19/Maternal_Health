"""
Load saved model and scaler, run predictions on data/captured_vitals.csv,
and report totals, probabilities, and mismatches with stored predictions.
"""
import pandas as pd
import numpy as np
import joblib
import tensorflow as tf
from pathlib import Path

BASE = Path(__file__).resolve().parent
MODEL_PATH = BASE / 'app' / 'keras_model_improved.keras'
SCALER_PATH = BASE / 'app' / 'scaler_improved.pkl'
CSV_PATH = BASE / 'data' / 'captured_vitals.csv'

print('Loading model & scaler...')
model = tf.keras.models.load_model(str(MODEL_PATH))
scaler = joblib.load(str(SCALER_PATH))
print('Loaded model and scaler')

# Read captured vitals
if not CSV_PATH.exists():
    print('No captured_vitals.csv found at', CSV_PATH)
    raise SystemExit(1)

df = pd.read_csv(CSV_PATH)
print('Rows in captured_vitals:', len(df))

# Normalize column names used in training
rename_map = {
    'Systolic BP': 'Systolic_BP',
    'Body Temp': 'Body_Temp',
    'Previous Complications': 'Previous_Complications',
    'Preexisting Diabetes': 'Preexisting_Diabetes',
    'Gestational Diabetes': 'Gestational_Diabetes',
    'Mental Health': 'Mental_Health',
}
df = df.rename(columns=rename_map)

features = [
    'Age',
    'Systolic_BP',
    'Diastolic',
    'BS',
    'BMI',
    'Previous_Complications',
    'Preexisting_Diabetes',
    'Gestational_Diabetes',
    'Mental_Health',
    'Heart_Rate'
]
# also tolerate 'Heart Rate' if present
if 'Heart_Rate' not in df.columns and 'Heart Rate' in df.columns:
    df['Heart_Rate'] = df['Heart Rate']

missing = [c for c in features if c not in df.columns]
if missing:
    print('ERROR - missing columns in captured CSV:', missing)
    # try to continue by adding NaNs
    for c in missing:
        df[c] = np.nan

X = df[features].copy()
# convert to numeric
for c in features:
    X[c] = pd.to_numeric(X[c], errors='coerce')

# fill NaNs with scaler mean
if hasattr(scaler, 'mean_'):
    means = scaler.mean_
    for i, c in enumerate(features):
        X[c] = X[c].fillna(means[i])
else:
    X = X.fillna(X.median())

X_arr = scaler.transform(X.values)
probs = model.predict(X_arr).ravel()
preds = (probs > 0.5).astype(int)
labels = np.where(preds==1, 'High Risk', 'Low Risk')

# Report
print('\nPrediction summary:')
print(' Pred Low Risk:', int((preds==0).sum()))
print(' Pred High Risk:', int((preds==1).sum()))

# Compare to existing stored predictions if present
if 'prediction' in df.columns:
    stored = pd.to_numeric(df['prediction'], errors='coerce').fillna(-1).astype(int)
    match = (stored == preds)
    matches = int(match.sum())
    total = len(df)
    print(f'Stored predictions present: {total} rows, matches: {matches}, mismatches: {total-matches}')
    if total-matches > 0:
        print('\nSamples with mismatches (first 10):')
        mism = df.loc[~match].copy()
        mism = mism.reset_index()
        for idx, row in mism.head(10).iterrows():
            i = int(row['index'])
            print(f" Row {i+1}: stored={row.get('prediction')}({row.get('prediction_label', '')}), new={preds[i]}({labels[i]}), prob={probs[i]:.4f}")
else:
    print('No stored prediction column found in CSV to compare')

# Save a quick report CSV with new predictions (not overwriting original)
out = df.copy()
out['pred_prob_new'] = probs
out['pred_new'] = preds
out['pred_label_new'] = labels
REPORT = BASE / 'data' / 'captured_vitals_predicted.csv'
out.to_csv(REPORT, index=False)
print('\nWrote predictions to', REPORT)

# Print a few rows for spot check
print('\nFirst 5 rows with new predictions:')
print(out[['Age','Systolic_BP','Diastolic','BS','BMI','Heart_Rate','pred_prob_new','pred_label_new']].head(5).to_string(index=False))
