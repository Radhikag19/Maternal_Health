"""
Retrain Keras model on the 10 features, converting BS from mmol/L to mg/dL (x18)
Saves model to app/keras_model_improved.keras and scaler to app/scaler_improved.pkl
"""
import pandas as pd
import numpy as np
from pathlib import Path
import joblib
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
import os

print("Starting 10-feature retraining pipeline with BS -> mg/dL conversion...")
BASE = Path(__file__).resolve().parent
DATA_PATH = BASE / 'data' / 'Dataset - Updated.csv'
APP_DIR = BASE / 'app'
APP_DIR.mkdir(exist_ok=True)

# Load dataset
print('Loading dataset:', DATA_PATH)
df = pd.read_csv(DATA_PATH)
print('Rows:', len(df))
print('Columns:', df.columns.tolist())

# Rename columns to canonical names expected by backend
rename_map = {
    'Systolic BP': 'Systolic_BP',
    'Body Temp': 'Body_Temp',
    'Previous Complications': 'Previous_Complications',
    'Preexisting Diabetes': 'Preexisting_Diabetes',
    'Gestational Diabetes': 'Gestational_Diabetes',
    'Mental Health': 'Mental_Health',
    'Risk Level': 'Risk_Level'
}
df = df.rename(columns=rename_map)

# Convert BS from mmol/L to mg/dL by multiplying by 18 (app uses ~100 range)
if 'BS' in df.columns:
    median_bs = pd.to_numeric(df['BS'], errors='coerce').median()
    print('Median BS before conversion:', median_bs)
    # If median appears to be in mmol/L (< 50), convert
    if median_bs is not None and median_bs < 50:
        df['BS'] = pd.to_numeric(df['BS'], errors='coerce') * 18.0
        print('Converted BS to mg/dL (x18)')

# Required feature order
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
    'Heart Rate'
]
# Harmonize column name for Heart Rate
if 'Heart Rate' not in df.columns and 'HeartRate' in df.columns:
    df['Heart Rate'] = df['HeartRate']

# Check presence
missing = [c for c in features if c not in df.columns]
if missing:
    print('ERROR - missing required feature columns:', missing)
    raise SystemExit(1)

# Clean and impute
X = df[features].copy()
# Convert numeric
for col in features:
    X[col] = pd.to_numeric(X[col], errors='coerce')
# Impute with median
X = X.fillna(X.median())

# Prepare labels
if 'Risk_Level' in df.columns:
    y_raw = df['Risk_Level'].astype(str).str.strip().str.lower()
    # Map 'high' or 'high risk' to 1, 'low' or 'low risk' to 0
    y = y_raw.apply(lambda v: 1 if 'high' in v else 0)
else:
    print('ERROR - no Risk_Level column')
    raise SystemExit(1)

print('\nLabel distribution:')
print(y.value_counts())

# Train/Test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, random_state=42, stratify=y)
print('Train/Test sizes:', len(X_train), len(X_test))

# Scale
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Class weight if imbalance
from sklearn.utils import class_weight
classes = np.unique(y_train)
class_weights = class_weight.compute_class_weight(class_weight='balanced', classes=classes, y=y_train)
class_weight_dict = {int(c): w for c, w in zip(classes, class_weights)}
print('Class weights:', class_weight_dict)

# Build model
tf.random.set_seed(42)
model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(len(features),)),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dropout(0.3),
    tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(1, activation='sigmoid')
])
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy', tf.keras.metrics.AUC(name='auc')])
model.summary()

# Callbacks
callbacks = [tf.keras.callbacks.EarlyStopping(monitor='val_auc', patience=8, mode='max', restore_best_weights=True)]

# Fit
history = model.fit(X_train_scaled, y_train, epochs=200, batch_size=16, validation_split=0.2, callbacks=callbacks, class_weight=class_weight_dict, verbose=2)

# Evaluate
y_pred_proba = model.predict(X_test_scaled).ravel()
y_pred = (y_pred_proba > 0.5).astype(int)
print('\nTest metrics:')
print('Accuracy:', accuracy_score(y_test, y_pred))
print('Precision:', precision_score(y_test, y_pred, zero_division=0))
print('Recall:', recall_score(y_test, y_pred, zero_division=0))
print('F1:', f1_score(y_test, y_pred, zero_division=0))
print('AUC:', roc_auc_score(y_test, y_pred_proba))
print('Confusion matrix:\n', confusion_matrix(y_test, y_pred))

# Backup old artifacts
model_path = APP_DIR / 'keras_model_improved.keras'
scaler_path = APP_DIR / 'scaler_improved.pkl'
if model_path.exists():
    bak = APP_DIR / 'keras_model_improved.keras.bak'
    if bak.exists():
        bak.unlink()
    model_path.rename(bak)
if scaler_path.exists():
    bak2 = APP_DIR / 'scaler_improved.pkl.bak'
    if bak2.exists():
        bak2.unlink()
    scaler_path.rename(bak2)

# Save new
model.save(model_path)
joblib.dump(scaler, scaler_path)
print('\nSaved model and scaler to app/')

# Sanity predictions for a few cases (app units: BS ~100, HR ~75)
samples = [
    ('Healthy', [25,120,80,100,22,0,0,0,0,72]),
    ('High BP', [30,160,95,100,24,0,0,0,0,80]),
    ('Multi-risk', [38,160,100,140,32,1,1,1,1,100])
]
print('\nSanity checks:')
for desc, s in samples:
    s_arr = scaler.transform([s])
    p = model.predict(s_arr)[0,0]
    print(f" {desc}: prob={p:.4f} -> {'High' if p>0.5 else 'Low'}")

print('\nRetraining (with BS conversion) complete.')
