"""
FIXED ML MODEL - Retraining with correct 10 features + proper validation
"""
import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import joblib
from pathlib import Path

print("=" * 80)
print("RETRAINING ML MODEL WITH CORRECT 10 FEATURES")
print("=" * 80)

# Load dataset
print("\n1. Loading dataset...")
df = pd.read_csv('data/maternal_health_risk.csv')
print(f"   Loaded {len(df)} samples")
print(f"   Columns: {list(df.columns)}")

# Prepare features - THE CORRECTED 10-FEATURE SET
# Order: Age, Systolic_BP, Diastolic, BS, BMI, Previous_Complications,
#        Preexisting_Diabetes, Gestational_Diabetes, Mental_Health, Heart_Rate
feature_columns = [
    'Age',
    'SystolicBP',  # or Systolic BP depending on CSV column name
    'DiastolicBP',  # or Diastolic
    'BS',
    'BMI',
    'Previous_Complications',
    'Preexisting_Diabetes',
    'Gestational_Diabetes',
    'Mental_Health',
    'Heart_Rate'
]

# Handle missing columns - use what's available
available_features = [col for col in feature_columns if col in df.columns]
if len(available_features) < len(feature_columns):
    print(f"   WARNING: Only {len(available_features)}/{len(feature_columns)} features found")
    print(f"   Available: {available_features}")

X = df[available_features].fillna(df[available_features].mean())
y = df['RiskLevel'].map({'high risk': 1, 'low risk': 0}) if 'RiskLevel' in df.columns else df.iloc[:, -1]

print(f"   Features: {len(feature_columns)} ({len(available_features)} available)")
print(f"   Target distribution: Low Risk={sum(y==0)}, High Risk={sum(y==1)}")

# Split data
print("\n2. Splitting data...")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
print(f"   Train: {len(X_train)} samples")
print(f"   Test: {len(X_test)} samples")

# Scale features
print("\n3. Scaling features...")
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
print(f"   Scaler fitted with {len(available_features)} features")
print(f"   Mean: {scaler.mean_}")
print(f"   Scale: {scaler.scale_}")

# Build neural network
print("\n4. Building model with 10 features input...")
model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(len(available_features),)),
    tf.keras.layers.Dense(32, activation='relu', name='hidden1'),
    tf.keras.layers.Dropout(0.3),
    tf.keras.layers.Dense(16, activation='relu', name='hidden2'),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(8, activation='relu', name='hidden3'),
    tf.keras.layers.Dense(1, activation='sigmoid', name='output')
])

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    loss='binary_crossentropy',
    metrics=['accuracy', tf.keras.metrics.AUC(name='auc')]
)

print(model.summary())

# Train model
print("\n5. Training model...")
history = model.fit(
    X_train_scaled, y_train,
    epochs=100,
    batch_size=16,
    validation_split=0.2,
    verbose=1,
    early_stopping=True if 'EarlyStopping' in dir(tf.keras.callbacks) else False
)

# Evaluate
print("\n6. Evaluating on test set...")
test_loss, test_acc, test_auc = model.evaluate(X_test_scaled, y_test, verbose=0)
print(f"   Test Accuracy: {test_acc:.4f}")
print(f"   Test AUC: {test_auc:.4f}")
print(f"   Test Loss: {test_loss:.4f}")

# Predictions on test set
y_pred_proba = model.predict(X_test_scaled, verbose=0)
y_pred = (y_pred_proba > 0.5).astype(int).flatten()

# Distribution of predictions
unique, counts = np.unique(y_pred, return_counts=True)
print(f"\n7. Prediction distribution on test set:")
for label, count in zip(unique, counts):
    print(f"   {'High Risk' if label == 1 else 'Low Risk'}: {count}")

# Save model and scaler
print("\n8. Saving model and scaler...")
model.save('app/keras_model_improved.keras')
joblib.dump(scaler, 'app/scaler_improved.pkl')
print("   ✓ Model saved: app/keras_model_improved.keras")
print("   ✓ Scaler saved: app/scaler_improved.pkl")

# Test with sample data
print("\n9. Testing with sample inputs...")

test_samples = [
    ("Healthy case", [25, 120, 80, 100, 22, 0, 0, 0, 0, 72]),
    ("High BP", [30, 160, 95, 100, 24, 0, 0, 0, 0, 80]),
    ("With complications", [35, 140, 90, 140, 28, 1, 1, 1, 1, 95]),
]

for desc, sample in test_samples:
    sample_scaled = scaler.transform([sample])
    prob = model.predict(sample_scaled, verbose=0)[0, 0]
    pred = "High Risk" if prob > 0.5 else "Low Risk"
    print(f"   {desc}: {pred} (prob={prob:.4f})")

print("\n" + "=" * 80)
print("MODEL RETRAINING COMPLETE")
print("=" * 80)
