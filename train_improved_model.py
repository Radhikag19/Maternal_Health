"""
Improved Keras Model for Maternal Risk Prediction
Deeper architecture with regularization for better accuracy
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix, classification_report
import tensorflow as tf
from tensorflow.keras import layers, models, callbacks
import joblib

print("=" * 70)
print("IMPROVED MATERNAL RISK PREDICTION MODEL")
print("=" * 70)

# 1. Load and prepare data
print("\n[1/5] Loading and preparing data...")
df = pd.read_csv('data/Dataset - Updated.csv')

# Drop rows with missing critical values
df_clean = df.dropna(subset=['Heart Rate', 'BS', 'Previous Complications', 'Preexisting Diabetes', 'Risk Level'])

# Encode labels: Low -> 1, High -> 0 (as per original training)
df_clean['Risk Level'] = pd.factorize(df_clean['Risk Level'])[0]
df_clean = df_clean.copy()
df_clean['Risk Level'] = df_clean['Risk Level'].apply(lambda col: 1 if col == 0 else 0)

# Remove Body Temp (low information gain based on MI analysis)
X = df_clean.drop(['Risk Level', 'Body Temp'], axis=1)
y = df_clean['Risk Level']

print(f"Dataset size: {len(X)}")
print(f"Features: {list(X.columns)}")
print(f"Class distribution: {pd.Series(y).value_counts().to_dict()}")

# 2. Train-test split
print("\n[2/5] Splitting data (80/20)...")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Fill missing values with median
X_train_filled = X_train.fillna(X_train.median())
X_test_filled = X_test.fillna(X_train.median())  # Use training median for test

# 3. Standardize features
print("\n[3/5] Standardizing features...")
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_filled)
X_test_scaled = scaler.transform(X_test_filled)

print(f"Training set shape: {X_train_scaled.shape}")
print(f"Test set shape: {X_test_scaled.shape}")

# 4. Build IMPROVED model (deeper with regularization)
print("\n[4/5] Building improved neural network model...")
print("Architecture:")
print("  Input(11) -> Dense(64, relu, L2) -> Dropout(0.3)")
print("           -> Dense(32, relu, L2) -> Dropout(0.3)")
print("           -> Dense(16, relu, L2) -> Dropout(0.2)")
print("           -> Dense(8, relu) -> Dense(1, sigmoid)")

model = models.Sequential([
    layers.Dense(64, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.001), input_shape=(X_train_scaled.shape[1],)),
    layers.Dropout(0.3),
    
    layers.Dense(32, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.001)),
    layers.Dropout(0.3),
    
    layers.Dense(16, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.001)),
    layers.Dropout(0.2),
    
    layers.Dense(8, activation='relu'),
    layers.Dense(1, activation='sigmoid')
])

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    loss='binary_crossentropy',
    metrics=['accuracy', tf.keras.metrics.AUC(name='auc')]
)

print(f"\nModel summary:")
model.summary()

# 5. Train model with early stopping
print("\n[5/5] Training model (with early stopping)...")

early_stopping = callbacks.EarlyStopping(
    monitor='val_loss',
    patience=10,
    restore_best_weights=True,
    verbose=1
)

history = model.fit(
    X_train_scaled, y_train,
    epochs=100,
    batch_size=16,
    validation_split=0.2,
    callbacks=[early_stopping],
    verbose=1
)

# 6. Evaluate on test set
print("\n" + "=" * 70)
print("MODEL EVALUATION")
print("=" * 70)

y_pred_proba = model.predict(X_test_scaled, verbose=0)
y_pred = (y_pred_proba > 0.5).astype(int).flatten()

accuracy = accuracy_score(y_test, y_pred)
auc = roc_auc_score(y_test, y_pred_proba)

print(f"\nTest Set Results:")
print(f"  Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
print(f"  AUC: {auc:.4f}")

print(f"\nConfusion Matrix:")
cm = confusion_matrix(y_test, y_pred)
print(cm)

print(f"\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=['High Risk', 'Low Risk']))

# 7. Compare with original model
print("\n" + "=" * 70)
print("COMPARISON: ORIGINAL vs IMPROVED")
print("=" * 70)
print(f"Original Model: ~59.07% accuracy")
print(f"Improved Model: {accuracy*100:.2f}% accuracy")
print(f"Improvement: {(accuracy - 0.5907)*100:+.2f}%")

# 8. Save all artifacts
print("\n[SAVING] Exporting models and artifacts...")

# Save Keras model
model.save('app/keras_model_improved.keras')
print("✓ Saved: app/keras_model_improved.keras")

# Save H5 backup
model.save('app/keras_model_improved.h5')
print("✓ Saved: app/keras_model_improved.h5")

# Save scaler (IMPORTANT: This includes feature names order)
joblib.dump(scaler, 'app/scaler_improved.pkl')
print("✓ Saved: app/scaler_improved.pkl")

# 9. Convert to TFLite for on-device inference
print("\n[CONVERTING] To TensorFlow Lite format...")
converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
tflite_model = converter.convert()

with open('AndroidApp/sanraksha/assets/finalmodel_improved.tflite', 'wb') as f:
    f.write(tflite_model)
    
print("✓ Saved: AndroidApp/sanraksha/assets/finalmodel_improved.tflite")

# 10. Summary
print("\n" + "=" * 70)
print("SUMMARY")
print("=" * 70)
print(f"✓ Model trained successfully")
print(f"✓ Test Accuracy: {accuracy*100:.2f}%")
print(f"✓ Improvement over original: {(accuracy - 0.5907)*100:+.2f}%")
print(f"✓ All artifacts saved (keras, h5, pkl, tflite)")
print(f"\nNext steps:")
print(f"  1. Review accuracy improvement")
print(f"  2. Compare predictions with original model")
print(f"  3. If satisfied, replace original files")
print(f"  4. Update backend (app/main.py) to use new scaler/model")
print(f"  5. Rebuild Android app to use new TFLite")
print("=" * 70)
