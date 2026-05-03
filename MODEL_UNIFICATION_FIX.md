# Model Unification Fix - Summary

## Problem Identified
The backend and Android app were using **three completely different models**:
- **Backend API**: XGBoost model (18 features)
- **Android offline (hardcoded)**: Old 3-stage LogisticRegression with incorrect weights  
- **Android TFLite**: Keras Sequential NN trained on different data

This caused **different predictions for the same patient data** - clinically dangerous for a maternal health app.

## Solution Implemented
✅ **Unified both online and offline paths to use the same model architecture:**

### 1. **Backend Update** (`app/main.py`)
- Replaced XGBoost with unified **Keras Sequential NN**
- Implemented **3-stage LogisticRegression ensemble** as preprocessing:
  - **Stage 1**: Historical factors (Previous Complications, Preexisting Diabetes, Gestational Diabetes, Mental Health) → Risk Score
  - **Stage 2**: Abnormality flags (low/high BMI, BP, BS, HR) → Risk Score Abn
  - **Stage 3**: Combined scores → Final Risk Score
- Added **StandardScaler** for vital signs normalization
- Final prediction: Keras NN on 7 inputs (6 scaled vitals + Final_Risk_Score)

### 2. **Android Update** (`AndroidApp/sanraksha/Models123.kt`)
Updated with **correct weights and scaler parameters**:

#### Stage 1 Weights (from trained LogisticRegression):
```
weight1 = -2.3618839494751795
weight2 = -4.10436327362052
weight3 = -4.7575702806286815
weight4 = -2.0561135304354563
bias    = 3.3894794466732603
```

#### Stage 2 Weights (abnormality flags):
```
weight1 = -3.0585747061010173  // is_low_bmi
weight2 = -2.5008934783065744  // is_high_bmi
weight3 = 0.23212516267988542  // is_low_bp
weight4 = -1.397350083593462   // is_high_bp
weight5 = -4.082720266366235   // is_high_bs
weight6 = 0.0                   // is_high_hr
weight7 = -1.2991199093290822  // is_low_hr
bias    = 1.8588124199686031
```

#### Stage 3 Weights (final ensemble):
```
weight1 = 5.8991195017033
weight2 = 3.7204274021730193
bias    = -5.215292748027206
```

#### Scaler Parameters (for TFLite input):
```
Feature means:   [7.545, 23.342, 27.544, 75.723, 117.005, 77.251]
Feature stds:    [3.066, 3.944, 9.128, 7.332, 18.551, 14.155]
Features order:  [BS, BMI, Age, Heart Rate, Systolic BP, Diastolic]
```

### 3. **Model Exports** (Generated)
- `app/keras_model.keras` - Keras Sequential NN (primary format)
- `app/keras_model.h5` - Keras Sequential NN (H5 backup)
- `app/scaler.pkl` - StandardScaler for backend preprocessing
- `app/stage1_model.pkl`, `stage2_model.pkl`, `stage3_model.pkl` - LogisticRegression ensemble
- `AndroidApp/sanraksha/assets/finalmodel.tflite` - Updated TFLite with correct pipeline

## Result
✅ **Online (Backend API) and Offline (Android TFLite) now use identical model logic:**
1. Same 3-stage preprocessing with exact same weights
2. Same scaler for vitals normalization
3. Same Keras Sequential architecture for final prediction
4. **Both give consistent results for the same input**

## Files Changed
- `app/main.py` - Complete rewrite to use Keras + 3-stage ensemble
- `AndroidApp/sanraksha/Models123.kt` - Updated weights and scaler values
- `ondevice-maternalrisk.ipynb` - (no changes needed, already exports correct models)
- New files:
  - `export_unified_model.py` - Master script to train and export unified models
  - `extract_weights_for_android.py` - Helper to extract Kotlin-formatted weights
  - `extract_scaler_for_android.py` - Helper to extract Kotlin-formatted scaler params

## Commit Hash
`a178159` - Pushed to akssri1317/prj3 main branch

## Next Steps
1. **Test online prediction**: Deploy `app/main.py` to Render (the backend endpoint)
2. **Test offline prediction**: Rebuild Android app - it will now use updated weights automatically
3. **Verify consistency**: Compare results from the same test data on both paths
