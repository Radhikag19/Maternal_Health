# SanRaksha

SanRaksha is a maternal health risk assessment platform with three connected parts:

- a FastAPI backend that serves predictions from the improved Keras model
- a Streamlit PHC dashboard for monitoring and analytics
- an Android app for offline and online risk assessment at the field level
- an offline NLP pipeline that turns voice notes into structured vitals

The system has been unified around a single improved model with **98.73% accuracy** so the online backend and the Android offline flow produce consistent results.

## Project Overview

SanRaksha is designed to help health workers capture maternal vital signs, assess risk, and review cases in a central dashboard. It supports both connected and offline workflows so it can be used in low-connectivity environments.

Core capabilities:

- maternal risk prediction from 10 clinical features
- backend API for online predictions and capture logging
- Android client for field use and offline inference
- dashboard for risk analytics, state-wise summaries, and trend views
- audio-to-structured-data pipeline for voice-based vitals capture

## Repository Structure

```text
SanRaksha/
├── app/                     # FastAPI backend and trained model artifacts
├── dashboard/               # Streamlit PHC dashboard
├── AndroidApp/              # Android application project
├── nlp/                     # Offline voice transcription and parsing pipeline
├── data/                    # CSV datasets and captured submissions
├── assets/                  # Shared project assets
├── README.md                # Root project guide
└── requirements.txt         # Python dependencies for the main Python components
```

## Main Components

### 1. Backend API

The backend is implemented in [app/main.py](app/main.py). It loads:

- `app/keras_model_improved.keras`
- `app/scaler_improved.pkl`

It exposes a prediction endpoint that accepts a single patient payload or a batch of payloads. Each prediction is also appended to `data/captured_vitals.csv` for downstream reporting.

### 2. PHC Dashboard

The dashboard lives in [dashboard/app.py](dashboard/app.py). It is a Streamlit app that provides:

- real-time summary metrics
- state-wise analysis
- national risk mapping
- week-over-week trend charts
- feedback and acknowledgment flows

### 3. Android App

The Android project is under [AndroidApp/sanraksha](AndroidApp/sanraksha). It includes:

- the main launcher in [AndroidApp/sanraksha/MainActivity.kt](AndroidApp/sanraksha/MainActivity.kt)
- offline TensorFlow Lite model support
- online API integration through Retrofit
- patient and vitals entry screens
- voice transcript parsing support

### 4. NLP Pipeline

The offline NLP pipeline is under [nlp/run_pipeline.py](nlp/run_pipeline.py). It converts spoken or written vitals into structured output using transcription, parsing, and validation steps.

## Model Summary

SanRaksha currently uses a unified improved Keras model with the following characteristics:

- input features: 10
- preprocessing: StandardScaler
- backend model format: `.keras`
- mobile model format: `.tflite`
- reported accuracy: **98.73%**

The model is used consistently across the backend and the Android app so predictions match across online and offline workflows.

## Python Setup

### Requirements

The main Python dependencies are listed in [requirements.txt](requirements.txt).

### Install

```bash
pip install -r requirements.txt
```

If you want to work only on the dashboard or NLP pipeline, see the module-specific requirements files in:

- [dashboard/requirements.txt](dashboard/requirements.txt)
- [nlp/requirements.txt](nlp/requirements.txt)

## Run the Backend

Start the FastAPI service from the `app/` directory:

```bash
cd app
uvicorn main:app --reload --port 8000
```

Useful endpoints:

- `GET /` - simple health response
- `POST /predict` - maternal risk prediction

Example request:

```bash
curl -X POST "http://localhost:8000/predict" ^
  -H "Content-Type: application/json" ^
  -d "{\"Age\":30,\"Systolic_BP\":120,\"Diastolic\":80,\"BS\":6.5,\"BMI\":25,\"Previous_Complications\":0,\"Preexisting_Diabetes\":0,\"Gestational_Diabetes\":0,\"Mental_Health\":0,\"Heart_Rate\":75}"
```

Expected response shape:

```json
{
  "prediction": [0],
  "prediction_label": ["Low Risk"]
}
```

## Run the Dashboard

Start the Streamlit dashboard from the `dashboard/` directory:

```bash
cd dashboard
streamlit run app.py
```

The dashboard reads from the CSV files in `data/` and can also display captured submissions created by the backend.

## Run the NLP Pipeline

The NLP module can process audio or transcript input.

### Audio input

```bash
cd nlp
python run_pipeline.py --audio path/to/file.wav --model tiny --accept-low-confidence
```

### Transcript input

```bash
cd nlp
python run_pipeline.py --transcript "blood pressure 130 over 85, sugar 120, bmi 24" --output-dir output_demo
```

See [nlp/README.md](nlp/README.md) for the full pipeline options and outputs.

## Android App

Open [AndroidApp/](AndroidApp) in Android Studio to work with the mobile client.

Key notes:

- the app supports offline inference with the improved TFLite model
- the app can also use the backend API when a network connection is available
- permissions for internet and network state are requested in `MainActivity`

For Android-specific implementation details, see [ANDROID_IMPROVEMENTS.md](ANDROID_IMPROVEMENTS.md).

## Data Files

Important data and generated files live in `data/` and `app/`:

- `data/Maternal_Health_Risk_Assessment Original Dataset.csv`
- `data/Dataset - Updated.csv`
- `data/risk_cases.csv`
- `data/captured_vitals.csv`
- `data/captured_vitals_predicted.csv`
- `app/keras_model_improved.keras`
- `app/scaler_improved.pkl`
- `app/finalmodel_improved.tflite`

## Documentation

Other project docs worth reading:

- [PROJECT_COMPLETION_SUMMARY.md](PROJECT_COMPLETION_SUMMARY.md)
- [DEPLOYMENT_CHECKLIST.md](DEPLOYMENT_CHECKLIST.md)
- [ANDROID_IMPROVEMENTS.md](ANDROID_IMPROVEMENTS.md)
- [dashboard/README.md](dashboard/README.md)
- [nlp/README.md](nlp/README.md)

## Deployment Notes

The backend and Android app are already aligned around the improved model. Typical next steps after local validation are:

1. deploy the FastAPI backend to your hosting platform
2. rebuild the Android app with the latest assets
3. verify that online and offline predictions match on the same patient input
4. confirm the dashboard reads the latest captured vitals CSV

See [DEPLOYMENT_CHECKLIST.md](DEPLOYMENT_CHECKLIST.md) for the manual deployment flow.

## Validation

Recommended sanity checks after setup:

- backend `/predict` returns a label for a sample payload
- dashboard starts without import errors
- Android app loads the new TFLite asset
- NLP pipeline can process either audio or transcript input

## License

No license file is currently included in this repository.
