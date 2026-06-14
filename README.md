![SanRaksha Banner](assets/Banner.png)

# SanRaksha: Maternal Health Risk Assessment Ecosystem

SanRaksha is an AI-assisted, offline-first system that helps field health workers capture maternal vitals and assess pregnancy risk. It includes a backend API, an Android app for offline use, an NLP pipeline for voice-based data entry, and a Streamlit dashboard for analytics.

## Key Capabilities

- Offline risk scoring on low-cost Android devices
- Unified ML pipeline for consistent online/offline predictions
- Voice-to-text data capture with parsing of vitals
- Dashboard analytics with maps, charts, and reports

## Project Structure

- `app/` - FastAPI backend and ML model artifacts
- `AndroidApp/` - Android application source
- `dashboard/` - Streamlit dashboard
- `nlp/` - Voice transcription and parsing pipeline
- `assets/` - Project images used by documentation and UI
- `data/` - Sample datasets and captured vitals

## Quick Start (Local)

### Backend API

```bash
pip install -r requirements.txt
cd app
python -m uvicorn main:app --reload --port 8000
```

### Dashboard

```bash
pip install -r dashboard\requirements.txt
streamlit run dashboard\app.py
```

### NLP Pipeline (optional)

```bash
pip install -r nlp\requirements.txt
python nlp\run_pipeline.py
```