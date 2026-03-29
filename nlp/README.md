# SanRaksha NLP Engine (Offline)

This module converts ASHA worker voice notes into structured maternal-health vitals.

Pipeline stages:

1. Audio transcription with Whisper
2. Rule-based parsing of vitals
3. Confidence and clinical-range validation
4. Output export to transcript and JSON

## Prerequisites

1. Python 3.10+
2. ffmpeg available in PATH

## Setup

```bash
pip install -r requirements.txt
```

## Run (Audio -> Structured Data)

```bash
python run_pipeline.py --audio path/to/file.wav --model tiny --accept-low-confidence
```

Optional flags:

- `--language hi` force Hindi
- `--output-dir output_demo` choose output folder
- `--model tiny|base|small|medium|large|turbo`

## Run (Transcript-only mode)

Useful when you want to test parser logic without ASR.

```bash
python run_pipeline.py --transcript "blood pressure 130 over 85, sugar 120, bmi 24" --output-dir output_demo
```

## Outputs

Default output directory: `output/`

- `transcript.txt`: final transcript used for parsing
- `parsed.json`: structured fields, confidence map, review flags

## Demonstration Example

```bash
python run_pipeline.py --audio demo_audio.wav --model tiny --accept-low-confidence --output-dir output_demo
```

Expected behavior:

1. Shows detected language and transcription confidence
2. Prints extracted vitals with status (`ok`, `uncertain`, `missing`)
3. Saves `output_demo/transcript.txt` and `output_demo/parsed.json`
