"""SanRaksha NLP module runner.

End-to-end flow:
1) Whisper transcription (or user-provided transcript)
2) Structured health data extraction
3) Confidence/review summary
4) Save outputs to text/json files
"""

import argparse
import json
import os
import sys
from dataclasses import asdict

from src.parser import CLINICAL_RANGES, CONF_UNCERTAIN, extract_health_data
from src.transcriber import CONFIDENCE_THRESHOLD, TranscriptionResult, transcribe_audio

RED = "\033[91m"
YELLOW = "\033[93m"
GREEN = "\033[92m"
CYAN = "\033[96m"
RESET = "\033[0m"
BOLD = "\033[1m"


def print_section(title: str) -> None:
    line = "-" * 50
    print(f"\n{BOLD}{CYAN}{line}{RESET}")
    print(f"{BOLD}{CYAN} {title}{RESET}")
    print(f"{BOLD}{CYAN}{line}{RESET}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run SanRaksha NLP pipeline (transcribe + parse + save)."
    )
    parser.add_argument(
        "--audio",
        type=str,
        default=None,
        help="Path to audio file (.wav/.mp3/.m4a).",
    )
    parser.add_argument(
        "--transcript",
        type=str,
        default=None,
        help="Skip transcription and parse this transcript directly.",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="medium",
        choices=["tiny", "base", "small", "medium", "large", "turbo"],
        help="Whisper model size (used only when --audio is provided).",
    )
    parser.add_argument(
        "--language",
        type=str,
        default=None,
        help="Optional ISO-639-1 language code (e.g. hi, en).",
    )
    parser.add_argument(
        "--task",
        type=str,
        default="translate",
        choices=["transcribe", "translate"],
        help="Whisper task: transcribe source language or translate to English. Default is translate.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="output",
        help="Directory where transcript and parsed.json will be saved.",
    )
    parser.add_argument(
        "--accept-low-confidence",
        action="store_true",
        help="Automatically continue when transcription confidence is low.",
    )
    return parser.parse_args()


def prompt_manual_correction(transcript: str) -> str:
    print(f"\n{YELLOW}{BOLD}! LOW CONFIDENCE TRANSCRIPT{RESET}")
    print(f"{YELLOW}The ASR model is uncertain about this transcription.{RESET}")
    print(f"\nRaw transcript:\n  {BOLD}{transcript}{RESET}\n")
    print("Options:")
    print("  [1] Use as-is and continue")
    print("  [2] Enter corrected transcript manually")
    choice = input("\nEnter 1 or 2: ").strip()
    if choice == "2":
        corrected = input("Enter corrected transcript: ").strip()
        return corrected if corrected else transcript
    return transcript


def run_transcription(args: argparse.Namespace) -> tuple[str, TranscriptionResult]:
    if args.transcript:
        print_section("Step 1 - Transcription (skipped)")
        print("[INFO] Using transcript provided via --transcript")
        stub = TranscriptionResult(
            text=args.transcript.strip(),
            confidence=1.0,
            language=args.language or "manual",
            language_probability=1.0,
            needs_review=False,
            segments=[],
            error=None,
        )
        return stub.text, stub

    if not args.audio:
        raise ValueError("Provide either --audio or --transcript.")

    print_section("Step 1 - Transcription")
    print(f"[INFO] Transcribing: {args.audio}")
    result = transcribe_audio(
        args.audio,
        model_size=args.model,
        language=args.language,
        task=args.task,
    )

    if result.error:
        raise RuntimeError(f"Transcription failed: {result.error}")

    print(
        f"[INFO] Language detected: {result.language} "
        f"(p={result.language_probability:.2f})"
    )
    print(
        f"[INFO] Confidence score : {result.confidence:.3f} "
        f"(threshold = {CONFIDENCE_THRESHOLD})"
    )

    transcript = result.text
    if result.needs_review:
        if args.accept_low_confidence:
            print("[WARN] Low confidence accepted via --accept-low-confidence")
        else:
            transcript = prompt_manual_correction(transcript)
    else:
        print(f"{GREEN}[OK] Confidence above threshold - proceeding automatically.{RESET}")

    print(f"\nTranscribed text:\n  {transcript}")
    return transcript, result


def run_parsing(transcript: str):
    print_section("Step 2 - Structured Extraction")
    parse = extract_health_data(transcript)

    print(f"\n{'Field':<20} {'Value':<15} {'Status'}")
    print("-" * 50)
    for key, value in parse.fields.items():
        conf = parse.confidence[key]
        status_str = (
            f"{GREEN}ok{RESET}"
            if conf == "ok"
            else f"{YELLOW}uncertain{RESET}"
            if conf == "uncertain"
            else f"{RED}missing{RESET}"
        )
        lo, hi = CLINICAL_RANGES.get(key, (None, None))
        range_str = f"  (valid: {lo}-{hi})" if lo and conf == CONF_UNCERTAIN else ""
        print(f"  {key:<20} {str(value):<15} {status_str}{range_str}")

    if parse.needs_review:
        print_section("Review Required")
        if parse.uncertain_fields:
            print(f"{YELLOW}Fields outside clinical range - verify with ASHA worker:{RESET}")
            for field in parse.uncertain_fields:
                lo, hi = CLINICAL_RANGES.get(field, ("?", "?"))
                print(f"  * {field}: {parse.fields[field]} (expected {lo}-{hi})")
        if len(parse.missing_fields) > 2:
            print(f"\n{RED}Many fields missing - consider re-recording:{RESET}")
            for field in parse.missing_fields:
                print(f"  * {field}")
    else:
        print(f"\n{GREEN}[OK] All fields extracted within clinical ranges.{RESET}")

    return parse


def save_outputs(
    output_dir: str,
    transcript: str,
    transcription_result: TranscriptionResult,
    parse_result,
) -> None:
    print_section("Step 3 - Saving Output")
    os.makedirs(output_dir, exist_ok=True)

    transcript_path = os.path.join(output_dir, "transcript.txt")
    parsed_path = os.path.join(output_dir, "parsed.json")

    with open(transcript_path, "w", encoding="utf-8") as file:
        file.write(transcript)

    output_data = {
        "transcript": transcript,
        "transcription": {
            "confidence": transcription_result.confidence,
            "language": transcription_result.language,
            "language_probability": transcription_result.language_probability,
            "needs_review": transcription_result.needs_review,
            "error": transcription_result.error,
        },
        "parsed": asdict(parse_result),
        "pipeline_needs_review": bool(
            transcription_result.needs_review or parse_result.needs_review
        ),
    }

    with open(parsed_path, "w", encoding="utf-8") as file:
        json.dump(output_data, file, indent=2, ensure_ascii=False)

    print(f"[INFO] Transcript saved  -> {transcript_path}")
    print(f"[INFO] Parsed data saved -> {parsed_path}")
    print("\n" + "-" * 50)
    print(
        "REVIEW REQUIRED"
        if output_data["pipeline_needs_review"]
        else "PIPELINE COMPLETE - READY FOR RISK MODEL"
    )
    print("-" * 50 + "\n")


def main() -> int:
    args = parse_args()
    try:
        transcript, transcription_result = run_transcription(args)
        parse_result = run_parsing(transcript)
        save_outputs(args.output_dir, transcript, transcription_result, parse_result)
        return 0
    except Exception as exc:
        print(f"{RED}[ERROR] {exc}{RESET}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
