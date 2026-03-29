
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from word2number import w2n


# ── Clinical validation ranges (NHM / FOGSI guidelines) ─────────────────────
# Values outside these ranges are "uncertain" — not rejected, but flagged.
# Ranges are for adult pregnant women.
CLINICAL_RANGES: Dict[str, tuple] = {
    "systolic_bp":   (70,  180),   # mmHg
    "diastolic_bp":  (40,  120),   # mmHg
    "blood_sugar":   (40,  400),   # mg/dL (random)
    "body_temp":     (95.0, 104.0),# °F
    "bmi":           (13.0, 55.0), # kg/m²
    "heart_rate":    (40,  180),   # bpm
    "hemoglobin":    (4.0, 18.0),  # g/dL
    "spo2":          (70,  100),   # %
}

# Confidence levels for each field
CONF_OK        = "ok"
CONF_UNCERTAIN = "uncertain"   # parsed but outside clinical range
CONF_MISSING   = "missing"     # could not extract


# ── Hindi / regional number maps ──────────────────
unit_map = {
    "shoonya": 0, "ek": 1, "do": 2, "teen": 3, "chaar": 4, "paanch": 5,
    "cheh": 6, "saat": 7, "aath": 8, "nau": 9, "das": 10,
}
teen_map = {
    "gyaarah": 11, "baarah": 12, "terah": 13, "chaudah": 14, "pandrah": 15,
    "solah": 16, "satrah": 17, "atharah": 18, "unnis": 19,
}
tens_map = {
    "bees": 20, "tees": 30, "chaalis": 40, "pachaas": 50, "saath": 60,
    "sattar": 70, "assi": 80, "nabbe": 90,
}
hundred_map = {"sau": 100}

DEVANAGARI_DIGITS = {
    "०": "0", "१": "1", "२": "2", "३": "3", "४": "4",
    "५": "5", "६": "6", "७": "7", "८": "8", "९": "9",
}


def normalize_text(text: str) -> str:
    for devanagari_digit, ascii_digit in DEVANAGARI_DIGITS.items():
        text = text.replace(devanagari_digit, ascii_digit)
    return text


def hindi_phrase_to_number(phrase: str) -> Optional[int]:
    words = phrase.lower().strip().split()
    total, current = 0, 0
    for word in words:
        if word in teen_map:
            current += teen_map[word]
        elif word in unit_map:
            current += unit_map[word]
        elif word in tens_map:
            current += tens_map[word]
        elif word in hundred_map:
            if current == 0:
                current = 1
            current *= 100
            total += current
            current = 0
    total += current
    return total if total else None


def words_to_number(text: str) -> Optional[float]:
    try:
        return w2n.word_to_num(text)
    except Exception:
        return hindi_phrase_to_number(text)


# ── Result dataclass ─────────────────────────────────────────────────────────
@dataclass
class ParseResult:
    fields: Dict[str, Any]              # extracted values (None if missing)
    confidence: Dict[str, str]          # per-field: "ok" | "uncertain" | "missing"
    uncertain_fields: List[str] = field(default_factory=list)
    missing_fields: List[str] = field(default_factory=list)

    @property
    def needs_review(self) -> bool:
        """True when any field is uncertain or more than 2 fields are missing."""
        return bool(self.uncertain_fields) or len(self.missing_fields) > 2

    def flat(self) -> Dict[str, Any]:
        """Return just the values dict — drop-in replacement for old dict return."""
        return self.fields


# ── Main extraction function ─────────────────────────────────────────────────
def extract_health_data(text: str) -> ParseResult:
    """
    Extract structured health vitals from a transcribed string.

    Supports: BP (systolic/diastolic), blood sugar, body temperature, BMI,
    heart rate, hemoglobin, SpO2.

    Returns ParseResult.  Use .flat() for backwards-compatible dict access.
    """
    text = normalize_text(text.lower())

    # ── Synonym normalisation (original mappings preserved + new additions) ──
    replacements = {
        # BP direction helpers
        "upar wala": "systolic",  "upar ka": "systolic",
        "first reading": "systolic", "upar": "systolic",
        "neeche wala": "diastolic", "neeche ka": "diastolic",
        "second reading": "diastolic", "neeche": "diastolic",

        # Blood pressure
        "rakt chaap": "blood pressure", "bp": "blood pressure",
        "blood pressure": "blood pressure",
        "रक्तचाप": "blood pressure", "ब्लड प्रेशर": "blood pressure",
        "रक्त चाप": "blood pressure",

        # Heart rate
        "dil ki dhadkan": "heart rate", "dhadkan": "heart rate",
        "nabz": "heart rate", "pulse": "heart rate", "heartbeat": "heart rate",
        "दिल की धड़कन": "heart rate", "दिल की धड़कनें": "heart rate",
        "हार्ट रेट": "heart rate", "नब्ज": "heart rate",

        # Body temperature
        "bukhar": "body temperature", "tapmaan": "body temperature",
        "taapmaan": "body temperature", "tapman": "body temperature",
        "temperature": "body temperature",
        "तापमान": "body temperature", "बुखार": "body temperature",
        "टेम्परेचर": "body temperature", "शरीर का तापमान": "body temperature",

        # Blood sugar
        "cheeni": "blood sugar", "sugar": "blood sugar",
        "sugar level": "blood sugar", "blood sugar": "blood sugar",
        "sugar test": "blood sugar",
        "ब्लड शुगर": "blood sugar", "शुगर": "blood sugar",
        "चीनी": "blood sugar", "शुगर लेवल": "blood sugar",

        # BMI
        "weight": "bmi", "weight ratio": "bmi",
        "height weight": "bmi", "bmi": "bmi",
        "बीएमआई": "bmi",

        # ── NEW: Hemoglobin ──────────────────────────────────────────────
        "khoon ki kami": "hemoglobin", "hemoglobin": "hemoglobin",
        "heme globin": "hemoglobin", "hem globin": "hemoglobin",
        "haemoglobin": "hemoglobin",  "hb": "hemoglobin",
        "hgb": "hemoglobin", "blood count": "hemoglobin",
        "khoon": "hemoglobin",
        "हीमोग्लोबिन": "hemoglobin", "हीमोग्लोबिन स्तर": "hemoglobin",
        "खून की कमी": "hemoglobin",

        # ── NEW: SpO2 / Oxygen saturation ────────────────────────────────
        "oxygen level": "spo2", "oxygen saturation": "spo2",
        "spo2": "spo2", "o2 level": "spo2", "o2": "spo2",
        "pulse oximeter": "spo2", "pran vayu": "spo2",
        "oxygen": "spo2",
        "ऑक्सीजन लेवल": "spo2", "ऑक्सीजन सैचुरेशन": "spo2",
        "ऑक्सीजन": "spo2",
    }

    for hin, eng in replacements.items():
        text = text.replace(hin, eng)

    text = text.replace("over", "/").replace("by", "/")
    text = text.replace(" बाय ", " /")

    # ── Helper: validate against clinical range ──────────────────────────────
    def validate(field_name: str, value) -> str:
        if value is None:
            return CONF_MISSING
        lo, hi = CLINICAL_RANGES.get(field_name, (None, None))
        if lo is None:
            return CONF_OK
        try:
            v = float(value)
            return CONF_OK if lo <= v <= hi else CONF_UNCERTAIN
        except (TypeError, ValueError):
            return CONF_UNCERTAIN

    # ── Helper: generic regex extractor (unchanged logic) ───────────────────
    def extract_pattern(pattern: str):
        match = re.search(pattern, text)
        if match:
            value = match.group(1)
            if not re.match(r"^\d", value):
                value = words_to_number(value.strip())
            try:
                return float(value) if "." in str(value) else int(value)
            except (TypeError, ValueError):
                return None
        return None

    # ── Blood pressure (unchanged logic) ────────────────────────────────────
    systolic, diastolic = None, None
    bp_match = re.search(
        r"blood pressure[^\d]*(\d{2,3})\s*/\s*(\d{2,3})", text
    )
    if not bp_match:
        word_bp = re.search(
            r"blood pressure[^\d]*([a-z\u0900-\u097f\s\-]+)\s*/\s*([a-z\u0900-\u097f\s\-]+)", text
        )
        if word_bp:
            systolic  = words_to_number(word_bp.group(1).strip())
            diastolic = words_to_number(word_bp.group(2).strip())
    else:
        systolic, diastolic = bp_match.group(1), bp_match.group(2)

    if not systolic:
        m = re.search(r"(systolic|upper)[^\d]*(\d{2,3}|[a-z\u0900-\u097f\s\-]+)", text)
        if m:
            v = m.group(2)
            systolic = v if v[0].isdigit() else words_to_number(v)

    if not diastolic:
        m = re.search(r"(diastolic|lower)[^\d]*(\d{2,3}|[a-z\u0900-\u097f\s\-]+)", text)
        if m:
            v = m.group(2)
            diastolic = v if v[0].isdigit() else words_to_number(v)

    # Handle translated phrasing like:
    # "blood pressure is above 120 and below 80"
    if not systolic or not diastolic:
        m = re.search(
            r"blood pressure[^\d]*(?:is\s+)?(?:above|upper|higher)[^\d]*(\d{2,3})[^\d]+(?:and\s+)?(?:below|lower)[^\d]*(\d{2,3})",
            text,
        )
        if m:
            systolic = systolic or m.group(1)
            diastolic = diastolic or m.group(2)

    # Handle translated phrasing like:
    # "upper part is 120 and lower part is 80"
    if not systolic or not diastolic:
        m = re.search(
            r"(?:upper\s+part|upper\s+value)[^\d]*(\d{2,3})[^\d]+(?:lower\s+part|lower\s+value)[^\d]*(\d{2,3})",
            text,
        )
        if m:
            systolic = systolic or m.group(1)
            diastolic = diastolic or m.group(2)

    # ── Extract all fields ───────────────────────────────────────────────────
    extracted = {
        "systolic_bp":  int(systolic)  if systolic  else None,
        "diastolic_bp": int(diastolic) if diastolic else None,
        "blood_sugar":  extract_pattern(
            r"blood sugar[^\d]*(\d+(?:\.\d+)?|[a-z\u0900-\u097f\s\-]+)"
        ),
        "body_temp":    extract_pattern(
            r"(?:body\s+temperature|body\s+temp|temperature|temp)[^\d]*(\d+(?:\.\d+)?|[a-z\u0900-\u097f\s\-]+)"
        ),
        "bmi":          extract_pattern(
            r"bmi[^\d]*(\d+(?:\.\d+)?|[a-z\u0900-\u097f\s\-]+)"
        ),
        "heart_rate":   extract_pattern(
            r"(?:heart rate)[^\d]*(\d+(?:\.\d+)?|[a-z\u0900-\u097f\s\-]+)"
        ),
        # ── NEW fields ───────────────────────────────────────────────────
        "hemoglobin":   extract_pattern(
            r"hemoglobin[^\d]*(\d+(?:\.\d+)?|[a-z\u0900-\u097f\s\-]+)"
        ),
        "spo2":         extract_pattern(
            r"spo2[^\d]*(\d+(?:\.\d+)?|[a-z\u0900-\u097f\s\-]+)"
        ),
    }

    # ── Build confidence map ─────────────────────────────────────────────────
    confidence = {k: validate(k, v) for k, v in extracted.items()}
    uncertain  = [k for k, c in confidence.items() if c == CONF_UNCERTAIN]
    missing    = [k for k, c in confidence.items() if c == CONF_MISSING]

    return ParseResult(
        fields=extracted,
        confidence=confidence,
        uncertain_fields=uncertain,
        missing_fields=missing,
    )
