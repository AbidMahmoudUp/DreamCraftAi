import re

def normalize_arabizi(text: str) -> str:
    """Normalize Arabizi characters to Arabic for Tounsi/Jazairi dialects."""
    arabizi_map = {
        "7": "ح", "3": "ع", "9": "ق", "5": "خ", "2": "ء",
        "ch": "ش", "sh": "ش", "th": "ث", "dh": "ذ", "gh": "غ"
    }
    for char, arabic in arabizi_map.items():
        text = text.replace(char, arabic)
    return text

def clean_text(text: str) -> str:
    """Remove URLs, special characters, and normalize whitespace."""
    text = re.sub(r"http[s]?://\S+", "", text)  # Remove URLs
    text = re.sub(r"[^\w\s,.!?آ-ي]", "", text)  # Keep Arabic, Latin, and basic punctuation
    text = re.sub(r"\s+", " ", text).strip()    # Normalize whitespace
    return text

def preprocess_text(text: str, is_dialect: bool = False) -> str:
    """Preprocess text, optionally normalizing Arabizi for dialects."""
    text = clean_text(text)
    if is_dialect:
        text = normalize_arabizi(text)
    return text