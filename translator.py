import time
from langdetect import detect, LangDetectException
from deep_translator import GoogleTranslator


def detect_language(text):
    try:
        return detect(text[:2000])
    except LangDetectException:
        return "en"


def has_non_english(scraped_data):
    """Return True if any successfully scraped page is not in English."""
    for doc in scraped_data:
        if not doc.get("error") and doc.get("combined"):
            lang = detect_language(doc["combined"])
            if lang not in ("en",):
                return True
    return False


def translate_terms(terms):
    """
    Translate a list of keyword phrases to English.
    Returns a dict {original_term: english_translation}.
    If translation matches original or fails, value is empty string.
    """
    if not terms:
        return {}

    translator = GoogleTranslator(source="auto", target="en")
    results = {}

    for term in terms:
        try:
            translated = translator.translate(term)
            results[term] = translated if translated and translated.lower() != term.lower() else ""
            time.sleep(0.05)  # avoid rate limiting
        except Exception:
            results[term] = ""

    return results
