"""
Grammar module: verb conjugation based on Spanish context.

This module provides a simple, deterministic conjugation system: given a verb
(base form) and a pronoun, it returns the correct conjugated form. It is used
by the overlay to automatically conjugate verbs in accumulated text based on
the previous word (pronoun).

No ML — pure lookup with a fallback: if the verb is not in the table, it is
returned unconjugated (safe for unknown words).
"""

from config import VERB_CONJUGATIONS, PRONOUN_TO_FORM, TIME_MARKERS, DEFAULT_TENSE


def conjugate_verb(verb: str, pronoun: str | None, tense: str = "present") -> str:
    """
    Conjugate a verb given a pronoun context and tense.

    Args:
        verb: base form of the verb (e.g., "creer", "ser", "ir").
        pronoun: word that precedes the verb (e.g., "yo", "ellos").
        tense: grammatical tense ("present", "past", "future"). Default: "present".

    Returns:
        Conjugated form if pronoun is recognized, verb is in the table, and tense exists.
        Otherwise, returns the verb unchanged (safe fallback).

    Example:
        conjugate_verb("creer", "yo", "present")  → "creo"
        conjugate_verb("creer", "yo", "past")     → "creí"
        conjugate_verb("creer", "yo", "future")   → "creeré"
        conjugate_verb("creer", "ellos")          → "creen" (default present)
        conjugate_verb("creer", "unknown", "past") → "creer"  (fallback)
    """
    if not pronoun:
        return verb

    # Normalize pronoun and tense to lowercase for case-insensitive matching.
    pronoun_norm = pronoun.lower().strip()
    tense_norm = tense.lower().strip()

    # Lookup the grammatical form (e.g., "yo" → "1p_sg").
    form = PRONOUN_TO_FORM.get(pronoun_norm)
    if not form:
        return verb  # Pronoun not recognized; return verb unchanged.

    # Lookup the conjugation table for this verb.
    if verb not in VERB_CONJUGATIONS:
        return verb  # Verb not in table; return unchanged.

    # Lookup the conjugation for this tense.
    tense_conjugations = VERB_CONJUGATIONS[verb].get(tense_norm)
    if not tense_conjugations:
        return verb  # Tense not available for this verb; return unchanged.

    # Return the conjugated form, or the verb if the form is missing.
    return tense_conjugations.get(form, verb)


def is_pronoun(word: str) -> bool:
    """
    Check if a word is a recognized pronoun.

    Used by the overlay to decide whether to attempt conjugation on the
    next word. Case-insensitive.
    """
    return word.lower().strip() in PRONOUN_TO_FORM


def detect_tense(text: str) -> str:
    """
    Detect tense from text based on time markers.

    Scans the text for known time markers (ayer, mañana, ahora, etc.)
    and returns the corresponding tense. If no marker is found,
    returns the default tense (present).

    Args:
        text: accumulated text (e.g., "ayer yo creer")

    Returns:
        Tense string ("past", "present", "future")

    Example:
        detect_tense("ayer yo creer") → "past"
        detect_tense("mañana vamos") → "future"
        detect_tense("yo creo") → "present"  (default)
    """
    words = text.lower().split()

    # Scan for time markers (first match wins)
    for word in words:
        if word in TIME_MARKERS:
            return TIME_MARKERS[word]

    # No marker found; return default tense
    return DEFAULT_TENSE


def export_grammar_table() -> dict:
    """
    Export the grammar tables (verbs and pronouns) as a single dictionary.

    Used to save a JSON file for web consumption, ensuring parity between
    Python and JavaScript conjugation logic.

    Returns:
        Dictionary with "verbs" and "pronouns" keys, directly serializable
        to JSON.
    """
    return {
        "verbs": VERB_CONJUGATIONS,
        "pronouns": PRONOUN_TO_FORM,
        "time_markers": TIME_MARKERS,
        "default_tense": DEFAULT_TENSE,
    }
