"""
Grammar module: verb conjugation based on Spanish context.

This module provides a simple, deterministic conjugation system: given a verb
(base form) and a pronoun, it returns the correct conjugated form. It is used
by the overlay to automatically conjugate verbs in accumulated text based on
the previous word (pronoun).

No ML — pure lookup with a fallback: if the verb is not in the table, it is
returned unconjugated (safe for unknown words).
"""

from config import VERB_CONJUGATIONS, PRONOUN_TO_FORM


def conjugate_verb(verb: str, pronoun: str | None) -> str:
    """
    Conjugate a verb given a pronoun context.

    Args:
        verb: base form of the verb (e.g., "creer", "ser", "ir").
        pronoun: word that precedes the verb (e.g., "yo", "ellos").

    Returns:
        Conjugated form if pronoun is recognized and verb is in the table.
        Otherwise, returns the verb unchanged (safe fallback).

    Example:
        conjugate_verb("creer", "yo")     → "creo"
        conjugate_verb("creer", "ellos")  → "creen"
        conjugate_verb("creer", "unknown") → "creer"  (fallback)
        conjugate_verb("unknown_verb", "yo") → "unknown_verb"  (fallback)
    """
    if not pronoun:
        return verb

    # Normalize pronoun to lowercase for case-insensitive matching.
    pronoun_norm = pronoun.lower().strip()

    # Lookup the grammatical form (e.g., "yo" → "1p_sg").
    form = PRONOUN_TO_FORM.get(pronoun_norm)
    if not form:
        return verb  # Pronoun not recognized; return verb unchanged.

    # Lookup the conjugation table for this verb.
    if verb not in VERB_CONJUGATIONS:
        return verb  # Verb not in table; return unchanged.

    # Return the conjugated form, or the verb if the form is missing.
    return VERB_CONJUGATIONS[verb].get(form, verb)


def is_pronoun(word: str) -> bool:
    """
    Check if a word is a recognized pronoun.

    Used by the overlay to decide whether to attempt conjugation on the
    next word. Case-insensitive.
    """
    return word.lower().strip() in PRONOUN_TO_FORM


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
    }
