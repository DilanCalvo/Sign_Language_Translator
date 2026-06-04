#!/usr/bin/env python3
"""Quick test of grammar module functionality."""

from src.grammar import conjugate_verb, is_pronoun

print("Testing grammar module...")
print()

# Test conjugation
tests = [
    ("creer", "yo", "creo"),
    ("creer", "ellos", "creen"),
    ("ser", "yo", "soy"),
    ("ser", "nosotros", "somos"),
    ("poder", "él", "puede"),
    ("tener", "tú", "tienes"),
]

print("Conjugation tests:")
for verb, pronoun, expected in tests:
    result = conjugate_verb(verb, pronoun)
    status = "PASS" if result == expected else "FAIL"
    print(f"  [{status}] conjugate_verb('{verb}', '{pronoun}') = '{result}' (expected '{expected}')")

print()
print("Pronoun detection tests:")
pronouns = ["yo", "ellos", "nosotros", "unknown"]
for p in pronouns:
    result = is_pronoun(p)
    print(f"  is_pronoun('{p}') = {result}")

print()
print("All tests completed!")
