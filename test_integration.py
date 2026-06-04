#!/usr/bin/env python3
"""
Integration test: simulate real letter-by-letter input with tense support.

This test mimics the actual flow:
1. User spells a pronoun letter by letter
2. User presses space
3. User spells a verb letter by letter
4. System conjugates automatically with correct tense (present, past, or future)
"""

from src.overlay import LetterBuffer

def test_conjugation_with_tenses():
    """Test conjugation across all tenses."""
    buffer = LetterBuffer()

    test_cases = [
        # Present tense (default)
        {
            "name": "PRESENT: yo creo",
            "sequence": "Y O space C R E E R",
            "tense": "present",
            "expected": "YO CREO",
        },
        {
            "name": "PRESENT: ellos creen",
            "sequence": "E L L O S space C R E E R",
            "tense": "present",
            "expected": "ELLOS CREEN",
        },
        # Past tense
        {
            "name": "PAST: yo creí",
            "sequence": "Y O space C R E E R",
            "tense": "past",
            "expected": "YO CREÍ",
        },
        {
            "name": "PAST: ellos creyeron",
            "sequence": "E L L O S space C R E E R",
            "tense": "past",
            "expected": "ELLOS CREYERON",
        },
        # Future tense
        {
            "name": "FUTURE: yo creeré",
            "sequence": "Y O space C R E E R",
            "tense": "future",
            "expected": "YO CREERÉ",
        },
        {
            "name": "FUTURE: ellos creerán",
            "sequence": "E L L O S space C R E E R",
            "tense": "future",
            "expected": "ELLOS CREERÁN",
        },
        # Irregular verbs
        {
            "name": "SER present: yo soy",
            "sequence": "Y O space S E R",
            "tense": "present",
            "expected": "YO SOY",
        },
        {
            "name": "SER past: yo fui",
            "sequence": "Y O space S E R",
            "tense": "past",
            "expected": "YO FUI",
        },
        {
            "name": "SER future: yo seré",
            "sequence": "Y O space S E R",
            "tense": "future",
            "expected": "YO SERÉ",
        },
        # Chained verbs
        {
            "name": "Chained PAST: ayer yo creí ellos creyeron",
            "sequence": "A Y E R space Y O space C R E E R space E L L O S space C R E E R",
            "tense": None,  # Auto-detect from "ayer"
            "expected": "AYER YO CREÍ ELLOS CREYERON",
        },
    ]

    print("Integration Tests: Conjugation with Tense Support")
    print("=" * 70)

    passed = 0
    failed = 0

    for test_case in test_cases:
        buffer.clear()
        name = test_case["name"]
        sequence = test_case["sequence"].split()
        tense = test_case.get("tense")
        expected = test_case["expected"]

        # Simulate letter-by-letter input with cooldown frames.
        for letter in sequence:
            if letter == "space":
                while not buffer.update("space"):
                    buffer.update(None)
            else:
                while not buffer.update(letter):
                    buffer.update(None)

        # Get text with explicit tense or auto-detect
        result = buffer.get_text(override_tense=tense)

        # Compare (case-insensitive for display)
        matches = result.lower() == expected.lower()

        if matches:
            print(f"[PASS] {name}")
            print(f"       Result: {result}")
            passed += 1
        else:
            print(f"[FAIL] {name}")
            print(f"       Expected: {expected}")
            print(f"       Got:      {result}")
            failed += 1

        print()

    print("=" * 70)
    print(f"Results: {passed} passed, {failed} failed")

    if failed == 0:
        print("All tests passed!")
        return True
    else:
        print(f"{failed} test(s) failed.")
        return False


if __name__ == "__main__":
    success = test_conjugation_with_tenses()
    exit(0 if success else 1)
