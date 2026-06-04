#!/usr/bin/env python3
"""
Integration test: simulate real letter-by-letter input and verify conjugation.

This test mimics the actual flow:
1. User spells a pronoun letter by letter
2. User presses space
3. User spells a verb letter by letter
4. System conjugates automatically
"""

from src.overlay import LetterBuffer

def test_conjugation_flow():
    """Simulate real user input."""
    buffer = LetterBuffer()

    test_cases = [
        {
            "name": "yo creo (1st person singular)",
            "sequence": "Y O space C R E E R",
            "expected": "yo creo",
        },
        {
            "name": "ellos creen (3rd person plural)",
            "sequence": "E L L O S space C R E E R",
            "expected": "ellos creen",
        },
        {
            "name": "nosotros creemos (1st person plural)",
            "sequence": "N O S O T R O S space C R E E R",
            "expected": "nosotros creemos",
        },
        {
            "name": "tu crees (2nd person singular, without accent)",
            "sequence": "T U space C R E E R",
            "expected": "TU CREES",
        },
        {
            "name": "yo soy (irregular verb)",
            "sequence": "Y O space S E R",
            "expected": "yo soy",
        },
        {
            "name": "Chained: yo puedo ellos pueden",
            "sequence": "Y O space P O D E R space E L L O S space P O D E R",
            "expected": "yo puedo ellos pueden",
        },
    ]

    print("Integration Tests: Letter-by-letter conjugation")
    print("=" * 60)

    passed = 0
    failed = 0

    for test_case in test_cases:
        buffer.clear()
        name = test_case["name"]
        sequence = test_case["sequence"].split()
        expected = test_case["expected"]

        # Simulate letter-by-letter input with cooldown frames.
        # Each call to update() with None decrements cooldown;
        # when cooldown reaches 0, the next letter is accepted.
        for letter in sequence:
            if letter == "space":
                # Keep calling with None until space is accepted
                while not buffer.update("space"):
                    buffer.update(None)  # decrement cooldown
            else:
                # Keep calling with None until letter is accepted
                while not buffer.update(letter):
                    buffer.update(None)  # decrement cooldown

        result = buffer.get_text()

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

    print("=" * 60)
    print(f"Results: {passed} passed, {failed} failed")

    if failed == 0:
        print("All tests passed!")
        return True
    else:
        print(f"{failed} test(s) failed.")
        return False


if __name__ == "__main__":
    success = test_conjugation_flow()
    exit(0 if success else 1)
