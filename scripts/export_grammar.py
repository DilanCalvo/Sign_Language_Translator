#!/usr/bin/env python3
"""
Export grammar tables (verbs and pronouns) to JSON for web consumption.

This ensures parity between Python and JavaScript conjugation logic.
Run this script whenever VERB_CONJUGATIONS or PRONOUN_TO_FORM in config.py
are updated.

    python scripts/export_grammar.py
"""

import json
import sys
from pathlib import Path

# Add project root to path so we can import config
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.grammar import export_grammar_table

OUTPUT_PATH = Path(__file__).parent.parent / "model" / "grammar.json"


def main():
    """Export grammar tables to JSON."""
    table = export_grammar_table()

    # Write with nice formatting for readability
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(table, f, indent=2, ensure_ascii=False)

    print(f"[OK] Grammar table exported to {OUTPUT_PATH}")
    print(f"  Verbs: {len(table['verbs'])}")
    print(f"  Pronouns: {len(table['pronouns'])}")


if __name__ == "__main__":
    main()
