"""
Exportable conversation log.

ConversationLog records the running dialogue between the two participants so it
can be saved to disk at the end of a session:

    - "Senas"  — utterances produced by the signer (whole words in word mode,
                 completed spelled words in letter/number mode).
    - "Oyente" — utterances transcribed from the hearing person's speech.

Each entry is timestamped. The log is append-only in memory; export() writes a
human-readable .txt and a machine-readable .csv next to each other. Nothing is
written until the user asks to export (E key in main.py), so it adds zero I/O
to the per-frame loop.
"""

from __future__ import annotations

import csv
import os
from datetime import datetime

# Source labels. Kept consistent with the on-screen "Oyente:" header so the
# exported file reads the same way the user saw it live.
SOURCE_SIGN   = "Senas"
SOURCE_SPEECH = "Oyente"


class ConversationLog:
    def __init__(self) -> None:
        # Each entry: (datetime, source, text).
        self._entries: list[tuple[datetime, str, str]] = []

    def add(self, source: str, text: str) -> None:
        """Append one utterance. Empty/whitespace text is ignored."""
        text = (text or "").strip()
        if not text:
            return
        self._entries.append((datetime.now(), source, text))

    def add_sign(self, text: str) -> None:
        self.add(SOURCE_SIGN, text)

    def add_speech(self, text: str) -> None:
        self.add(SOURCE_SPEECH, text)

    def __len__(self) -> int:
        return len(self._entries)

    def clear(self) -> None:
        self._entries.clear()

    def export(self, out_dir: str = "logs") -> str | None:
        """
        Write the log to out_dir as both .txt and .csv.

        Returns the path of the .txt file, or None if the log is empty
        (nothing to export). Creates out_dir if needed.
        """
        if not self._entries:
            return None

        os.makedirs(out_dir, exist_ok=True)
        stamp    = datetime.now().strftime("%Y%m%d_%H%M%S")
        txt_path = os.path.join(out_dir, f"conversation_{stamp}.txt")
        csv_path = os.path.join(out_dir, f"conversation_{stamp}.csv")

        with open(txt_path, "w", encoding="utf-8") as f:
            f.write(f"Conversation log - {stamp}\n")
            f.write("=" * 40 + "\n\n")
            for when, source, text in self._entries:
                f.write(f"[{when:%H:%M:%S}] {source}: {text}\n")

        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["timestamp", "source", "text"])
            for when, source, text in self._entries:
                writer.writerow([when.isoformat(timespec="seconds"), source, text])

        return txt_path
