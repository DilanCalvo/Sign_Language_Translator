"""
Letter accumulation buffer and subtitle UI.

LetterBuffer collects stable letter predictions with a per-letter cooldown
to prevent the same held pose from flooding the buffer, then draws the
accumulated text as a subtitle strip at the bottom of the video frame.

It also automatically conjugates verbs based on the preceding pronoun context.
"""

import cv2
import numpy as np

from config import LETTER_COOLDOWN_FRAMES, DEFAULT_TENSE
from src.grammar import conjugate_verb, is_pronoun, detect_tense

_MAX_VISIBLE_CHARS = 50


class LetterBuffer:
    """
    Accumulates confirmed letter predictions into a running text line with
    automatic verb conjugation based on pronoun context.

    Call update() every frame with the current stable letter (or None).
    It returns True the frame a new character is accepted — use that
    signal to trigger voice output.

    Special values:
        "del"   → remove the last character
        "space" → append a space (and trigger conjugation logic)
        any str → append uppercase

    Conjugation:
        When a space is added, the buffer checks if the previous complete word
        is a pronoun. If it is, and the next complete word is a verb in the
        grammar table, the verb is automatically conjugated to match the pronoun.
    """

    def __init__(self):
        self._chars    = []
        self._cooldown = 0

    def _get_last_word(self) -> str | None:
        """Extract the last complete word from _chars (before the final space)."""
        if not self._chars:
            return None
        # Join all chars and split by spaces to get the last word
        text = "".join(self._chars).rstrip()
        words = text.split()
        return words[-1] if words else None

    def _extract_words(self) -> list[str]:
        """Extract all words (non-space sequences) from _chars."""
        text = "".join(self._chars)
        return text.split()

    def update(self, letter) -> bool:
        """
        Offer the current stable letter to the buffer.

        Returns True if a character was accepted this frame, False otherwise.
        """
        if self._cooldown > 0:
            self._cooldown -= 1

        if letter is None or self._cooldown > 0:
            return False

        if letter == "del":
            if self._chars:
                self._chars.pop()
        elif letter == "space":
            # After space is added, check if we need to conjugate the next word.
            # We mark a special state to conjugate when the next word arrives.
            self._chars.append(" ")
        else:
            # Before adding the letter, check if we're starting a new word
            # right after a space that follows a pronoun.
            if self._should_conjugate_next_word():
                # Replace the next word with its conjugated form.
                letter = self._apply_conjugation(letter)
            self._chars.append(letter.upper())

        self._cooldown = LETTER_COOLDOWN_FRAMES
        return True

    def _should_conjugate_next_word(self) -> bool:
        """
        Check if the last word (before the last space) is a pronoun.

        This is used to decide if we should conjugate the word that is about
        to be added. If the last complete word is a pronoun, the next word
        (the one about to start) should be conjugated.
        """
        text = "".join(self._chars).rstrip()
        words = text.split()
        if not words:
            return False

        last_word = words[-1]
        return is_pronoun(last_word)

    def _apply_conjugation(self, first_letter: str) -> str:
        """
        If the upcoming word is a verb, conjugate it and return the first letter
        of the conjugated form.

        This is called at the moment the first letter of a new word arrives.
        We don't have the full word yet, so we just return the first letter
        (which will be uppercase). The conjugation is deferred until the
        word is complete.

        For now, we return the letter unchanged and conjugate at display time.
        """
        # Defer conjugation to display time (see _conjugate_for_display).
        return first_letter

    def _conjugate_for_display(self, text: str, override_tense: str | None = None) -> str:
        """
        Apply conjugation to the displayed text based on pronoun context and tense.

        This processes the full accumulated text and conjugates verbs that
        follow pronouns, applying the appropriate tense. It is called just
        before display, ensuring we have complete words to work with.

        Args:
            text: accumulated text to conjugate
            override_tense: if provided, use this tense instead of auto-detecting
        """
        words = text.split()
        if len(words) < 2:
            return text

        # Detect tense: either from override or auto-detect from text
        if override_tense:
            tense = override_tense.lower()
        else:
            tense = detect_tense(text)

        conjugated_words = []
        for i, word in enumerate(words):
            if i == 0:
                conjugated_words.append(word)
            else:
                prev_word = words[i - 1]
                if is_pronoun(prev_word):
                    # Conjugate: convert word to lowercase, conjugate with tense, then uppercase
                    conjugated = conjugate_verb(word.lower(), prev_word, tense=tense).upper()
                    conjugated_words.append(conjugated)
                else:
                    conjugated_words.append(word)

        return " ".join(conjugated_words)

    def get_text(self, override_tense: str | None = None) -> str:
        """
        Return the accumulated text (last _MAX_VISIBLE_CHARS characters).

        Args:
            override_tense: if provided, force this tense instead of auto-detecting.
                          Useful for manual tense selection (e.g., user presses 'T' for past).
        """
        raw_text = "".join(self._chars[-_MAX_VISIBLE_CHARS:])
        return self._conjugate_for_display(raw_text, override_tense=override_tense)

    def clear(self) -> None:
        self._chars.clear()
        self._cooldown = 0

    def draw_subtitle(self, frame, override_tense: str | None = None) -> None:
        """
        Draw a semi-transparent subtitle bar at the bottom of the frame.

        Args:
            frame: video frame to draw on
            override_tense: if provided, use this tense instead of auto-detecting
        """
        text = self.get_text(override_tense=override_tense)
        if not text:
            return

        h, w = frame.shape[:2]
        bar_h = 52
        y0    = h - bar_h
        padding = 14
        max_w = w - 2 * padding
        font = cv2.FONT_HERSHEY_SIMPLEX
        scale = 1.0
        thick = 2

        # Scale down if text is wider than available space
        (txt_w, _), _ = cv2.getTextSize(text, font, scale, thick)
        if txt_w > max_w:
            scale = max(0.55, max_w / txt_w)
            (txt_w, _), _ = cv2.getTextSize(text, font, scale, thick)
            # Still too wide: trim oldest characters to keep the most recent
            while txt_w > max_w and len(text) > 1:
                text = text[1:]
                (txt_w, _), _ = cv2.getTextSize(text, font, scale, thick)

        # Semi-transparent dark background over the bottom strip
        roi = frame[y0:h, 0:w]
        dark = roi.copy()
        dark[:] = (20, 20, 20)
        cv2.addWeighted(dark, 0.55, roi, 0.45, 0, roi)
        frame[y0:h, 0:w] = roi

        cv2.putText(
            frame, text,
            (padding, h - 14),
            font, scale,
            (255, 255, 255), thick, cv2.LINE_AA,
        )


class SpeechBuffer:
    """
    Accumulates speech-to-text transcripts and draws a subtitle bar at the
    TOP of the frame — visually distinct (dark navy) from the ASL
    LetterBuffer bar at the bottom (dark gray).

    Call add() when a new transcript arrives. Call draw() every frame,
    passing the current speech state as a lowercase string so the bar
    shows live status even before the first transcript appears.

    status strings (from SpeechState.name.lower()):
        ""            — hide bar when no transcripts exist
        "idle"        — hide bar when no transcripts exist
        "loading"     — show "cargando modelo..." indicator
        "listening"   — show "[ REC ]" indicator
        "transcribing"— show "procesando..." indicator
        "unavailable" — show "no disponible" indicator
    """

    _MAX_LINES      = 2   # keep the last two utterances visible at once
    _MAX_LINE_CHARS = 55  # truncate longer utterances to prevent overflow

    def __init__(self) -> None:
        self._lines: list[str] = []

    def add(self, text: str) -> None:
        """Append a new transcript, trimming history to _MAX_LINES."""
        if len(text) > self._MAX_LINE_CHARS:
            text = text[:self._MAX_LINE_CHARS - 1] + "~"
        self._lines.append(text)
        if len(self._lines) > self._MAX_LINES:
            self._lines = self._lines[-self._MAX_LINES:]

    def clear(self) -> None:
        self._lines.clear()

    def draw(self, frame: np.ndarray, status: str = "") -> None:
        """
        Draw a semi-transparent bar at the top of the frame.

        The bar is hidden when there is no content and status is idle/empty.
        """
        is_active = status in ("loading", "listening", "transcribing")
        if not self._lines and not is_active:
            return

        h, w   = frame.shape[:2]
        font   = cv2.FONT_HERSHEY_SIMPLEX
        scale  = 0.65
        thick  = 1
        line_h = 26
        pad    = 7

        # 1 header row ("Oyente:" + status indicator) + N transcript rows
        n_rows = 1 + len(self._lines)
        bar_h  = n_rows * line_h + pad * 2

        # Dark navy semi-transparent background
        roi = frame[0:bar_h, 0:w]
        bg  = roi.copy()
        bg[:] = (45, 20, 5)   # very dark blue-navy in BGR
        cv2.addWeighted(bg, 0.68, roi, 0.32, 0, roi)
        frame[0:bar_h, 0:w] = roi

        # Header row — "Oyente:" label (left) and status indicator (right)
        header_y = pad + line_h - 4
        cv2.putText(frame, "Oyente:", (pad, header_y),
                    font, 0.55, (140, 190, 230), 1, cv2.LINE_AA)

        _STATUS = {
            "loading":      ("[ cargando modelo... ]", (255, 160,  80)),
            "listening":    ("[ REC ]",                (60,   60, 220)),
            "transcribing": ("[ procesando... ]",      (0,   200, 200)),
            "unavailable":  ("[ no disponible ]",      (80,   80,  80)),
        }
        if status in _STATUS:
            tag, color = _STATUS[status]
            (tw, _), _ = cv2.getTextSize(tag, font, 0.55, 1)
            cv2.putText(frame, tag, (w - tw - pad, header_y),
                        font, 0.55, color, 1, cv2.LINE_AA)

        # Transcript lines (oldest first, newest last)
        for i, line in enumerate(self._lines):
            y = pad + (i + 2) * line_h - 4
            cv2.putText(frame, line, (pad + 10, y),
                        font, scale, (240, 240, 240), thick, cv2.LINE_AA)
