"""
Letter accumulation buffer and subtitle UI.

LetterBuffer collects stable letter predictions with a per-letter cooldown
to prevent the same held pose from flooding the buffer, then draws the
accumulated text as a subtitle strip at the bottom of the video frame.
"""

import cv2
import numpy as np

from config import LETTER_COOLDOWN_FRAMES, WORD_SENTENCE_PAUSE_FRAMES

_MAX_VISIBLE_CHARS = 50


class LetterBuffer:
    """
    Accumulates confirmed letter predictions into a running text line.

    Call update() every frame with the current stable letter (or None).
    It returns True the frame a new character is accepted — use that
    signal to trigger voice output.

    Special values:
        "del"   → remove the last character
        "space" → append a space
        any str → append uppercase
    """

    def __init__(self):
        self._chars     = []
        self._cooldown  = 0
        # Words finalized by a space, waiting to be drained into the
        # conversation log. See pop_completed_words().
        self._completed = []

    def _get_last_word(self) -> str | None:
        """Extract the last complete word from _chars (before the final space)."""
        if not self._chars:
            return None
        # Join all chars and split by spaces to get the last word
        text = "".join(self._chars).rstrip()
        words = text.split()
        return words[-1] if words else None

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
            # A space finalizes the current word. Record it (once) so the
            # conversation log can capture spelled words, then append the space.
            if self._chars and self._chars[-1] != " ":
                finished = self._get_last_word()
                if finished:
                    self._completed.append(finished)
            self._chars.append(" ")
        else:
            self._chars.append(letter.upper())

        self._cooldown = LETTER_COOLDOWN_FRAMES
        return True

    def get_text(self) -> str:
        """Return the accumulated text (last _MAX_VISIBLE_CHARS characters)."""
        return "".join(self._chars[-_MAX_VISIBLE_CHARS:])

    def pop_completed_words(self) -> list[str]:
        """
        Return and clear the words finalized since the last call.

        Used by the conversation log to capture spelled words at word
        boundaries without coupling the buffer to the logger.
        """
        words = self._completed
        self._completed = []
        return words

    def clear(self) -> None:
        self._chars.clear()
        self._completed.clear()
        self._cooldown = 0

    def draw_subtitle(self, frame) -> None:
        """Draw a semi-transparent subtitle bar at the bottom of the frame."""
        text = self.get_text()
        if not text:
            return

        h, w = frame.shape[:2]
        bar_h = 52
        # Lift the bar off the very bottom edge so it is never clipped by the
        # window border / OS taskbar.
        margin = 40
        y1 = h - margin
        y0 = y1 - bar_h
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

        # Semi-transparent dark background over the strip
        roi = frame[y0:y1, 0:w]
        dark = roi.copy()
        dark[:] = (20, 20, 20)
        cv2.addWeighted(dark, 0.55, roi, 0.45, 0, roi)
        frame[y0:y1, 0:w] = roi

        cv2.putText(
            frame, text,
            (padding, y1 - 16),
            font, scale,
            (255, 255, 255), thick, cv2.LINE_AA,
        )


class WordBuffer:
    """
    Accumulates whole-word signs (word mode) into a running sentence.

    Each detected sign is appended so the bottom strip shows the recent
    sequence of signs (e.g. "yo pensar"). The sentence auto-clears after
    WORD_SENTENCE_PAUSE_FRAMES with no new sign, so the user just pauses to
    start a fresh sentence (no key required).

    Usage (word mode):
        spoken = word_buffer.add(detected_word)  # on each new sign
        word_buffer.tick()                       # every frame
        word_buffer.draw_subtitle(frame)
    """

    def __init__(self):
        self._words = []
        self._idle  = 0   # frames since the last sign was added

    def add(self, word) -> str | None:
        """Append a detected sign and return it (the form to speak), or None."""
        if not word:
            return None
        self._words.append(word)
        self._idle = 0
        return word

    def tick(self) -> None:
        """Advance the inactivity timer; auto-clear after the pause window."""
        if not self._words:
            return
        self._idle += 1
        if self._idle >= WORD_SENTENCE_PAUSE_FRAMES:
            self.clear()

    def get_text(self) -> str:
        return " ".join(self._words)

    def clear(self) -> None:
        self._words.clear()
        self._idle = 0

    def draw_subtitle(self, frame) -> None:
        """
        Draw the accumulated sentence as a green-tinted bar at the bottom —
        tinted to visually distinguish word mode from the gray letter subtitle.
        """
        text = self.get_text()
        if not text:
            return

        h, w = frame.shape[:2]
        bar_h = 52
        # Lift the bar off the very bottom edge so it is never clipped.
        margin = 40
        y1 = h - margin
        y0 = y1 - bar_h
        padding = 14
        max_w = w - 2 * padding
        font = cv2.FONT_HERSHEY_SIMPLEX
        scale = 1.0
        thick = 2

        (txt_w, _), _ = cv2.getTextSize(text, font, scale, thick)
        if txt_w > max_w:
            scale = max(0.55, max_w / txt_w)
            (txt_w, _), _ = cv2.getTextSize(text, font, scale, thick)
            while txt_w > max_w and len(text) > 1:
                text = text[1:]
                (txt_w, _), _ = cv2.getTextSize(text, font, scale, thick)

        roi = frame[y0:y1, 0:w]
        dark = roi.copy()
        dark[:] = (0, 30, 0)   # dark green tint
        cv2.addWeighted(dark, 0.55, roi, 0.45, 0, roi)
        frame[y0:y1, 0:w] = roi

        cv2.putText(
            frame, text,
            (padding, y1 - 16),
            font, scale,
            (180, 255, 180), thick, cv2.LINE_AA,
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
