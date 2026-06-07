"""
Real-time WORD data capture with the webcam (temporal sequences).

ASL words are movement, not a static pose, so each take is a SEQUENCE of
landmark frames. Unlike the old version (which collapsed each take into a
single mean+std vector), this saves the full ORDERED sequence: the temporal
word model (TCN) reads the movement over time, which is what lets it tell
similar signs apart as the vocabulary grows.

Each take is recorded at a variable length, then resampled to WORD_SEQ_LEN
frames (src.utils.resample_sequence) and saved as one (WORD_SEQ_LEN, 63) array.
This is the EXACT same preprocessing inference uses, so what you capture matches
what the model sees live.

Output (same layout the trainer reads):
    data/real_capture/words/seq/<gloss>_<n>.npy   # (WORD_SEQ_LEN, 63)
    data/real_capture/words/manifest.csv          # sample_id, gloss, subset

Re-runs APPEND (counts continue), so capturing across several days builds a
multi-session dataset — the single most important thing for a model that
generalizes. Capture the same words on different days / lighting / clothing.

Keys during capture:
    SPACE  — start/stop recording one take (auto-stops at WORD_BUFFER_FRAMES)
    S      — skip to the next word
    Q      — finish and save

Usage:
    python capture/capture_words.py
"""

import csv
import os
import sys
from collections import deque

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import (
    CAPTURE_TARGET_PER_WORD as _CFG_TARGET,
    CAPTURE_SEQ_LEN         as _SEQ_LEN,
    CAPTURE_OUTPUT_DIR      as _OUTPUT_DIR,
    WORD_BUFFER_FRAMES      as _MAX_REC,
    WORD_MIN_FRAMES         as _MIN_REC,
    WORD_NULL_LABEL         as _NEGATIVE_LABEL,
)
from src.detector import Detector
from src.utils import normalize_landmarks, resample_sequence

# -------------------------------------------------------------------
# Vocabulary to capture. ASL GLOSSES in English (citation form) — the LLM
# layer turns recognized glosses into fluent Spanish, so the recognizer only
# needs the base sign, never conjugations.
#
# Curated for: (1) conversational frequency, (2) being visually DISTINCT so the
# model can separate them. Start small (~12-15) and grow in blocks of ~5 once
# accuracy holds. Edit this list freely; already-captured words are auto-skipped
# once they reach the target, so adding a word and re-running only records the
# new one.
# -------------------------------------------------------------------
WORDS = [
    "yes", "no", "hello", "thanks", "please", "sorry",
    "want", "need", "help", "eat", "drink", "more",
    "good", "finished", "you",
    _NEGATIVE_LABEL,   # <- capture MANY varied non-signs for this one
]

TARGET_PER_WORD = _CFG_TARGET
# The negative class must cover many non-sign poses/motions, so capture more.
WORD_TARGET = {_NEGATIVE_LABEL: 40}

# Rotating prompts so each "nothing" take is a DIFFERENT non-sign — the
# diversity is what makes the negative class actually reject non-signs instead
# of memorizing one. All keep the hand IN FRAME, since that is exactly when the
# model otherwise misfires (relaxed hand, transition between signs).
NEGATIVE_PROMPTS = [
    "relaxed open hand in frame (no sign)", "move hand slowly side to side",
    "transition between two signs (don't hold)", "wiggle fingers randomly",
    "scratch face / touch hair", "hand up, relaxed, no handshape",
    "slowly raise and lower the hand", "random meaningless gesture",
    "rotate the hand, no handshape", "rest hand in frame, palm down",
]

VAL_EVERY = 5   # every Nth take of a word goes to the validation split

HERE     = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_DIR  = os.path.join(HERE, _OUTPUT_DIR)
SEQ_DIR  = os.path.join(OUT_DIR, "seq")
MANIFEST = os.path.join(OUT_DIR, "manifest.csv")

# BGR colors
_WHITE  = (255, 255, 255); _BLACK = (0, 0, 0); _GREEN = (0, 220, 0)
_RED    = (50, 50, 220);   _YELLOW = (0, 210, 255); _GRAY = (160, 160, 160)


def _existing_counts() -> dict:
    """Count saved takes per word so re-runs append instead of clobbering."""
    counts = {}
    if os.path.isfile(MANIFEST):
        for r in csv.DictReader(open(MANIFEST, encoding="utf-8")):
            counts[r["gloss"]] = counts.get(r["gloss"], 0) + 1
    return counts


def _save_take(word, buf, counts, new_rows):
    """Resample the recorded take to WORD_SEQ_LEN and persist it."""
    if len(buf) < _MIN_REC:
        print(f"  [{word}] take too short ({len(buf)} frames), discarded")
        return False
    n = counts.get(word, 0)
    seq = resample_sequence(buf, _SEQ_LEN)        # (WORD_SEQ_LEN, 63)
    sample_id = f"{word}_{n:03d}"
    np.save(os.path.join(SEQ_DIR, f"{sample_id}.npy"), seq)
    # Spread val takes across the run; never make take 0 a val sample.
    subset = "val" if (n > 0 and n % VAL_EVERY == 0) else "train"
    new_rows.append([sample_id, word, subset])
    counts[word] = n + 1
    print(f"  [{word}] saved take {n + 1}  ({subset})")
    return True


def _draw_hud(frame, word, count, target, w_idx, recording, rec_len, flash, prompt):
    h, w = frame.shape[:2]
    band_h = 150 if prompt else 120
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, band_h), (30, 30, 30), -1)
    cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)

    color_word = _RED if recording else _WHITE
    cv2.putText(frame, word.upper(), (20, 80),
                cv2.FONT_HERSHEY_SIMPLEX, 1.8, _BLACK, 6, cv2.LINE_AA)
    cv2.putText(frame, word.upper(), (20, 80),
                cv2.FONT_HERSHEY_SIMPLEX, 1.8, color_word, 3, cv2.LINE_AA)
    cv2.putText(frame, f"{count}/{target}", (20, 112),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, _GREEN, 2, cv2.LINE_AA)

    if prompt:
        cv2.putText(frame, f"-> {prompt}", (160, 112),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, _YELLOW, 2, cv2.LINE_AA)
    if recording:
        cv2.putText(frame, f"REC {rec_len}/{_MAX_REC}", (20, 142),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, _RED, 2, cv2.LINE_AA)

    instr = ("Perform the FULL sign - auto-stops when full"
             if recording else "SPACE: record take | S: skip | Q: save & quit")
    cv2.putText(frame, instr, (10, h - 14),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, _GRAY, 1, cv2.LINE_AA)

    if flash > 0:
        ov = frame.copy()
        cv2.rectangle(ov, (0, 0), (w, h), (0, 200, 0), -1)
        cv2.addWeighted(ov, flash * 0.25, frame, 1 - flash * 0.25, 0, frame)


def main():
    os.makedirs(SEQ_DIR, exist_ok=True)
    counts = _existing_counts()
    new_rows = []

    print("Starting camera...")
    detector = Detector()
    print(f"Capturing {len(WORDS)} words (camera). Re-runs append for multi-session data.")
    print("SPACE: record take | S: skip word | Q: save & quit\n")

    w_idx = 0
    recording = False
    buf = deque(maxlen=_MAX_REC)
    flash = 0.0

    while w_idx < len(WORDS):
        word = WORDS[w_idx]
        target = WORD_TARGET.get(word, TARGET_PER_WORD)
        if counts.get(word, 0) >= target:
            w_idx += 1
            continue

        frame, lm = detector.get_frame()
        if frame is None:
            break
        flash = max(0.0, flash - 0.06)
        hand_ok = lm["num_hands"] >= 1 and lm["landmarks_hand1"] is not None

        if recording and hand_ok:
            buf.append(normalize_landmarks(lm["landmarks_hand1"]))
            if len(buf) >= _MAX_REC:
                recording = False
                if _save_take(word, list(buf), counts, new_rows):
                    flash = 1.0
                buf.clear()

        prompt = (NEGATIVE_PROMPTS[counts.get(word, 0) % len(NEGATIVE_PROMPTS)]
                  if word == _NEGATIVE_LABEL else None)
        _draw_hud(frame, word, counts.get(word, 0), target, w_idx,
                  recording, len(buf), flash, prompt)
        cv2.imshow("Word capture (sequences) - Sign Language Translator", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            print("\nFinishing...")
            break
        if key == ord("s"):
            print(f"  [{word}] skipped")
            recording = False
            buf.clear()
            w_idx += 1
            continue
        if key == ord(" "):
            if not recording:
                recording = True
                buf.clear()
            else:
                recording = False
                if _save_take(word, list(buf), counts, new_rows):
                    flash = 1.0
                buf.clear()

    detector.release()
    cv2.destroyAllWindows()

    if not new_rows:
        print("No new takes captured.")
        return

    exists = os.path.isfile(MANIFEST)
    with open(MANIFEST, "a", newline="", encoding="utf-8") as f:
        wr = csv.writer(f)
        if not exists:
            wr.writerow(["sample_id", "gloss", "subset"])
        wr.writerows(new_rows)

    print(f"\nSaved {len(new_rows)} new takes -> {SEQ_DIR}")
    print(f"Manifest -> {MANIFEST}")
    print("\nNext: python training/train_words.py")


if __name__ == "__main__":
    main()
