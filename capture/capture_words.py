"""
Real-time WORD data capture with the webcam.

Unlike letters (static poses), ASL words require motion. This script captures
SEQUENCES of landmarks (N consecutive frames) and computes temporal
statistics (mean + standard deviation) that encode the movement. The result
is a CSV with 126 features per sample (63 mean + 63 std) that the word model
uses as input.

Usage:
    python capture/capture_words.py

Keys during capture:
    SPACE  — start/stop recording a sign
    S      — skip the current word
    Q      — finish and save

Result:
    data/real_capture/words/words_<timestamp>.csv
"""

import csv
import os
import sys
import time
from datetime import datetime
from collections import deque

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import (
    CAPTURE_TARGET_PER_WORD as _CFG_TARGET,
    CAPTURE_SEQ_FRAMES      as _CFG_SEQ_FRAMES,
    CAPTURE_OUTPUT_DIR      as _CFG_OUTPUT_DIR,
)
from src.detector import Detector
from src.utils import normalize_landmarks

# -------------------------------------------------------------------
# Configuration — edit this list with the words you want to capture.
# Labels are the words (in any language) your signs map to. Reference set:
#
# WORDS = [
#     "J", "Z", "hola", "adios", "gracias", "ayuda", "si", "no", "agua",
#     "bano", "por_favor", "perdon", "bien", "mal", "nombre", "comer", "dormir",
# ]
# -------------------------------------------------------------------

WORDS = [
    "adios",
]

TARGET_PER_WORD  = _CFG_TARGET      # sequences to capture per word
SEQ_FRAMES       = _CFG_SEQ_FRAMES  # frames per sequence (~0.67s at 30fps)
OUTPUT_DIR       = _CFG_OUTPUT_DIR

# Classes that need different instructions on each capture to force diversity
# in the data. The "nada" class is the model's negative class: it represents
# "everything that is NOT a deliberate word" (static poses, transitions,
# random gestures). If every sample looked the same the model would learn a
# single pattern; the prompts rotate in order so each capture is a different
# gesture. (The key "nada" must match the trained label — do not translate it.)
SPECIAL_INSTRUCTIONS = {
    "nada": [
        "Hold the letter B (flat open hand)",
        "Hold the letter A (closed fist)",
        "Move the hand randomly",
        "Hold the letter O (curved fingers)",
        "Hold the letter C",
        "Slow transition between two letters",
        "Hold the letter L",
        "Hold the letter Y (fist with thumb and pinky)",
        "Hold the letter D",
        "Short meaningless gesture",
        "Hold the letter F",
        "Raise the hand and lower it",
        "Hold the letter W (three fingers)",
        "Rotate the hand without changing the pose",
        "Hold the letter V (peace)",
        "Fast transition between two letters",
        "Hold the letter P",
        "Move the hand closer and farther from the camera",
        "Hold the letter R",
        "Move only the fingers without changing the palm",
    ],
}

# -------------------------------------------------------------------
# BGR colors
# -------------------------------------------------------------------
_WHITE  = (255, 255, 255)
_BLACK  = (0,   0,   0)
_GREEN  = (0,   220, 0)
_RED    = (50,  50,  220)
_YELLOW = (0,   210, 255)
_GRAY   = (160, 160, 160)
_CYAN   = (220, 200, 0)
_BG     = (30,  30,  30)


def _shadow_text(frame, text, pos, scale, color, thick):
    cv2.putText(frame, text, (pos[0]+2, pos[1]+2),
                cv2.FONT_HERSHEY_SIMPLEX, scale, _BLACK, thick + 2, cv2.LINE_AA)
    cv2.putText(frame, text, pos,
                cv2.FONT_HERSHEY_SIMPLEX, scale, color, thick, cv2.LINE_AA)


def _draw_word_hud(frame, word, count, words_done, recording, seq_progress, flash_alpha, prompt=None):
    h, w = frame.shape[:2]

    # Taller top band if there is a rotating prompt (extra lines).
    band_h = 150 if prompt else 120
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, band_h), _BG, -1)
    cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)

    # Target word
    color_word = _RED if recording else _WHITE
    _shadow_text(frame, word.upper(), (20, 95), 3.2, color_word, 5)

    # Per-capture prompt (only for classes with SPECIAL_INSTRUCTIONS).
    # Shown under the class name and changes with each sample.
    if prompt:
        _shadow_text(frame, f"-> {prompt}", (20, 135), 0.7, _YELLOW, 2)

    # Sequence progress (accumulated frames)
    if recording:
        ratio = seq_progress / SEQ_FRAMES
        bar_color = _RED
        bar_label = f"Recording: {seq_progress}/{SEQ_FRAMES} frames"
    else:
        ratio = count / TARGET_PER_WORD
        bar_color = _GREEN
        bar_label = f"Samples: {count}/{TARGET_PER_WORD}"

    bar_x, bar_y, bar_w = 230, 15, w - 250
    cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_w, bar_y + 20), (60, 60, 60), -1)
    if ratio > 0:
        cv2.rectangle(frame, (bar_x, bar_y),
                      (bar_x + int(bar_w * ratio), bar_y + 20), bar_color, -1)
    cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_w, bar_y + 20), _GRAY, 1)
    cv2.putText(frame, bar_label, (bar_x, bar_y + 38),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, _WHITE, 1, cv2.LINE_AA)

    progress_global = f"Words completed: {words_done}/{len(WORDS)}"
    cv2.putText(frame, progress_global, (bar_x, bar_y + 58),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, _GRAY, 1, cv2.LINE_AA)

    # Flash
    if flash_alpha > 0:
        green_overlay = frame.copy()
        cv2.rectangle(green_overlay, (0, 0), (w, h), (0, 200, 0), -1)
        cv2.addWeighted(green_overlay, flash_alpha * 0.3, frame, 1 - flash_alpha * 0.3, 0, frame)

    # Instructions
    if recording:
        instr = "Perform the full sign - recording automatically"
    else:
        instr = "SPACE: start sign  |  S: skip  |  Q: save and quit"
    cv2.putText(frame, instr, (10, h - 12),
                cv2.FONT_HERSHEY_SIMPLEX, 0.42, _GRAY, 1, cv2.LINE_AA)

    # Current-sign hint
    cv2.putText(frame, "Perform the full sign of the word shown above",
                (10, h - 35), cv2.FONT_HERSHEY_SIMPLEX, 0.4, _CYAN, 1, cv2.LINE_AA)


def _compute_temporal_features(seq_normalized):
    """
    Given a sequence of normalized arrays (SEQ_FRAMES x 63), returns
    [mean(63) + std(63)] = 126 features. This encodes both the average static
    shape and the variation (movement).
    """
    arr = np.stack(seq_normalized, axis=0)  # (SEQ_FRAMES, 63)
    mean = arr.mean(axis=0)                 # (63,)
    std  = arr.std(axis=0)                  # (63,)
    return np.concatenate([mean, std]).astype(np.float32)


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = os.path.join(OUTPUT_DIR, f"words_{timestamp}.csv")

    # Columns: label + 63 means + 63 stds
    cols_mean = [f"mean_{c}{i}" for i in range(21) for c in ("x", "y", "z")]
    cols_std  = [f"std_{c}{i}"  for i in range(21) for c in ("x", "y", "z")]
    columns = ["label"] + cols_mean + cols_std

    rows = []

    print("Starting camera...")
    detector = Detector()
    print(f"Capturing {len(WORDS)} words x {TARGET_PER_WORD} sequences")
    print(f"Each sequence = {SEQ_FRAMES} frames of landmarks\n")

    word_idx    = 0
    count       = 0
    words_done  = 0
    recording   = False
    seq_buffer  = deque(maxlen=SEQ_FRAMES)
    flash_alpha = 0.0

    while word_idx < len(WORDS):
        word = WORDS[word_idx]
        frame, lm_data = detector.get_frame()
        if frame is None:
            break

        flash_alpha = max(0.0, flash_alpha - 0.07)
        hand_ok = lm_data["num_hands"] >= 1 and lm_data["landmarks_hand1"] is not None

        # While recording, accumulate frames
        if recording:
            if hand_ok:
                normed = normalize_landmarks(lm_data["landmarks_hand1"])
                seq_buffer.append(normed)

            if len(seq_buffer) >= SEQ_FRAMES:
                # Full sequence — extract temporal features
                features = _compute_temporal_features(list(seq_buffer))
                rows.append([word] + features.tolist())
                count       += 1
                flash_alpha  = 1.0
                recording    = False
                seq_buffer.clear()
                print(f"  [{word}] {count}/{TARGET_PER_WORD}", end="\r")

                if count >= TARGET_PER_WORD:
                    print(f"\n  [{word}] completed.")
                    words_done += 1
                    word_idx   += 1
                    count       = 0

        seq_progress = len(seq_buffer)

        # If the class has rotating prompts, pick the one matching how many
        # samples we have so far. This makes each capture a different gesture.
        prompts = SPECIAL_INSTRUCTIONS.get(word)
        prompt  = prompts[count % len(prompts)] if prompts else None

        _draw_word_hud(frame, word, count, words_done, recording, seq_progress, flash_alpha, prompt)
        cv2.imshow("Word capture - Sign Language Translator", frame)

        key = cv2.waitKey(1) & 0xFF

        if key == ord("q"):
            print("\nCapture interrupted.")
            break
        if key == ord("s"):
            print(f"  [{word}] skipped ({count} samples saved)")
            word_idx += 1
            count     = 0
            recording = False
            seq_buffer.clear()
            continue
        if key == ord(" ") and not recording and hand_ok:
            recording = True
            seq_buffer.clear()

    detector.release()
    cv2.destroyAllWindows()

    if not rows:
        print("No samples captured.")
        return

    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(columns)
        writer.writerows(rows)

    from collections import Counter
    label_counts = Counter(r[0] for r in rows)
    print(f"\n{'=' * 55}")
    print(f"  Total captured: {len(rows):,} sequences")
    print(f"\n  Distribution:")
    for w in WORDS:
        cnt = label_counts.get(w, 0)
        bar = "#" * (cnt * 20 // TARGET_PER_WORD) if TARGET_PER_WORD > 0 else ""
        print(f"  {w:>12}: {cnt:>4}  [{bar:<20}]")
    print(f"{'=' * 55}")
    print(f"\nFile saved to: {output_path}")
    print(f"\nAdd this CSV to DATA_CSVS in training/train_words.py to train.")


if __name__ == "__main__":
    main()
