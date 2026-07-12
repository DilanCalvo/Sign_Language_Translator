"""
Real-time LETTER landmark capture with the webcam.

Produces a CSV with the raw landmarks of each signed letter. The result is
used directly to train the model with training/train_letters.py.

Usage:
    python capture/capture_letters.py

Keys during capture:
    SPACE  — manually capture the current frame (on top of auto-capture)
    S      — skip the current letter
    Q      — finish and save whatever was captured

Flow:
    The script shows which letter to sign. Whenever it detects a stable hand
    it captures automatically (every MIN_INTERVAL seconds). Once it reaches
    TARGET_PER_LABEL for a letter it moves on to the next one. When done it
    saves a CSV ready for training.

Result:
    data/real_capture/letters/capture_<timestamp>.csv
"""

import csv
import os
import sys
import time
from datetime import datetime

import cv2

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.detector import Detector

# -------------------------------------------------------------------
# Configuration
# -------------------------------------------------------------------

# Letters to capture. Edit this list to capture only the ones you need
# (e.g. the most confusing: N, M, E, S, A). Full alphabet for reference:
#
LABELS = [
     "A","B","C","D","E","F","G","H","I","J",
     "K","L","M","N","O","P","Q","R","S","T",
     "U","V","W","X","Y","Z","del","space",
 ]

TARGET_PER_LABEL  = 150    # samples to capture per letter
MIN_INTERVAL      = 0.65   # minimum seconds between automatic captures
OUTPUT_DIR        = "data/real_capture/letters"

# -------------------------------------------------------------------
# BGR colors
# -------------------------------------------------------------------
_WHITE  = (255, 255, 255)
_BLACK  = (0,   0,   0)
_GREEN  = (0,   220, 0)
_YELLOW = (0,   210, 255)
_GRAY   = (160, 160, 160)
_RED    = (50,  50,  220)
_CYAN   = (220, 200, 0)
_BG     = (30,  30,  30)


def _bar(frame, x, y, w, h, filled_ratio, color):
    cv2.rectangle(frame, (x, y), (x + w, y + h), (60, 60, 60), -1)
    if filled_ratio > 0:
        cv2.rectangle(frame, (x, y), (x + int(w * filled_ratio), y + h), color, -1)
    cv2.rectangle(frame, (x, y), (x + w, y + h), _GRAY, 1)


def _shadow_text(frame, text, pos, scale, color, thick):
    cv2.putText(frame, text, (pos[0]+2, pos[1]+2),
                cv2.FONT_HERSHEY_SIMPLEX, scale, _BLACK, thick + 2, cv2.LINE_AA)
    cv2.putText(frame, text, pos,
                cv2.FONT_HERSHEY_SIMPLEX, scale, color, thick, cv2.LINE_AA)


def _draw_capture_hud(frame, label, count, total_labels_done,
                      hand_detected, last_capture_ago, flash_alpha):
    h, w = frame.shape[:2]

    # Semi-transparent top panel
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, 110), _BG, -1)
    cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)

    # Target letter (large, left)
    _shadow_text(frame, label.upper(), (20, 90), 4.5, _WHITE, 7)

    # Progress of this letter
    ratio = min(count / TARGET_PER_LABEL, 1.0)
    bar_x, bar_y = 200, 20
    bar_w = w - bar_x - 20
    _bar(frame, bar_x, bar_y, bar_w, 22, ratio, _GREEN)
    prog_text = f"{count}/{TARGET_PER_LABEL}"
    cv2.putText(frame, prog_text, (bar_x + bar_w + 5, bar_y + 16),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, _WHITE, 1, cv2.LINE_AA)

    # Overall progress
    global_text = f"Letters completed: {total_labels_done}/{len(LABELS)}"
    cv2.putText(frame, global_text, (bar_x, bar_y + 45),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, _GRAY, 1, cv2.LINE_AA)

    # Hand status
    if hand_detected:
        interval_left = max(0, MIN_INTERVAL - last_capture_ago)
        if interval_left < 0.05:
            status = "CAPTURING..."
            status_color = _GREEN
        else:
            status = f"hand detected  ({interval_left:.1f}s)"
            status_color = _YELLOW
    else:
        status = "waiting for hand..."
        status_color = _RED
    cv2.putText(frame, status, (bar_x, bar_y + 68),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, status_color, 1, cv2.LINE_AA)

    # Green flash on capture
    if flash_alpha > 0:
        green_overlay = frame.copy()
        cv2.rectangle(green_overlay, (0, 0), (w, h), (0, 200, 0), -1)
        cv2.addWeighted(green_overlay, flash_alpha * 0.35, frame, 1 - flash_alpha * 0.35, 0, frame)

    # Instructions (bottom)
    instructions = "SPACE: manual capture  |  S: skip  |  Q: save and quit"
    cv2.putText(frame, instructions, (10, h - 12),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, _GRAY, 1, cv2.LINE_AA)

    # Variety tip
    if count < TARGET_PER_LABEL:
        tips = [
            "Vary the distance to the camera",
            "Try different hand angles",
            "Move the hand slightly between captures",
            "Different lighting helps the model",
        ]
        tip = tips[(count // 30) % len(tips)]
        cv2.putText(frame, f"Tip: {tip}", (10, h - 35),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, _CYAN, 1, cv2.LINE_AA)


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = os.path.join(OUTPUT_DIR, f"capture_{timestamp}.csv")

    # CSV columns: label + 63 raw landmark values (21 points x,y,z) + the frame
    # aspect ratio (width/height). MediaPipe normalizes x by width and y by
    # height, so the aspect is part of the data: loaders need it to un-stretch
    # the landmarks (see normalize_landmarks). Recording it per row makes each
    # CSV self-describing — no assumed camera geometry ever again.
    columns = ["label"] + [f"{c}{i}" for i in range(21) for c in ("x", "y", "z")] + ["aspect"]

    rows = []  # all captured samples

    print("Starting camera...")
    detector = Detector()
    print(f"Capturing {len(LABELS)} letters x {TARGET_PER_LABEL} samples = "
          f"{len(LABELS) * TARGET_PER_LABEL} total")
    print(f"Output: {output_path}\n")

    label_idx       = 0
    count           = 0          # samples captured for the current letter
    last_capture_t  = 0.0
    flash_alpha     = 0.0        # 0 to 1, for the green flash
    labels_done     = 0
    res_printed     = False      # camera resolution announced once

    while label_idx < len(LABELS):
        label = LABELS[label_idx]
        frame, lm_data = detector.get_frame()

        if frame is None:
            break

        if not res_printed:
            h, w = frame.shape[:2]
            print(f"Camera frame: {w}x{h} (aspect {w / h:.4f})")
            res_printed = True

        now              = time.time()
        hand_detected    = lm_data["num_hands"] >= 1 and lm_data["landmarks_hand1"] is not None
        since_last       = now - last_capture_t
        flash_alpha      = max(0.0, flash_alpha - 0.08)  # flash decay

        # Auto-capture: hand detected + interval elapsed
        do_capture = hand_detected and (since_last >= MIN_INTERVAL) and count < TARGET_PER_LABEL

        _draw_capture_hud(
            frame, label, count, labels_done,
            hand_detected, since_last, flash_alpha,
        )

        cv2.imshow("Data capture - Sign Language Translator", frame)

        key = cv2.waitKey(1) & 0xFF

        if key == ord("q"):
            print("\nCapture interrupted by the user.")
            break

        if key == ord("s"):
            print(f"  [{label}] skipped ({count} samples saved)")
            label_idx += 1
            count = 0
            continue

        if key == ord(" "):
            do_capture = hand_detected  # manual capture ignores the interval

        if do_capture:
            flat = lm_data["landmarks_hand1"]
            rows.append([label] + flat + [lm_data["frame_aspect"]])
            count       += 1
            last_capture_t = now
            flash_alpha    = 1.0
            print(f"  [{label}] {count}/{TARGET_PER_LABEL}", end="\r")

            if count >= TARGET_PER_LABEL:
                print(f"\n  [{label}] completed.")
                labels_done += 1
                label_idx   += 1
                count        = 0

    detector.release()
    cv2.destroyAllWindows()

    if not rows:
        print("No samples captured. Exiting without saving.")
        return

    # Save CSV
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(columns)
        writer.writerows(rows)

    # Summary
    from collections import Counter
    label_counts = Counter(r[0] for r in rows)
    print(f"\n{'=' * 55}")
    print(f"  Total captured: {len(rows):,} samples")
    print(f"  Letters with data: {len(label_counts)}")
    print(f"\n  Distribution:")
    for lbl in LABELS:
        cnt = label_counts.get(lbl, 0)
        bar = "#" * (cnt * 20 // TARGET_PER_LABEL)
        print(f"    {lbl:>6}: {cnt:>4}  [{bar:<20}]")
    print(f"{'=' * 55}")
    print(f"\nFile saved to: {output_path}")
    print("\nTrain now: python training/train_letters.py")
    print("(the trainer auto-discovers every CSV here, so just re-run this")
    print(" script on other days to add sessions -- no code to edit.)")


if __name__ == "__main__":
    main()
