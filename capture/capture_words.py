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
    data/real_capture/words/manifest.csv          # sample_id, gloss, subset, session_id

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
from datetime import datetime

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import (
    CAPTURE_TARGET_PER_WORD      as _CFG_TARGET,
    CAPTURE_PER_SESSION_PER_WORD as _CFG_SESSION_TARGET,
    CAPTURE_SEQ_LEN              as _SEQ_LEN,
    CAPTURE_OUTPUT_DIR          as _OUTPUT_DIR,
    WORD_BUFFER_FRAMES          as _MAX_REC,
    WORD_MIN_FRAMES             as _MIN_REC,
    WORD_NULL_LABEL             as _NEGATIVE_LABEL,
)
from src.detector import Detector
from src.utils import resample_sequence, build_word_features

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
    "good", "finished", "you", "me", 
    _NEGATIVE_LABEL,   # <- capture MANY varied non-signs for this one
]

TARGET_PER_WORD = _CFG_TARGET
# The negative class must cover many non-sign poses/motions, so capture more.
# Twice the per-word target (legacy ~40 + two more sessions of 14, see below).
WORD_TARGET = {_NEGATIVE_LABEL: 68}

# Per-session caps (takes of a word recorded in ONE run before auto-advancing).
# Mirrors WORD_TARGET: the negative class is captured at twice the rate, here too.
PER_SESSION_PER_WORD = _CFG_SESSION_TARGET
WORD_PER_SESSION = {_NEGATIVE_LABEL: 14}

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

# Manifest schema. session_id groups every take of one capture run so a
# session-aware split (GroupKFold in training) can keep whole sessions out of
# validation instead of leaking them across folds. subset is retained for
# backward compatibility with the current trainer.
MANIFEST_COLUMNS = ["sample_id", "gloss", "subset", "session_id"]

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


def _session_count() -> int:
    """Distinct capture sessions recorded in the manifest (multi-session is what
    makes the model generalize, so it is worth reporting)."""
    sessions = set()
    if os.path.isfile(MANIFEST):
        for r in csv.DictReader(open(MANIFEST, encoding="utf-8")):
            sessions.add(r.get("session_id") or "legacy")
    return len(sessions)


def _save_take(word, buf, counts, session_counts, new_rows, session_id):
    """Resample the recorded take to WORD_SEQ_LEN and persist it."""
    if len(buf) < _MIN_REC:
        print(f"  [{word}] take too short ({len(buf)} frames), discarded")
        return False
    n = counts.get(word, 0)                         # cumulative -> unique filename
    seq = resample_sequence(buf, _SEQ_LEN)        # (WORD_SEQ_LEN, 63)
    sample_id = f"{word}_{n:03d}"
    np.save(os.path.join(SEQ_DIR, f"{sample_id}.npy"), seq)
    # Spread val takes across the run; never make take 0 a val sample.
    subset = "val" if (n > 0 and n % VAL_EVERY == 0) else "train"
    new_rows.append([sample_id, word, subset, session_id])
    counts[word] = n + 1
    session_counts[word] = session_counts.get(word, 0) + 1
    print(f"  [{word}] saved take {n + 1}  ({subset})")
    return True


def _migrate_manifest_schema():
    """Add the session_id column to a manifest written before it existed.

    Rewrites the manifest in place, backfilling old rows with session_id
    'legacy' so a session-aware split treats all pre-existing takes as one
    unknown session rather than leaking them across folds. Reads and rewrites
    only the manifest CSV; never touches the .npy data. No-op when the schema
    is already current.
    """
    if not os.path.isfile(MANIFEST):
        return
    with open(MANIFEST, newline="", encoding="utf-8") as f:
        rows = list(csv.reader(f))
    if not rows or rows[0] == MANIFEST_COLUMNS:
        return
    if rows[0] == ["sample_id", "gloss", "subset"]:
        migrated = [MANIFEST_COLUMNS] + [r + ["legacy"] for r in rows[1:]]
        with open(MANIFEST, "w", newline="", encoding="utf-8") as f:
            csv.writer(f).writerows(migrated)
        print(f"  migrated manifest to {len(MANIFEST_COLUMNS)}-column schema "
              "(old takes tagged session_id=legacy)")


def _draw_hud(frame, word, s_count, s_target, t_count, t_target,
              recording, rec_len, flash, prompt):
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
    cv2.putText(frame, f"session {s_count}/{s_target}   total {t_count}/{t_target}",
                (20, 112), cv2.FONT_HERSHEY_SIMPLEX, 0.8, _GREEN, 2, cv2.LINE_AA)

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
    counts = _existing_counts()        # cumulative across all prior sessions
    session_counts = {}                # this run only (enforces the per-session cap)
    new_rows = []
    # One capture run = one session. Every take recorded now shares this id so a
    # session-aware split can hold this whole run out of validation.
    session_id = datetime.now().strftime("%Y%m%d_%H%M%S")

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
        session_target = WORD_PER_SESSION.get(word, PER_SESSION_PER_WORD)
        # Advance when the cumulative goal is met OR this session's cap is hit.
        # The latter is what forces multi-session capture: one sitting can never
        # fill the whole quota, so the user comes back another day.
        if (counts.get(word, 0) >= target
                or session_counts.get(word, 0) >= session_target):
            w_idx += 1
            continue

        frame, lm = detector.get_frame()
        if frame is None:
            break
        flash = max(0.0, flash - 0.06)
        by_side = lm["hands_by_side"]
        hand_ok = by_side["Left"] is not None or by_side["Right"] is not None
        pose_ok = lm["shoulder_l"] is not None and lm["shoulder_r"] is not None

        if recording and hand_ok:
            # Body-anchored two-hand feature — identical to what inference uses.
            buf.append(build_word_features(
                by_side["Left"], by_side["Right"],
                lm["shoulder_l"], lm["shoulder_r"]))
            if len(buf) >= _MAX_REC:
                recording = False
                if _save_take(word, list(buf), counts, session_counts, new_rows, session_id):
                    flash = 1.0
                buf.clear()

        prompt = (NEGATIVE_PROMPTS[counts.get(word, 0) % len(NEGATIVE_PROMPTS)]
                  if word == _NEGATIVE_LABEL else None)
        _draw_hud(frame, word, session_counts.get(word, 0), session_target,
                  counts.get(word, 0), target, recording, len(buf), flash, prompt)
        # Body-anchor indicator: warn if shoulders are not in frame, since then
        # the captured takes lose the body anchor (keep your torso visible).
        if not pose_ok:
            cv2.putText(frame, "! keep upper body in frame (no pose)",
                        (10, frame.shape[0] - 34),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, _RED, 2, cv2.LINE_AA)
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
                if _save_take(word, list(buf), counts, session_counts, new_rows, session_id):
                    flash = 1.0
                buf.clear()

    detector.release()
    cv2.destroyAllWindows()

    if not new_rows:
        print("No new takes captured.")
        return

    _migrate_manifest_schema()
    exists = os.path.isfile(MANIFEST)
    with open(MANIFEST, "a", newline="", encoding="utf-8") as f:
        wr = csv.writer(f)
        if not exists:
            wr.writerow(MANIFEST_COLUMNS)
        wr.writerows(new_rows)

    n_sessions = _session_count()   # manifest already includes this run's rows
    print(f"\nSaved {len(new_rows)} new takes (session {session_id}) -> {SEQ_DIR}")
    print(f"Manifest -> {MANIFEST}")
    if n_sessions < 3:
        print(f"\nSessions on disk: {n_sessions}. Capture this vocabulary again on a")
        print("DIFFERENT day/lighting before training — one session memorizes, "
              "several generalize.")
    print("\nNext: python training/train_words.py")


if __name__ == "__main__":
    main()
