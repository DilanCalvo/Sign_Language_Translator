"""
Evaluate the letter (one-hand, static) model — "which letters get confused, and
how confident is it really?".

Two modes (see training/eval_common for the details shared with numbers):

  default (holdout)  Evaluate the SAVED model/model_one_hand.h5 on a stratified
                     20% split. Always available, but OPTIMISTIC: it mixes each
                     capture session across train/val.

  --cv K             File-grouped K-fold cross-validation (each capture CSV = one
                     session). Leak-free, but needs each letter to appear in
                     >= 2 capture files. The current letter dataset captured each
                     letter in a SINGLE file, so --cv will warn and score those
                     letters ~0 — capture a second session first for a real CV.

Usage:
  python training/evaluate_letters.py            # holdout eval of the saved model
  python training/evaluate_letters.py --cv 5     # file-grouped cross-validation
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import (
    MODEL_ONE_HAND_PATH         as _MODEL,
    LABELS_ONE_HAND_PATH        as _LABELS,
    LETTER_CONFIDENCE_THRESHOLD as _THRESHOLD,
)
from training import train_letters
from training.eval_common import load_static_csv_dir, run_static_holdout, run_static_cv

HERE      = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR  = os.path.join(HERE, "data", "real_capture", "letters")
PNG       = os.path.join(HERE, "model", "confusion_letters.png")


def main():
    ap = argparse.ArgumentParser(description="Evaluate the letter model.")
    ap.add_argument("--cv", type=int, default=0, metavar="K",
                    help="run K-fold file-grouped cross-validation instead of holdout")
    ap.add_argument("--epochs", type=int, default=min(train_letters.EPOCHS, 120),
                    help="epochs per fold in --cv mode (default: capped for speed)")
    args = ap.parse_args()

    # Letters exclude the same non-letter classes the trainer does (e.g. nothing).
    X, y_str, groups = load_static_csv_dir(
        DATA_DIR, excluded=train_letters.EXCLUDED_CLASSES)

    if args.cv and args.cv >= 2:
        run_static_cv(X, y_str, groups, args.cv, args.epochs,
                      train_letters.train_model, PNG, _THRESHOLD, model_name="letter")
    else:
        run_static_holdout(os.path.join(HERE, _MODEL), os.path.join(HERE, _LABELS),
                           X, y_str, PNG, _THRESHOLD, model_name="letter")


if __name__ == "__main__":
    main()
