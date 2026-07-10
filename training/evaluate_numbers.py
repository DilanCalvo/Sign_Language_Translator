"""
Evaluate the numbers (digit 0-9, static) model — "which digits get confused, and
how confident is it really?".

Two modes (see training/eval_common for the details shared with letters):

  default (holdout)  Evaluate the SAVED model/model_numbers.h5 on a stratified
                     20% split. Always available, but OPTIMISTIC: it mixes each
                     capture session across train/val.

  --cv K             File-grouped K-fold cross-validation (each capture CSV = one
                     session). Leak-free — this is the honest readout. Needs each
                     digit to appear in >= 2 capture files, so capture across
                     several sessions (a new run writes a new file) first.

Usage:
  python training/evaluate_numbers.py            # holdout eval of the saved model
  python training/evaluate_numbers.py --cv 5     # file-grouped cross-validation
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import (
    MODEL_NUMBERS_PATH          as _MODEL,
    LABELS_NUMBERS_PATH         as _LABELS,
    NUMBER_CONFIDENCE_THRESHOLD as _THRESHOLD,
)
from training import train_numbers
from training.eval_common import load_static_csv_dir, run_static_holdout, run_static_cv

HERE      = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR  = os.path.join(HERE, train_numbers.DATA_DIR)
PNG       = os.path.join(HERE, "model", "confusion_numbers.png")


def main():
    ap = argparse.ArgumentParser(description="Evaluate the numbers model.")
    ap.add_argument("--cv", type=int, default=0, metavar="K",
                    help="run K-fold file-grouped cross-validation instead of holdout")
    ap.add_argument("--epochs", type=int, default=min(train_numbers.EPOCHS, 120),
                    help="epochs per fold in --cv mode (default: capped for speed)")
    args = ap.parse_args()

    X, y_str, groups = load_static_csv_dir(DATA_DIR)

    if args.cv and args.cv >= 2:
        run_static_cv(X, y_str, groups, args.cv, args.epochs,
                      train_numbers.train_model, PNG, _THRESHOLD, model_name="numbers")
    else:
        run_static_holdout(os.path.join(HERE, _MODEL), os.path.join(HERE, _LABELS),
                           X, y_str, PNG, _THRESHOLD, model_name="numbers")


if __name__ == "__main__":
    main()
