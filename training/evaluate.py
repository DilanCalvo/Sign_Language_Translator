"""
Evaluate the word (TCN) model — the diagnostic tool for "which signs does it
confuse, and how confident is it really?".

Two modes:

  default (holdout)  Evaluate the SAVED model/model_words.h5 on the val split
                     from the manifest. Fast, but only as trustworthy as the val
                     set — which is often tiny (a couple of takes per class), so
                     the numbers are noisy.

  --cv K (k-fold)    Session-grouped K-fold CROSS-VALIDATION over ALL takes
                     (GroupKFold by session_id). Each fold holds out WHOLE
                     capture sessions, trains a fresh model and predicts them, so
                     every take is tested on a model that never saw its session.
                     This is the leak-free readout. Needs >= 2 sessions. Slower
                     (it trains K models).

Both print: a confusion matrix, per-class precision/recall/F1, the top
confusions (which sign gets mistaken for which), and a confidence-threshold
sweep (coverage vs. accuracy) to help tune WORD_CONFIDENCE_THRESHOLD. A PNG of
the confusion matrix is saved to model/confusion_words.png.

Usage:
  python training/evaluate.py                  # quick holdout eval of saved model
  python training/evaluate.py --cv 5           # 5-fold cross-validation (recommended)
  python training/evaluate.py --cv 5 --epochs 80   # faster folds
"""

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import (
    MODEL_WORDS_PATH  as _MODEL_WORDS,
    LABELS_WORDS_PATH as _LABELS_WORDS,
    WORD_CONFIDENCE_THRESHOLD as _WORD_THRESHOLD,
)
from training.train_words import (
    HERE, MANIFEST, EPOCHS,
    load_manifest, load_samples, session_holdout, train_model,
)
from training.eval_common import report, run_grouped_cv

# Where the confusion-matrix image is written for the word model.
_PNG = os.path.join(HERE, "model", "confusion_words.png")


# ----------------------------------------------------------------------
# Modes
# ----------------------------------------------------------------------

def evaluate_holdout():
    import tensorflow as tf

    if not os.path.isfile(_MODEL_WORDS) or not os.path.isfile(_LABELS_WORDS):
        print("[ERROR] No trained word model found. Train first: "
              "python training/train_words.py")
        sys.exit(1)

    labels = json.load(open(_LABELS_WORDS, encoding="utf-8"))
    names = [labels[str(i)] for i in range(len(labels))]
    label_to_idx = {g: i for i, g in enumerate(names)}

    rows, _, _ = load_manifest()
    X, y, groups = load_samples(rows, label_to_idx)
    # Reproduce the SAME session holdout the trainer used (deterministic seed),
    # so we evaluate on exactly the sessions the saved model never trained on.
    _, va_idx = session_holdout(y, groups)
    Xva, yva = X[va_idx], y[va_idx]
    if len(Xva) == 0:
        print("[ERROR] No held-out session to evaluate on (only one capture "
              "session exists). Capture a second session, or run session-grouped "
              "cross-validation: python training/evaluate.py --cv 5")
        sys.exit(1)

    print(f"Holdout eval of the SAVED model on held-out sessions (n={len(Xva)}).")
    print("  For a per-class breakdown across all sessions use --cv.")
    model = tf.keras.models.load_model(_MODEL_WORDS, compile=False)
    probs = model.predict(Xva, verbose=0)
    y_pred = probs.argmax(axis=1)
    y_conf = probs.max(axis=1)
    report(yva, y_pred, y_conf, names, png_path=_PNG, threshold=_WORD_THRESHOLD)


def evaluate_cv(k, epochs):
    rows, glosses, label_to_idx = load_manifest()
    X, y, groups = load_samples(rows, label_to_idx)   # ALL takes + session ids
    if len(X) == 0:
        print("[ERROR] No samples. Capture first: python capture/capture_words.py")
        sys.exit(1)

    # Word-specific warning; the generic group checks live in run_grouped_cv.
    if "nothing" not in glosses:
        print("[WARN] No 'nothing' class — the live model will never stay silent. "
              "This evaluation still runs, but capture 'nothing' for real use.")

    run_grouped_cv(X, y, groups, glosses, k, epochs, train_model,
                   _PNG, _WORD_THRESHOLD, unit="session")


def main():
    ap = argparse.ArgumentParser(description="Evaluate the word TCN model.")
    ap.add_argument("--cv", type=int, default=0, metavar="K",
                    help="run K-fold cross-validation instead of holdout eval")
    ap.add_argument("--epochs", type=int, default=min(EPOCHS, 120),
                    help="epochs per fold in --cv mode (default: capped for speed)")
    args = ap.parse_args()

    if not os.path.isfile(MANIFEST):
        print("[ERROR] No manifest. Capture first: python capture/capture_words.py")
        sys.exit(1)

    if args.cv and args.cv >= 2:
        evaluate_cv(args.cv, args.epochs)
    else:
        evaluate_holdout()


if __name__ == "__main__":
    main()
