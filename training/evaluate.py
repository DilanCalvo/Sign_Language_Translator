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

import numpy as np

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

_THRESHOLDS = [0.50, 0.60, 0.70, 0.80, 0.90]


# ----------------------------------------------------------------------
# Reporting (shared by both modes)
# ----------------------------------------------------------------------

def _confusion(y_true, y_pred, n):
    cm = np.zeros((n, n), dtype=int)
    for t, p in zip(y_true, y_pred):
        cm[t, p] += 1
    return cm


def _print_confusion(cm, names):
    w = max(7, max(len(g) for g in names))
    head = " " * (w + 2) + "".join(f"{g[:6]:>7}" for g in names)
    print("\nConfusion matrix (rows = true, cols = predicted):")
    print(head)
    for i, g in enumerate(names):
        row = "".join(f"{cm[i, j]:>7}" for j in range(len(names)))
        acc = cm[i, i] / cm[i].sum() * 100 if cm[i].sum() else 0.0
        print(f"{g[:w]:<{w}}  {row}   {acc:5.0f}%")


def _print_per_class(cm, names):
    print("\nPer-class precision / recall / F1:")
    print(f"  {'class':<12} {'prec':>6} {'recall':>7} {'f1':>6} {'support':>8}")
    for i, g in enumerate(names):
        tp = cm[i, i]
        support = cm[i].sum()
        pred_pos = cm[:, i].sum()
        prec = tp / pred_pos if pred_pos else 0.0
        rec  = tp / support if support else 0.0
        f1   = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
        print(f"  {g:<12} {prec:>6.2f} {rec:>7.2f} {f1:>6.2f} {support:>8}")


def _print_top_confusions(cm, names, k=8):
    pairs = []
    for i in range(len(names)):
        for j in range(len(names)):
            if i != j and cm[i, j] > 0:
                pairs.append((cm[i, j], names[i], names[j]))
    pairs.sort(reverse=True)
    if not pairs:
        print("\nNo confusions -- every take classified into its own class.")
        return
    print("\nTop confusions (true -> predicted):")
    for cnt, t, p in pairs[:k]:
        print(f"  {cnt:>3}x   {t:<12} -> {p}")


def _print_threshold_sweep(y_true, y_pred, y_conf):
    print("\nConfidence-threshold sweep (how WORD_CONFIDENCE_THRESHOLD trades off):")
    print(f"  {'thresh':>7} {'coverage':>9} {'accuracy':>9}   (accuracy = among accepted)")
    y_true = np.asarray(y_true); y_pred = np.asarray(y_pred); y_conf = np.asarray(y_conf)
    for thr in _THRESHOLDS:
        accepted = y_conf >= thr
        cov = accepted.mean() if len(y_conf) else 0.0
        if accepted.sum():
            acc = (y_pred[accepted] == y_true[accepted]).mean()
        else:
            acc = 0.0
        mark = "  <- current" if abs(thr - _WORD_THRESHOLD) < 1e-9 else ""
        print(f"  {thr:>7.2f} {cov*100:>8.0f}% {acc*100:>8.0f}%{mark}")


def _save_png(cm, names):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        print("\n(matplotlib not available — skipping confusion PNG)")
        return
    cmn = cm / np.clip(cm.sum(axis=1, keepdims=True), 1, None)
    fig, ax = plt.subplots(figsize=(1.1 * len(names) + 2, 1.1 * len(names) + 2))
    im = ax.imshow(cmn, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(len(names))); ax.set_xticklabels(names, rotation=45, ha="right")
    ax.set_yticks(range(len(names))); ax.set_yticklabels(names)
    ax.set_xlabel("predicted"); ax.set_ylabel("true")
    for i in range(len(names)):
        for j in range(len(names)):
            if cm[i, j]:
                ax.text(j, i, cm[i, j], ha="center", va="center",
                        color="white" if cmn[i, j] > 0.5 else "black", fontsize=8)
    fig.colorbar(im, fraction=0.046, pad=0.04)
    fig.tight_layout()
    out = os.path.join(HERE, "model", "confusion_words.png")
    fig.savefig(out, dpi=120)
    print(f"\nConfusion matrix image -> model/confusion_words.png")


def _report(y_true, y_pred, y_conf, names):
    cm = _confusion(y_true, y_pred, len(names))
    overall = np.mean(np.asarray(y_true) == np.asarray(y_pred)) if len(y_true) else 0.0
    print(f"\nOverall accuracy: {overall*100:.1f}%   (n={len(y_true)})")
    _print_confusion(cm, names)
    _print_per_class(cm, names)
    _print_top_confusions(cm, names)
    _print_threshold_sweep(y_true, y_pred, y_conf)
    _save_png(cm, names)


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
    _report(yva, y_pred, y_conf, names)


def evaluate_cv(k, epochs):
    import tensorflow as tf
    from sklearn.model_selection import GroupKFold

    rows, glosses, label_to_idx = load_manifest()
    num_classes = len(glosses)
    X, y, groups = load_samples(rows, label_to_idx)   # ALL takes + session ids
    if len(X) == 0:
        print("[ERROR] No samples. Capture first: python capture/capture_words.py")
        sys.exit(1)

    n_sessions = len(set(groups.tolist()))
    if n_sessions < 2:
        print(f"[ERROR] Session-grouped CV needs >= 2 capture sessions; found "
              f"{n_sessions}. Capture another session (different day) first.")
        sys.exit(1)
    if k > n_sessions:
        print(f"[WARN] Only {n_sessions} sessions; reducing folds from {k} to "
              f"{n_sessions} (one session held out per fold).")
        k = n_sessions

    # A class captured in only ONE session cannot be validated cross-session: in
    # the fold that holds out its session it has zero training data, so it scores
    # ~0 there and drags the average down. That is a data gap, not a model fault.
    sess_per_class = {}
    for cls, g in zip(y.tolist(), groups.tolist()):
        sess_per_class.setdefault(cls, set()).add(g)
    single = [glosses[c] for c, s in sorted(sess_per_class.items()) if len(s) < 2]
    if single:
        print(f"[WARN] Captured in a single session (can't be cross-validated, will "
              f"score low in the fold that holds out their session): {', '.join(single)}.")
        print("       Capture these in another session for a fair number.")

    if "nothing" not in glosses:
        print("[WARN] No 'nothing' class — the live model will never stay silent. "
              "This evaluation still runs, but capture 'nothing' for real use.")

    print(f"\n{k}-fold session-grouped cross-validation over {len(X)} takes, "
          f"{num_classes} classes ({epochs} epochs/fold). Training {k} models, "
          "please wait...\n")

    gkf = GroupKFold(n_splits=k)
    y_true_all, y_pred_all, y_conf_all = [], [], []
    for fold, (tr, te) in enumerate(gkf.split(X, y, groups), 1):
        tf.keras.backend.clear_session()
        model = train_model(X[tr], y[tr],
                            np.empty((0,) + X.shape[1:], np.float32),
                            np.empty((0,), np.int64),
                            num_classes, epochs=epochs, verbose=0)
        probs = model.predict(X[te], verbose=0)
        y_true_all.extend(y[te].tolist())
        y_pred_all.extend(probs.argmax(axis=1).tolist())
        y_conf_all.extend(probs.max(axis=1).tolist())
        acc = np.mean(probs.argmax(axis=1) == y[te])
        print(f"  fold {fold}/{k}:  held-out {len(te)}  acc {acc*100:5.1f}%")

    _report(y_true_all, y_pred_all, y_conf_all, glosses)


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
