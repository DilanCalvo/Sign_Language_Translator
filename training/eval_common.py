"""
Shared evaluation utilities for every model (letters, numbers, words).

Two independent layers so all three evaluators stay consistent and DRY:

  Reporting (model-agnostic)
      confusion matrix, per-class precision/recall/F1, top confusions, and a
      confidence-threshold sweep. Takes plain arrays (y_true, y_pred, y_conf)
      plus class names, so it works for any classifier. `evaluate.py` (words),
      `evaluate_letters.py` and `evaluate_numbers.py` all call report().

  Static-model evaluation (single-frame 63-value models: letters + numbers)
      Both share the exact same data format (one CSV per capture run: a `label`
      column + 63 raw landmark values) and pipeline (normalize_landmarks -> dense
      net). This module holds the two evaluation modes so each evaluator is a
      thin wrapper that only supplies its paths and its trainer's train_model:

        holdout  Evaluate the SAVED model on a stratified split reproduced with
                 the trainer's seed. Always available, but the split mixes each
                 capture session across train/val, so the number is OPTIMISTIC
                 (the same as what the trainer prints, plus a full breakdown).

        cv       Session-grouped K-fold, grouping by SOURCE FILE (each capture
                 run = one session). Whole files are held out, a fresh model is
                 trained per fold, so every sample is scored by a model that
                 never saw its capture session. This is the leak-free readout —
                 but it needs each class to appear in >= 2 files, otherwise a
                 class has no training data in the fold that holds out its only
                 file and scores ~0 (flagged with a warning).
"""

import glob
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.utils import normalize_landmarks

_THRESHOLDS = [0.50, 0.60, 0.70, 0.80, 0.90]


# ----------------------------------------------------------------------
# Reporting (shared by every model)
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
        print("\nNo confusions -- every sample classified into its own class.")
        return
    print("\nTop confusions (true -> predicted):")
    for cnt, t, p in pairs[:k]:
        print(f"  {cnt:>3}x   {t:<12} -> {p}")


def _expected_calibration_error(y_true, y_pred, y_conf, n_bins=10):
    """Standard 10-bin ECE (Guo et al., 2017): how far top-1 confidence is from
    actual accuracy, averaged over equal-width confidence bins and weighted by
    bin size. 0 = perfectly calibrated; a model that says "100%" while wrong
    drives this up regardless of overall accuracy, which is exactly the
    "sounds more confident than it should" symptom this metric is meant to
    catch (accuracy alone does not)."""
    y_true = np.asarray(y_true); y_pred = np.asarray(y_pred); y_conf = np.asarray(y_conf)
    correct = (y_pred == y_true).astype(np.float64)
    n = len(y_conf)
    if n == 0:
        return 0.0
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        in_bin = (y_conf > lo) & (y_conf <= hi) if lo > 0 else (y_conf >= lo) & (y_conf <= hi)
        if not in_bin.any():
            continue
        bin_acc = correct[in_bin].mean()
        bin_conf = y_conf[in_bin].mean()
        ece += in_bin.sum() / n * abs(bin_acc - bin_conf)
    return ece


def _print_threshold_sweep(y_true, y_pred, y_conf, threshold):
    print("\nConfidence-threshold sweep (coverage vs. accuracy among accepted):")
    print(f"  {'thresh':>7} {'coverage':>9} {'accuracy':>9}")
    y_true = np.asarray(y_true); y_pred = np.asarray(y_pred); y_conf = np.asarray(y_conf)
    for thr in _THRESHOLDS:
        accepted = y_conf >= thr
        cov = accepted.mean() if len(y_conf) else 0.0
        acc = (y_pred[accepted] == y_true[accepted]).mean() if accepted.sum() else 0.0
        mark = "  <- current" if threshold is not None and abs(thr - threshold) < 1e-9 else ""
        print(f"  {thr:>7.2f} {cov*100:>8.0f}% {acc*100:>8.0f}%{mark}")


def _save_png(cm, names, png_path):
    if not png_path:
        return
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        print("\n(matplotlib not available -- skipping confusion PNG)")
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
    os.makedirs(os.path.dirname(png_path), exist_ok=True)
    fig.savefig(png_path, dpi=120)
    print(f"\nConfusion matrix image -> {os.path.relpath(png_path)}")


def report(y_true, y_pred, y_conf, names, png_path=None, threshold=None):
    """Print the full diagnostic for one set of predictions and save a PNG.

    y_true / y_pred are integer class indices into `names`; y_conf is the top-1
    probability of each prediction. threshold (optional) marks the active
    acceptance threshold in the sweep.
    """
    cm = _confusion(y_true, y_pred, len(names))
    overall = np.mean(np.asarray(y_true) == np.asarray(y_pred)) if len(y_true) else 0.0
    ece = _expected_calibration_error(y_true, y_pred, y_conf)
    print(f"\nOverall accuracy: {overall*100:.1f}%   (n={len(y_true)})")
    print(f"Expected Calibration Error (ECE, 10 bins): {ece*100:.1f}%   "
          f"(0% = confidence always matches actual accuracy)")
    _print_confusion(cm, names)
    _print_per_class(cm, names)
    _print_top_confusions(cm, names)
    _print_threshold_sweep(y_true, y_pred, y_conf, threshold)
    _save_png(cm, names, png_path)


# ----------------------------------------------------------------------
# Static-model data loading (letters + numbers share this exactly)
# ----------------------------------------------------------------------

def load_static_csv_dir(data_dir, excluded=None):
    """Load every capture CSV in a directory as (X, y_str, groups).

    X          normalized (N, 63) float32 landmarks, aspect-corrected (see
               normalize_landmarks): rows with an "aspect" column use their
               recorded per-frame value; legacy rows without one fall back to
               config.LEGACY_CAPTURE_ASPECT. This loader is the ONLY parser of
               the capture-CSV format — trainers wrap it so the correction can
               never drift between training and evaluation.
    y_str      string labels (kept as strings so "0".."9" / "A".."Y" are stable).
    groups     the SOURCE FILE stem each row came from. One capture run writes
               one timestamped file, so the file is the natural "session" unit
               for leak-free grouping — no extra schema needed.
    """
    import pandas as pd

    from config import LEGACY_CAPTURE_ASPECT

    excluded = set(excluded or [])
    paths = sorted(glob.glob(os.path.join(data_dir, "*.csv")))
    if not paths:
        return (np.empty((0, 63), np.float32),
                np.empty((0,), object), np.empty((0,), object))

    X, y, groups = [], [], []
    for p in paths:
        df = pd.read_csv(p)
        stem = os.path.splitext(os.path.basename(p))[0]
        # Read COLUMN-WISE, not row-wise: df.iterrows() coerces every value in a
        # row to one dtype, turning an integer digit label into a float ("0" ->
        # "0.0") when the 63 landmark columns are floats. Pulling the label column
        # on its own preserves its dtype, so "0".."9" and "A".."Y" stay intact.
        labels = [str(v) for v in df["label"].tolist()]
        if "aspect" in df.columns:
            aspects = df["aspect"].to_numpy(dtype=np.float32)
            feature_df = df.drop(columns=["label", "aspect"])
        else:
            aspects = np.full(len(df), LEGACY_CAPTURE_ASPECT, dtype=np.float32)
            feature_df = df.drop(columns=["label"])
        raw = feature_df.to_numpy(dtype=np.float32)
        n_before = len(X)
        for label, row, aspect in zip(labels, raw, aspects):
            if label in excluded:
                continue
            X.append(normalize_landmarks(row, aspect=float(aspect)))
            y.append(label)
            groups.append(stem)
        print(f"  + {len(X) - n_before:,} samples from {os.path.basename(p)}")

    if not X:
        return (np.empty((0, 63), np.float32),
                np.empty((0,), object), np.empty((0,), object))
    return (np.stack(X).astype(np.float32),
            np.array(y, dtype=object),
            np.array(groups, dtype=object))


# ----------------------------------------------------------------------
# Static-model evaluation modes
# ----------------------------------------------------------------------

def run_static_holdout(model_path, labels_path, X, y_str, png_path,
                       threshold, seed=42, model_name="model"):
    """Evaluate the SAVED model on a stratified split reproduced with `seed`.

    Mirrors how the trainer splits its data (stratified 80/20, same seed), so the
    numbers match the trainer's own report plus a full confusion/threshold view.
    Optimistic: the split mixes capture sessions across train/val.
    """
    import tensorflow as tf
    from sklearn.model_selection import train_test_split

    if not (os.path.isfile(model_path) and os.path.isfile(labels_path)):
        print(f"[ERROR] No trained {model_name} model found "
              f"({model_path}). Train it first.")
        sys.exit(1)

    labels = json.load(open(labels_path, encoding="utf-8"))
    names = [labels[str(i)] for i in range(len(labels))]
    name_to_idx = {g: i for i, g in enumerate(names)}

    try:
        _, x_val, _, y_val_str = train_test_split(
            X, y_str, test_size=0.2, stratify=y_str, random_state=seed)
    except ValueError:
        _, x_val, _, y_val_str = train_test_split(
            X, y_str, test_size=0.2, random_state=seed)

    keep = np.array([g in name_to_idx for g in y_val_str])
    if not keep.all():
        dropped = sorted(set(y_val_str[~keep]))
        print(f"  [WARN] Val labels absent from the model: {dropped} -> dropped")
    x_val = x_val[keep]
    y_true = np.array([name_to_idx[g] for g in y_val_str[keep]])

    if len(x_val) == 0:
        print(f"[ERROR] No evaluable samples: the {model_name} data labels do not "
              f"match the model's classes ({names}). Re-train so the labels align.")
        sys.exit(1)

    print(f"Holdout eval of the SAVED {model_name} model on a stratified 20% "
          f"split (n={len(x_val)}).")
    print("  NOTE optimistic: capture sessions are mixed across train/val. For a")
    print("  leak-free read (whole files held out) use --cv once classes span")
    print("  >= 2 capture files.")

    model = tf.keras.models.load_model(model_path, compile=False)
    probs = model.predict(x_val, verbose=0)
    report(y_true, probs.argmax(axis=1), probs.max(axis=1), names,
           png_path=png_path, threshold=threshold)


def run_grouped_cv(X, y, groups, names, k, epochs, train_fn, png_path, threshold,
                   unit="capture file"):
    """Session-grouped K-fold cross-validation — the leak-free readout, shared by
    every model (letters, numbers AND words).

    Whole groups (a `unit`: one capture file for static models, one session_id
    for words) are held out per fold, a fresh model is trained on the rest and
    predicts the held-out group, so nothing is validated against its own session.

    X        indexable features (2-D for static models, 3-D for the word TCN).
    y        integer class labels (index-aligned to `names`).
    groups   the group id (session / file) each sample belongs to.
    names    class names, index i == class i.
    train_fn the model's OWN train_model(Xtr, ytr, Xva, yva, num_classes, epochs,
             verbose) -> fitted model, so the evaluated architecture/recipe are
             exactly what training uses (no drift). CV passes an empty val set.

    Callers do any model-specific warnings (e.g. a missing negative class) before
    calling this; the generic ">= 2 groups" and "single-group class" checks live
    here so all three models behave identically.
    """
    import tensorflow as tf
    from sklearn.model_selection import GroupKFold

    num_classes = len(names)
    n_groups = len(set(groups.tolist()))
    if n_groups < 2:
        print(f"[ERROR] Session-grouped CV needs >= 2 {unit}s; found {n_groups}. "
              f"Capture another session (a new run writes a new {unit}) first, or "
              "use the default holdout mode.")
        sys.exit(1)
    if k > n_groups:
        print(f"[WARN] Only {n_groups} {unit}s; reducing folds from {k} to "
              f"{n_groups} (one {unit} held out per fold).")
        k = n_groups

    # A class present in a single group cannot be validated cross-session: the
    # fold holding out its group has zero training data for it, so it scores ~0
    # and drags the average down. That is a data gap, not a model fault.
    groups_per_class = {}
    for cls, g in zip(y.tolist(), groups.tolist()):
        groups_per_class.setdefault(cls, set()).add(g)
    single = [names[c] for c, s in sorted(groups_per_class.items()) if len(s) < 2]
    if single:
        print(f"[WARN] Present in a single {unit} (can't be cross-validated; they "
              f"score ~0 in the fold that holds out their {unit}): "
              f"{', '.join(single)}.")
        print(f"       Re-capture these in another session for a fair number. If")
        print(f"       ALL classes are single-{unit}, prefer the default holdout mode.")

    print(f"\n{k}-fold {unit}-grouped cross-validation over {len(X)} samples, "
          f"{num_classes} classes ({epochs} epochs/fold). Training {k} models, "
          "please wait...\n")

    gkf = GroupKFold(n_splits=k)
    y = np.asarray(y)
    y_true_all, y_pred_all, y_conf_all = [], [], []
    empty_x = np.empty((0,) + X.shape[1:], np.float32)
    empty_y = np.empty((0,), np.int64)
    for fold, (tr, te) in enumerate(gkf.split(X, y, groups), 1):
        tf.keras.backend.clear_session()
        model = train_fn(X[tr], y[tr], empty_x, empty_y,
                         num_classes, epochs=epochs, verbose=0)
        probs = model.predict(X[te], verbose=0)
        y_true_all.extend(y[te].tolist())
        y_pred_all.extend(probs.argmax(axis=1).tolist())
        y_conf_all.extend(probs.max(axis=1).tolist())
        acc = float(np.mean(probs.argmax(axis=1) == y[te]))
        print(f"  fold {fold}/{k}:  held-out {len(te):>4}  acc {acc*100:5.1f}%")

    report(y_true_all, y_pred_all, y_conf_all, names,
           png_path=png_path, threshold=threshold)


def run_static_cv(X, y_str, groups, k, epochs, train_fn, png_path, threshold,
                  model_name="model"):
    """File-grouped CV for static (single-frame) models. Thin adapter that maps
    string labels to integer indices, then delegates to run_grouped_cv."""
    if len(X) == 0:
        print(f"[ERROR] No samples. Capture first for the {model_name} model.")
        sys.exit(1)
    names = sorted(set(y_str.tolist()))
    idx = {g: i for i, g in enumerate(names)}
    y = np.array([idx[g] for g in y_str], dtype=np.int64)
    run_grouped_cv(X, y, groups, names, k, epochs, train_fn, png_path, threshold,
                   unit="capture file")
