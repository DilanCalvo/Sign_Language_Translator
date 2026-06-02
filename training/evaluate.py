"""
Comprehensive evaluation of the letter and word models.

For each model it evaluates on the 80/20 validation split (same SEED as
training) and reports:

  - Global and per-class accuracy (precision, recall, F1)
  - Normalized confusion matrix (PNG)
  - Top confusions (true -> predicted)
  - Confidence distribution splitting correct vs wrong (PNG)
  - Threshold sweep: coverage vs accuracy as confidence is tightened

At the end it prints a comparison table to decide where to invest effort
(capture more data, change the architecture, tune thresholds).

The DATA_CSVS lists must match EXACTLY the ones in the training scripts to
reproduce the same validation split.

Usage:
    python training/evaluate.py
"""

import json
import os
import sys

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import tensorflow as tf

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.utils import normalize_landmarks


# -------------------------------------------------------------------
# Configuration — mirror of the DATA_CSVS in the training scripts
# -------------------------------------------------------------------

LETTERS_CSVS = [
    "data/real_capture/letters/capture_20260520_180454_A-B-C.csv",
    "data/real_capture/letters/capture_20260520_143836_D-E-F.csv",
    "data/real_capture/letters/capture_20260520_145116_G-H-I-.csv",
    "data/real_capture/letters/capture_20260520_173739_I-K-L.csv",
    "data/real_capture/letters/capture_20260520_174943_M-N-O.csv",
    "data/real_capture/letters/capture_20260520_182414_P-Q-R.csv",
    "data/real_capture/letters/capture_20260520_184531_S-T-U.csv",
    "data/real_capture/letters/capture_20260520_204902_V-W-X.csv",
]

WORDS_CSVS = [
    "data/real_capture/words/words_20260520_211258_hola.csv",
    "data/real_capture/words/words_20260520_223012_adios.csv",
    "data/real_capture/words/words_20260520_222047_nada.csv",
]

# External test set: CSVs captured in a SEPARATE session that the model never
# touches during training. Measures real generalization (not inflated by
# overfitting to the val split of the same training set).
#
# Strict rules:
#   - These CSVs must NOT appear in train_words.py's DATA_CSVS.
#   - Capture in conditions DIFFERENT from training (other time, lighting,
#     distance or angle).
#   - 20-30 samples per class is enough.
EXTERNAL_TEST_CSVS_WORDS = [
    # "data/real_capture/words/words_<timestamp>_external.csv",
]

LETTERS_MODEL  = "model/model_one_hand.h5"
LETTERS_LABELS = "model/labels_one_hand.json"
WORDS_MODEL    = "model/model_words.h5"
WORDS_LABELS   = "model/labels_words.json"

OUTPUT_DIR     = "model"
EXCLUDED_LETTERS = {"nothing"}
SEED             = 42
TOP_CONFUSIONS   = 10
THRESHOLD_SWEEP  = [0.50, 0.60, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95]


# -------------------------------------------------------------------
# Validation loaders
# -------------------------------------------------------------------

def _load_csvs(paths):
    frames = []
    for p in paths:
        if p and os.path.isfile(p):
            frames.append(pd.read_csv(p))
        elif p:
            print(f"  [WARN] Missing: {p}")
    if not frames:
        return None
    return pd.concat(frames, ignore_index=True)


def _split_val(x, labels):
    """80/20 stratified with the same SEED the training scripts use."""
    try:
        _, x_val, _, y_val = train_test_split(
            x, labels, test_size=0.2, stratify=labels, random_state=SEED,
        )
    except ValueError:
        _, x_val, _, y_val = train_test_split(
            x, labels, test_size=0.2, random_state=SEED,
        )
    return x_val, y_val


def _load_letters_val():
    df = _load_csvs(LETTERS_CSVS)
    if df is None:
        return None, None
    df     = df[~df["label"].isin(EXCLUDED_LETTERS)].reset_index(drop=True)
    labels = np.array(df["label"].tolist())
    raw    = df.drop(columns=["label"]).to_numpy(dtype=np.float32)
    # Letters are stored raw: normalization is applied on load.
    x = np.stack([normalize_landmarks(row) for row in raw])
    return _split_val(x, labels)


def _load_words_val():
    df = _load_csvs(WORDS_CSVS)
    if df is None:
        return None, None
    labels = np.array(df["label"].tolist())
    # Words are already temporal features (mean+std, 126 cols).
    x = df.drop(columns=["label"]).to_numpy(dtype=np.float32)
    return _split_val(x, labels)


def _load_words_external():
    """Load the FULL external test set (no split). Returns None if absent."""
    if not EXTERNAL_TEST_CSVS_WORDS:
        return None, None
    df = _load_csvs(EXTERNAL_TEST_CSVS_WORDS)
    if df is None:
        return None, None
    labels = np.array(df["label"].tolist())
    x      = df.drop(columns=["label"]).to_numpy(dtype=np.float32)
    return x, labels


# -------------------------------------------------------------------
# Evaluation: run the model on the val set and gather metrics
# -------------------------------------------------------------------

def _evaluate_model(model, labels_dict, x_val, y_val_str, name):
    class_names = [labels_dict[i] for i in range(len(labels_dict))]
    name_to_idx = {v: k for k, v in labels_dict.items()}

    # Drop samples whose labels the model does not know (rare but possible if
    # the model was trained on a different subset).
    valid_mask = np.isin(y_val_str, list(name_to_idx.keys()))
    x_val      = x_val[valid_mask]
    y_val_str  = y_val_str[valid_mask]
    if len(x_val) == 0:
        print(f"  [ERROR] No valid samples after filtering by known labels.")
        return None

    y_val       = np.array([name_to_idx[s] for s in y_val_str], dtype=np.int32)
    probs       = model.predict(x_val, verbose=0)
    y_pred      = np.argmax(probs, axis=1)
    confidences = probs[np.arange(len(probs)), y_pred]

    return {
        "name":            name,
        "y_val":           y_val,
        "y_pred":          y_pred,
        "probs":           probs,
        "confidences":     confidences,
        "class_names":     class_names,
        "labels_present":  np.unique(np.concatenate([y_val, y_pred])),
        "accuracy":        accuracy_score(y_val, y_pred),
        "mean_confidence": float(confidences.mean()),
    }


# -------------------------------------------------------------------
# Console reports
# -------------------------------------------------------------------

def _print_per_class(result):
    print(f"\nPer-class report:")
    print(classification_report(
        result["y_val"], result["y_pred"],
        labels=result["labels_present"],
        target_names=[result["class_names"][i] for i in result["labels_present"]],
        digits=3, zero_division=0,
    ))


def _print_top_confusions(result):
    cm            = confusion_matrix(result["y_val"], result["y_pred"], labels=result["labels_present"])
    names_present = [result["class_names"][i] for i in result["labels_present"]]
    pairs = sorted(
        [(names_present[i], names_present[j], int(cm[i, j]))
         for i in range(len(names_present))
         for j in range(len(names_present))
         if i != j and cm[i, j] > 0],
        key=lambda t: t[2], reverse=True,
    )
    if pairs:
        print(f"Top {min(TOP_CONFUSIONS, len(pairs))} confusions (true -> predicted):")
        for true, pred, count in pairs[:TOP_CONFUSIONS]:
            print(f"  {true:>12s} -> {pred:<12s}  {count:>4d}")
    else:
        print("No confusions — perfect classification on val.")
    return cm, names_present


def _print_threshold_sweep(result):
    print(f"\nConfidence threshold sweep:")
    print(f"  {'threshold':>10s}  {'coverage':>10s}  {'accuracy':>10s}")
    n          = len(result["y_val"])
    is_correct = result["y_pred"] == result["y_val"]
    for t in THRESHOLD_SWEEP:
        keep = result["confidences"] >= t
        if keep.sum() == 0:
            print(f"  {t:>10.2f}  {'0.0%':>10s}  {'n/a':>10s}")
            continue
        coverage = keep.sum() / n
        accuracy = is_correct[keep].mean()
        print(f"  {t:>10.2f}  {coverage*100:>9.1f}%  {accuracy:>10.4f}")
    print("  (coverage = % of accepted predictions; accuracy = of the accepted)")


# -------------------------------------------------------------------
# Plots
# -------------------------------------------------------------------

def _plot_confusion(cm, class_names, output_path, title):
    n = len(class_names)
    fig, ax = plt.subplots(figsize=(max(6, n * 0.5), max(5, n * 0.5)))

    row_sums = cm.sum(axis=1, keepdims=True)
    cm_norm  = np.divide(cm, row_sums, out=np.zeros_like(cm, dtype=float), where=row_sums > 0)

    im = ax.imshow(cm_norm, interpolation="nearest", cmap="Blues", vmin=0, vmax=1)
    ax.figure.colorbar(im, ax=ax)
    ax.set(
        xticks=np.arange(n), yticks=np.arange(n),
        xticklabels=class_names, yticklabels=class_names,
        ylabel="True label", xlabel="Predicted label",
        title=title,
    )
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    for i in range(n):
        for j in range(n):
            v = cm_norm[i, j]
            if v > 0.01:
                ax.text(j, i, f"{v:.2f}", ha="center", va="center",
                        color="white" if v > 0.5 else "black", fontsize=7)
    plt.tight_layout()
    plt.savefig(output_path, dpi=120)
    plt.close(fig)


def _plot_confidence_analysis(result, output_path):
    """Two subplots: confidence histogram + threshold curve."""
    is_correct  = result["y_pred"] == result["y_val"]
    confidences = result["confidences"]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # 1) Histogram — ideal: greens on the right, reds on the left, no overlap.
    bins = np.linspace(0, 1, 21)
    ax1.hist(confidences[is_correct],  bins=bins, color="#2c8a2c", alpha=0.75, label="correct", edgecolor="black", linewidth=0.3)
    ax1.hist(confidences[~is_correct], bins=bins, color="#c83232", alpha=0.75, label="wrong",   edgecolor="black", linewidth=0.3)
    ax1.set_xlabel("Confidence")
    ax1.set_ylabel("Number of predictions")
    ax1.set_title(f"Confidence distribution — {result['name']}")
    ax1.legend()
    ax1.grid(alpha=0.3)

    # 2) Threshold sweep — coverage drops, accuracy ideally rises.
    thresholds = np.linspace(0.5, 1.0, 26)
    coverages, accuracies = [], []
    for t in thresholds:
        keep = confidences >= t
        if keep.sum() == 0:
            coverages.append(0.0)
            accuracies.append(np.nan)
        else:
            coverages.append(keep.sum() / len(confidences))
            accuracies.append(is_correct[keep].mean())

    ax2.plot(thresholds, coverages,  "b-", linewidth=2, label="Coverage (% accepted)")
    ax2.plot(thresholds, accuracies, "g-", linewidth=2, label="Accuracy (of accepted)")
    ax2.axhline(y=1.0, color="gray", linestyle="--", linewidth=0.5)
    ax2.set_xlabel("Confidence threshold")
    ax2.set_ylabel("Proportion")
    ax2.set_title(f"Threshold trade-off — {result['name']}")
    ax2.grid(alpha=0.3)
    ax2.legend(loc="lower left")
    ax2.set_ylim(0, 1.05)

    plt.tight_layout()
    plt.savefig(output_path, dpi=120)
    plt.close(fig)


# -------------------------------------------------------------------
# Orchestration
# -------------------------------------------------------------------

def _evaluate(model_path, labels_path, csvs_loader, name, external_loader=None):
    if not (os.path.isfile(model_path) and os.path.isfile(labels_path)):
        print(f"\n[SKIP] Model '{name}' not found at {model_path}")
        return None

    print(f"\n{'=' * 64}")
    print(f"  EVALUATION — {name.upper()}")
    print(f"{'=' * 64}")

    with open(labels_path, "r", encoding="utf-8") as f:
        labels_dict = {int(k): v for k, v in json.load(f).items()}

    print(f"Loading model: {model_path}")
    model = tf.keras.models.load_model(model_path, compile=False)

    x_val, y_val_str = csvs_loader()
    if x_val is None:
        print(f"  [ERROR] Could not load validation data for '{name}'")
        return None

    print(f"  {len(x_val):,} validation samples over {len(labels_dict)} classes\n")

    result = _evaluate_model(model, labels_dict, x_val, y_val_str, name)
    if result is None:
        return None

    print(f"  Global accuracy:  {result['accuracy']:.4f}  ({result['accuracy']*100:.2f} %)")
    print(f"  Mean confidence:  {result['mean_confidence']:.4f}")

    _print_per_class(result)
    cm, names_present = _print_top_confusions(result)
    _print_threshold_sweep(result)

    confusion_png = f"{OUTPUT_DIR}/confusion_{name}.png"
    analysis_png  = f"{OUTPUT_DIR}/analysis_{name}.png"
    _plot_confusion(cm, names_present, confusion_png, f"Confusion matrix — {name}")
    _plot_confidence_analysis(result, analysis_png)
    print(f"\nPNGs: {confusion_png}, {analysis_png}")

    # External test-set evaluation (if configured).
    if external_loader is not None:
        x_ext, y_ext_str = external_loader()
        if x_ext is not None:
            _evaluate_external(model, labels_dict, x_ext, y_ext_str, name, result["accuracy"])

    return result


def _evaluate_external(model, labels_dict, x_ext, y_ext_str, name, val_accuracy):
    """Evaluate the model on an external test set and compare with val."""
    print(f"\n{'-' * 64}")
    print(f"  EXTERNAL TEST — {name.upper()}  ({len(x_ext)} new samples)")
    print(f"{'-' * 64}")

    ext = _evaluate_model(model, labels_dict, x_ext, y_ext_str, f"{name}_external")
    if ext is None:
        return

    print(f"  Accuracy:         {ext['accuracy']:.4f}  ({ext['accuracy']*100:.2f} %)")
    print(f"  Mean confidence:  {ext['mean_confidence']:.4f}")

    _print_per_class(ext)
    _print_top_confusions(ext)

    gap = val_accuracy - ext["accuracy"]
    print(f"\n  val - external gap: {gap*100:+.2f} percentage points")
    if abs(gap) < 0.05:
        print("    -> Healthy generalization. The val set was representative.")
    elif gap > 0.10:
        print("    -> Significant overfit. Consider more diverse data or more augmentation.")
    elif gap > 0.05:
        print("    -> Moderate overfit. Acceptable for a small dataset.")
    else:
        print("    -> External beat val. Check that the samples are labeled correctly.")

    analysis_png = f"{OUTPUT_DIR}/analysis_{name}_external.png"
    _plot_confidence_analysis(ext, analysis_png)
    print(f"  PNG: {analysis_png}")


def _print_comparison(letters, words):
    print(f"\n{'=' * 64}")
    print("  MODEL COMPARISON")
    print(f"{'=' * 64}")
    if letters is None and words is None:
        print("  No models to compare.")
        return

    rows = [
        ("Number of classes", letters and len(letters["class_names"]), words and len(words["class_names"])),
        ("Val samples",       letters and len(letters["y_val"]),       words and len(words["y_val"])),
        ("Global accuracy",   letters and letters["accuracy"],         words and words["accuracy"]),
        ("Mean confidence",   letters and letters["mean_confidence"],  words and words["mean_confidence"]),
    ]
    print(f"  {'metric':<22s}  {'letters':>14s}  {'words':>14s}")
    print(f"  {'-' * 54}")
    for label, lv, wv in rows:
        ls = f"{lv:.4f}" if isinstance(lv, float) else (f"{lv:,}" if lv is not None else "—")
        ws = f"{wv:.4f}" if isinstance(wv, float) else (f"{wv:,}" if wv is not None else "—")
        print(f"  {label:<22s}  {ls:>14s}  {ws:>14s}")

    # Decision heuristics to guide the user.
    print(f"\n  Reading:")
    if letters and words:
        if letters["accuracy"] - words["accuracy"] > 0.15:
            print("    - The letter model is noticeably more reliable.")
            print("      To improve words: capture more samples or change the architecture.")
        if words["accuracy"] < 0.80:
            print("    - Word accuracy below 80%: insufficient dataset")
            print("      or mean+std architecture too weak for the added classes.")
        if words["mean_confidence"] < 0.70:
            print("    - Low mean word confidence: the current threshold is probably")
            print("      rejecting many valid predictions. Check the sweep.")


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    letters = _evaluate(LETTERS_MODEL, LETTERS_LABELS, _load_letters_val, "letters")
    words   = _evaluate(WORDS_MODEL,   WORDS_LABELS,   _load_words_val,   "words",
                        external_loader=_load_words_external)
    _print_comparison(letters, words)
    print()


if __name__ == "__main__":
    main()
