"""
Word / moving-sign model training — temporal sequence model (TCN).

Each sample is a (WORD_SEQ_LEN, WORD_FEATURE_DIM) sequence of body-anchored,
two-hand features captured by capture/capture_words.py (see config for the
layout: both handshapes + each wrist's position relative to the shoulders).
Unlike the old mean+std model (which was order-blind and collapsed as the
vocabulary grew), this is a TEMPORAL model: stacked dilated 1D convolutions read
the movement OVER TIME, so it can tell similar signs apart by HOW the hands move
and WHERE they are on the body, not just the average pose.

The architecture (Conv1D + global pooling, ~80K params) was validated earlier on
sequence data and is trained from scratch on your own captures — no external
dataset, no domain gap. It stays portable to the web (tensorflowjs) for the
future browser version.

Input:
    data/real_capture/words/seq/*.npy   (from capture_words.py)
    data/real_capture/words/manifest.csv

Output:
    model/model_words.h5
    model/labels_words.json

Usage:
    1. Capture data (ideally across several days):  python capture/capture_words.py
    2. Train:                                        python training/train_words.py
"""

import csv
import json
import os
import sys
from collections import Counter

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import (
    WORD_SEQ_LEN       as T,
    WORD_FEATURE_DIM   as FEATURE_DIM,
    CAPTURE_OUTPUT_DIR as _DATA_DIR,
    MODEL_WORDS_PATH   as MODEL_OUT,
    LABELS_WORDS_PATH  as LABELS_OUT,
)
from src.utils import mirror_word_sequence, resample_sequence
from training.run_log import write_run_metadata

HERE     = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(HERE, _DATA_DIR)
SEQ_DIR  = os.path.join(DATA_DIR, "seq")
MANIFEST = os.path.join(DATA_DIR, "manifest.csv")

BATCH_SIZE  = 16
EPOCHS      = 200
LR          = 1e-3
NOISE_STD   = 0.02
SEED        = 42


def load_manifest():
    """Read the manifest and the sorted class list. Shared by train + evaluate."""
    rows = list(csv.DictReader(open(MANIFEST, encoding="utf-8")))
    glosses = sorted({r["gloss"] for r in rows})
    label_to_idx = {g: i for i, g in enumerate(glosses)}
    return rows, glosses, label_to_idx


def load_samples(rows, label_to_idx):
    """Load (X, y, groups) for ALL takes in the manifest.

    groups holds each take's session_id (the capture run it came from). It is
    what makes validation honest: whole sessions are held out instead of random
    takes, so the model is never validated against the same day/lighting it
    trained on. Takes written before session_id existed are tagged "legacy".
    """
    X, y, groups = [], [], []
    for r in rows:
        path = os.path.join(SEQ_DIR, f"{r['sample_id']}.npy")
        if not os.path.isfile(path):
            continue
        seq = np.load(path)
        if seq.shape != (T, FEATURE_DIM):
            print(f"  [WARN] {r['sample_id']} has shape {seq.shape}, expected "
                  f"{(T, FEATURE_DIM)} — skipped (re-capture after changing WORD_SEQ_LEN).")
            continue
        X.append(seq)
        y.append(label_to_idx[r["gloss"]])
        groups.append(r.get("session_id") or "legacy")
    if not X:
        return (np.empty((0, T, FEATURE_DIM), np.float32),
                np.empty((0,), np.int64), np.empty((0,), object))
    return (np.stack(X).astype(np.float32),
            np.array(y, np.int64),
            np.array(groups, dtype=object))


def session_holdout(y, groups, val_fraction=0.2, seed=SEED):
    """Train/val indices that hold WHOLE sessions out for validation.

    Whole capture sessions go to validation (never split within a session) so
    there is no train/val leakage. Returns an empty val set when fewer than two
    sessions exist: an honest holdout is then impossible, so the caller trains on
    everything and reports that the metric would be optimistic.
    """
    groups = np.asarray(groups, dtype=object)
    sessions = sorted(set(groups.tolist()))
    if len(sessions) < 2:
        return np.arange(len(y)), np.empty((0,), dtype=int)

    rng = np.random.default_rng(seed)
    rng.shuffle(sessions)
    val_sessions, n_val, target = set(), 0, val_fraction * len(y)
    for s in sessions:
        if n_val >= target:
            break
        val_sessions.add(s)
        n_val += int(np.sum(groups == s))
    if len(val_sessions) == len(sessions):       # never leave train empty
        val_sessions.discard(sessions[-1])

    val_mask = np.array([g in val_sessions for g in groups])
    return np.where(~val_mask)[0], np.where(val_mask)[0]


_HAND_BLOCK = FEATURE_DIM // 2   # 65: one hand's [63 shape + 2 position] block


def _time_warp(seq, gamma):
    """Speed change: same length, non-linear time remap. Teaches the model the
    SAME sign performed faster/slower (attacks the clean-capture vs live-pace gap)."""
    u = np.linspace(0.0, 1.0, T) ** gamma
    idx = (u * (T - 1)).round().astype(int)
    return seq[idx]


def _frame_dropout(seq, p, rng):
    """Zero whole frames at random. MediaPipe intermittently loses the hands in
    live use; an all-zero frame is exactly the 'hands absent' pattern the feature
    builder emits, so this teaches the model to ride through dropped frames
    instead of treating clean capture as the only reality."""
    out = seq.copy()
    out[rng.random(len(seq)) < p] = 0.0
    return out


def _hand_dropout(seq, rng):
    """Zero one hand's block over a contiguous span — a hand briefly leaving the
    frame. Span-limited (20-50% of the take, not all of it) so the sign's
    identity survives on two-handed signs while still teaching single-hand
    robustness, the other common live detection failure."""
    out = seq.copy()
    side = rng.integers(2)                       # 0 = left block, 1 = right block
    lo = side * _HAND_BLOCK
    span = max(1, int(len(seq) * rng.uniform(0.2, 0.5)))
    start = rng.integers(0, len(seq) - span + 1)
    out[start:start + span, lo:lo + _HAND_BLOCK] = 0.0
    return out


def _temporal_crop(seq, rng):
    """Drop 10-25% of the frames from the start OR the end, then resample back
    to T — a misaligned live window (the rolling buffer slides mid-sign, unlike
    the aligned SPACE->stop capture takes).

    NOT APPLIED. Measured 2026-07-09 on 612 takes / 6 sessions: adding this as a
    9th offline variant dropped session-grouped CV from 94.1% to 93.0% and
    `nothing` recall from 0.76 to 0.74 — every multi-class fold got ~1.2pp worse.
    The aligned-takes CV can't see any live-misalignment benefit, but it does
    price the cost, and the cost was real. Kept (unused) so the experiment isn't
    blindly repeated; revisit only with an eval that scores misaligned windows."""
    frac = rng.uniform(0.10, 0.25)
    cut = max(1, int(len(seq) * frac))
    cropped = seq[cut:] if rng.integers(2) else seq[:-cut]
    return resample_sequence(cropped, T)


def _offline_augment(X, y):
    """Enlarge the tiny dataset with structural variants applied once before
    training: mirror, time-warp (pace) and detector-failure (dropped frames /
    a hand leaving the frame). Per-batch jitter is added separately in tf.data.
    (_temporal_crop was tried and measured as a net regression — see its
    docstring — so it is deliberately not in this list.)"""
    rng = np.random.default_rng(SEED)
    variants = [X, np.stack([mirror_word_sequence(s) for s in X])]
    for g in (0.7, 1.4):
        variants.append(np.stack([_time_warp(s, g) for s in X]))
        variants.append(np.stack([_time_warp(mirror_word_sequence(s), g) for s in X]))
    variants.append(np.stack([_frame_dropout(s, 0.10, rng) for s in X]))
    variants.append(np.stack([_hand_dropout(s, rng) for s in X]))
    Xa = np.concatenate(variants).astype(np.float32)
    ya = np.concatenate([y] * len(variants))
    return Xa, ya


def _make_train_dataset(X, y, tf):
    """tf.data pipeline with per-batch jitter (global scale + gaussian noise).

    Layout-agnostic on purpose: the word feature mixes 3-value landmark triples
    with 2-value position tails, so a per-landmark rotation does not apply
    cleanly. Mirror + time-warp (offline) handle the structural variety;
    multi-session capture provides the real-world variety."""
    def _jitter(x, yy):
        x = x * tf.random.uniform((), 0.9, 1.1)
        x = x + tf.random.normal(tf.shape(x), stddev=NOISE_STD)
        return x, yy

    return (tf.data.Dataset.from_tensor_slices((X, y))
            .shuffle(len(X), seed=SEED, reshuffle_each_iteration=True)
            .map(_jitter, num_parallel_calls=tf.data.AUTOTUNE)
            .batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE))


def train_model(Xtr, ytr, Xva, yva, num_classes, epochs=EPOCHS,
                verbose=2, summary=False):
    """
    Train one TCN on the given (un-augmented) arrays and return the fitted model.

    Offline augmentation + per-batch jitter + class weighting are applied
    internally, so training/main and evaluation/cross-validation share ONE
    definition of "how a word model is trained" (no architecture drift).
    """
    import tensorflow as tf
    from tensorflow.keras import callbacks
    from sklearn.utils.class_weight import compute_class_weight

    Xtr, ytr = _offline_augment(Xtr, ytr)
    # Weight only the classes actually present in this train split. A class can be
    # absent when its only session is the held-out one (session-grouped CV/holdout
    # of a class that was captured in a single session); absent classes get a
    # neutral weight so Keras never looks up a missing key.
    present = np.unique(ytr)
    weights = compute_class_weight("balanced", classes=present, y=ytr)
    class_weight = {int(c): float(w) for c, w in zip(present, weights)}
    for c in range(num_classes):
        class_weight.setdefault(c, 1.0)

    train_ds = _make_train_dataset(Xtr, ytr, tf)
    val_ds = (tf.data.Dataset.from_tensor_slices((Xva, yva)).batch(BATCH_SIZE)
              if len(Xva) else None)

    model = _build_model(num_classes)
    if summary:
        model.summary()

    monitor = "val_accuracy" if val_ds else "accuracy"
    cb = [
        callbacks.EarlyStopping(monitor=monitor, mode="max", patience=30,
                                restore_best_weights=True, verbose=0),
        callbacks.ReduceLROnPlateau(monitor=monitor, mode="max", factor=0.6,
                                    patience=10, min_lr=1e-6, verbose=0),
    ]
    model.fit(train_ds, validation_data=val_ds, epochs=epochs,
              callbacks=cb, class_weight=class_weight, verbose=verbose)
    return model


def _build_model(num_classes):
    import tensorflow as tf
    from tensorflow.keras import layers, models

    inp = layers.Input(shape=(T, FEATURE_DIM), name="sequence")
    x = layers.Conv1D(64, 3, padding="causal", dilation_rate=1, activation="relu")(inp)
    x = layers.BatchNormalization()(x)
    x = layers.Conv1D(64, 3, padding="causal", dilation_rate=2, activation="relu")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Conv1D(128, 3, padding="causal", dilation_rate=4, activation="relu")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.3)(x)
    x = layers.Conv1D(128, 3, padding="causal", dilation_rate=8, activation="relu")(x)
    x = layers.GlobalAveragePooling1D()(x)
    x = layers.Dropout(0.4)(x)
    x = layers.Dense(128, activation="relu")(x)
    out = layers.Dense(num_classes, activation="softmax", name="gloss")(x)

    model = models.Model(inp, out, name="word_tcn")
    model.compile(
        optimizer=tf.keras.optimizers.Adam(LR),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def main():
    import tensorflow as tf

    np.random.seed(SEED)
    tf.random.set_seed(SEED)

    if not os.path.isfile(MANIFEST):
        print(f"[ERROR] No manifest at {MANIFEST}")
        print("  Capture data first: python capture/capture_words.py")
        sys.exit(1)

    rows, glosses, label_to_idx = load_manifest()
    num_classes = len(glosses)
    if num_classes < 2:
        print(f"[ERROR] Only {num_classes} class captured. Need at least 2.")
        sys.exit(1)

    X, y, groups = load_samples(rows, label_to_idx)
    if len(X) == 0:
        print("[ERROR] No training samples loaded.")
        sys.exit(1)

    n_sessions = len(set(groups.tolist()))
    tr_idx, va_idx = session_holdout(y, groups)
    Xtr, ytr = X[tr_idx], y[tr_idx]
    Xva, yva = X[va_idx], y[va_idx]

    print(f"Vocabulary ({num_classes}): {', '.join(glosses)}")
    print(f"  sessions {n_sessions}  train {len(Xtr)}  val {len(Xva)}  "
          "(x8 after offline augmentation)")
    if n_sessions < 2:
        print("  [WARN] Only one capture session, so validation is empty: holding "
              "out whole sessions is the only leak-free split, and there is no "
              "second session to hold out. Train accuracy alone is optimistic — "
              "capture another session (different day/lighting) for an honest read.")
    if len(Xtr) < num_classes * 5:
        print("  [WARN] Few takes per word. Capture more — ideally across several "
              "sessions (different days/lighting) so the model generalizes.")
    if "nothing" not in glosses:
        print("  [WARN] No 'nothing' (negative) class found. Without it the model "
              "is forced to label every frame as some word and never stays silent. "
              "Capture it with capture/capture_words.py.")

    print("\nBuilding + training model...\n")
    model = train_model(Xtr, ytr, Xva, yva, num_classes, summary=True)

    print("\n" + "=" * 60)
    val_acc = None
    if len(Xva):
        _, val_acc = model.evaluate(Xva, yva, verbose=0)
        print(f"  val accuracy: {val_acc * 100:.1f}%  (n={len(Xva)}, held-out sessions)")
        print("        Val takes come from sessions the model never trained on, so")
        print("        this reflects live performance. For a per-class breakdown")
        print("        run: python training/evaluate.py --cv 5")
    print("=" * 60)

    os.makedirs(os.path.dirname(os.path.join(HERE, MODEL_OUT)), exist_ok=True)
    model.save(os.path.join(HERE, MODEL_OUT))
    with open(os.path.join(HERE, LABELS_OUT), "w", encoding="utf-8") as f:
        json.dump({str(i): g for g, i in label_to_idx.items()},
                  f, indent=2, ensure_ascii=False)
    write_run_metadata(
        "words", glosses,
        samples_per_class=Counter(glosses[i] for i in y),
        n_sessions=len(set(groups.tolist())),
        val_accuracy=val_acc,
        config={"WORD_SEQ_LEN": T, "WORD_FEATURE_DIM": FEATURE_DIM,
                "epochs": EPOCHS, "batch_size": BATCH_SIZE, "lr": LR},
    )

    print(f"Model  -> {MODEL_OUT}")
    print(f"Labels -> {LABELS_OUT}")
    print("\nWord training complete. Run the app: python main.py")


if __name__ == "__main__":
    main()
