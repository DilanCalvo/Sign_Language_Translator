"""
Word / moving-sign model training — temporal sequence model (TCN).

Each sample is a (WORD_SEQ_LEN, 63) sequence of normalized one-hand landmarks
captured by capture/capture_words.py. Unlike the old mean+std model (which was
order-blind and collapsed as the vocabulary grew), this is a TEMPORAL model:
stacked dilated 1D convolutions read the movement OVER TIME, so it can tell
similar signs apart by HOW the hand moves, not just the average pose.

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

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import (
    WORD_SEQ_LEN       as T,
    CAPTURE_OUTPUT_DIR as _DATA_DIR,
    MODEL_WORDS_PATH   as MODEL_OUT,
    LABELS_WORDS_PATH  as LABELS_OUT,
)
from src.utils import mirror_sequence

HERE     = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(HERE, _DATA_DIR)
SEQ_DIR  = os.path.join(DATA_DIR, "seq")
MANIFEST = os.path.join(DATA_DIR, "manifest.csv")

FEATURE_DIM = 63          # one hand: 21 landmarks x (x, y, z)
NPTS        = FEATURE_DIM // 3
BATCH_SIZE  = 16
EPOCHS      = 200
LR          = 1e-3
NOISE_STD   = 0.02
SEED        = 42


def _load_split(rows, subset, label_to_idx):
    X, y = [], []
    for r in rows:
        if r["subset"] != subset:
            continue
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
    if not X:
        return (np.empty((0, T, FEATURE_DIM), np.float32),
                np.empty((0,), np.int64))
    return np.stack(X).astype(np.float32), np.array(y, np.int64)


def _time_warp(seq, gamma):
    """Speed change: same length, non-linear time remap. Teaches the model the
    SAME sign performed faster/slower (attacks the clean-capture vs live-pace gap)."""
    u = np.linspace(0.0, 1.0, T) ** gamma
    idx = (u * (T - 1)).round().astype(int)
    return seq[idx]


def _offline_augment(X, y):
    """Mirror + time-warp variants, applied once before training to enlarge the
    tiny dataset. (Per-batch jitter is added separately in the tf pipeline.)"""
    variants = [X, np.stack([mirror_sequence(s) for s in X])]
    for g in (0.7, 1.4):
        variants.append(np.stack([_time_warp(s, g) for s in X]))
        variants.append(np.stack([_time_warp(mirror_sequence(s), g) for s in X]))
    Xa = np.concatenate(variants).astype(np.float32)
    ya = np.concatenate([y] * len(variants))
    return Xa, ya


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
    from tensorflow.keras import callbacks
    from sklearn.utils.class_weight import compute_class_weight

    np.random.seed(SEED)
    tf.random.set_seed(SEED)

    if not os.path.isfile(MANIFEST):
        print(f"[ERROR] No manifest at {MANIFEST}")
        print("  Capture data first: python capture/capture_words.py")
        sys.exit(1)

    rows = list(csv.DictReader(open(MANIFEST, encoding="utf-8")))
    glosses = sorted({r["gloss"] for r in rows})
    label_to_idx = {g: i for i, g in enumerate(glosses)}
    num_classes = len(glosses)
    if num_classes < 2:
        print(f"[ERROR] Only {num_classes} class captured. Need at least 2.")
        sys.exit(1)

    Xtr, ytr = _load_split(rows, "train", label_to_idx)
    Xva, yva = _load_split(rows, "val", label_to_idx)
    print(f"Vocabulary ({num_classes}): {', '.join(glosses)}")
    print(f"  train {len(Xtr)}  val {len(Xva)}")
    if len(Xtr) < num_classes * 5:
        print("  [WARN] Few takes per word. Capture more — ideally across several "
              "sessions (different days/lighting) so the model generalizes.")
    if len(Xtr) == 0:
        print("[ERROR] No training samples loaded.")
        sys.exit(1)

    Xtr, ytr = _offline_augment(Xtr, ytr)
    print(f"  train after offline augmentation: {len(Xtr)}")

    weights = compute_class_weight("balanced", classes=np.arange(num_classes), y=ytr)
    class_weight = {i: float(w) for i, w in enumerate(weights)}

    def _jitter(x, y):
        """Per-batch augmentation: small scale, 2D rotation, joint dropout, noise."""
        x = x * tf.random.uniform((), 0.9, 1.1)
        th = tf.random.uniform((), -0.20, 0.20)
        pts = tf.reshape(x, (T, NPTS, 3))
        px, py, pz = pts[..., 0], pts[..., 1], pts[..., 2]
        c, s = tf.cos(th), tf.sin(th)
        pts = tf.stack([px * c - py * s, px * s + py * c, pz], axis=-1)
        keep = tf.cast(tf.random.uniform((T, NPTS, 1)) > 0.07, tf.float32)
        x = tf.reshape(pts * keep, (T, FEATURE_DIM))
        x = x + tf.random.normal(tf.shape(x), stddev=NOISE_STD)
        return x, y

    train_ds = (tf.data.Dataset.from_tensor_slices((Xtr, ytr))
                .shuffle(len(Xtr), seed=SEED, reshuffle_each_iteration=True)
                .map(_jitter, num_parallel_calls=tf.data.AUTOTUNE)
                .batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE))
    val_ds = (tf.data.Dataset.from_tensor_slices((Xva, yva)).batch(BATCH_SIZE)
              if len(Xva) else None)

    print("\nBuilding model...")
    model = _build_model(num_classes)
    model.summary()

    monitor = "val_accuracy" if val_ds else "accuracy"
    cb = [
        callbacks.EarlyStopping(monitor=monitor, mode="max", patience=30,
                                restore_best_weights=True, verbose=1),
        callbacks.ReduceLROnPlateau(monitor=monitor, mode="max", factor=0.6,
                                    patience=10, min_lr=1e-6, verbose=1),
    ]

    print("\nTraining...\n")
    model.fit(train_ds, validation_data=val_ds, epochs=EPOCHS,
              callbacks=cb, class_weight=class_weight, verbose=2)

    print("\n" + "=" * 60)
    if val_ds is not None and len(Xva):
        loss, acc = model.evaluate(Xva, yva, verbose=0)
        print(f"  val accuracy: {acc * 100:.1f}%  (n={len(Xva)})")
        print("  NOTE: if all val takes come from the SAME session as train,")
        print("        this number is optimistic. Real precision needs val takes")
        print("        captured on a DIFFERENT day.")
    print("=" * 60)

    os.makedirs(os.path.dirname(os.path.join(HERE, MODEL_OUT)), exist_ok=True)
    model.save(os.path.join(HERE, MODEL_OUT))
    with open(os.path.join(HERE, LABELS_OUT), "w", encoding="utf-8") as f:
        json.dump({str(i): g for g, i in label_to_idx.items()},
                  f, indent=2, ensure_ascii=False)

    print(f"Model  -> {MODEL_OUT}")
    print(f"Labels -> {LABELS_OUT}")
    print("\nWord training complete. Run the app: python main.py")


if __name__ == "__main__":
    main()
