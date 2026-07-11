"""
One-hand (letter) model training.

Input:
    Every CSV in data/real_capture/letters/ (auto-discovered). Capture more with
    capture/capture_letters.py and just re-run this — no paths to edit, so a
    multi-session dataset (different days / lighting / distance) is frictionless.

Output:
    model/model_one_hand.h5         (best model by val_accuracy)
    model/labels_one_hand.json      (index -> class map)

Strategy:
    - Wrist-centered, middle-base-scaled normalization.
    - Stratified 80/20 split over the captured data.
    - Class weights to correct imbalance.
    - Small Gaussian-noise augmentation (robustness to MediaPipe jitter).
    - Random horizontal mirror (learns both hands).
    - Dense architecture with BatchNorm + Dropout + L2 regularization.
    - EarlyStopping + ReduceLROnPlateau + ModelCheckpoint.
"""

import glob
import json
import os
import sys
from collections import Counter

import numpy as np
import pandas as pd
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_class_weight

import tensorflow as tf
from tensorflow.keras import callbacks, layers, models, regularizers

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.utils import normalize_landmarks
from training.run_log import write_run_metadata


MODEL_OUT  = "model/model_one_hand.h5"
LABELS_OUT = "model/labels_one_hand.json"

# Directory scanned for capture CSVs. Every *.csv here is used, so adding a
# session is just re-running capture/capture_letters.py (no code change).
DATA_DIR = "data/real_capture/letters"

EXCLUDED_CLASSES = {"nothing"}

BATCH_SIZE   = 64      # small = more gradient steps per epoch on a small dataset
EPOCHS       = 300
NOISE_STDDEV = 0.012   # more noise -> more robust to MediaPipe variations
SCALE_JITTER = 0.08    # +/-8% scale -> simulates the hand at different distances
L2   = 1e-4            # more regularization for a small dataset
SEED = 42

_MIRROR_MASK_63 = tf.constant(
    [-1.0 if i % 3 == 0 else 1.0 for i in range(63)],
    dtype=tf.float32,
)


def _load_data(data_dir) -> tuple[np.ndarray, np.ndarray, int]:
    """Load every capture CSV in `data_dir`. Returns (X, labels, n_files);
    n_files doubles as the session count for the run record (one capture run
    writes one timestamped file)."""
    paths = sorted(glob.glob(os.path.join(data_dir, "*.csv")))
    if not paths:
        print(f"[ERROR] No CSV found in {data_dir}/. "
              "Capture first: python capture/capture_letters.py")
        raise SystemExit(1)

    frames = []
    for p in paths:
        df = pd.read_csv(p)
        frames.append(df)
        print(f"  + {len(df):,} samples from {p}")

    df = pd.concat(frames, ignore_index=True)
    df = df[~df["label"].isin(EXCLUDED_CLASSES)].reset_index(drop=True)
    labels = np.array([str(v) for v in df["label"].tolist()])
    raw = df.drop(columns=["label"]).to_numpy(dtype=np.float32)
    normalized = np.stack([normalize_landmarks(row) for row in raw])
    return normalized, labels, len(paths)


def _augment(x, y):
    # Gaussian noise: robustness to MediaPipe jitter.
    x = x + tf.random.normal(tf.shape(x), mean=0.0, stddev=NOISE_STDDEV)

    # Scale jitter: simulates the hand at different distances from the camera.
    # Since the landmarks are normalized (no absolute scale), a single global
    # multiplicative factor is the correct way to simulate this effect.
    scale = tf.random.uniform((), 1.0 - SCALE_JITTER, 1.0 + SCALE_JITTER)
    x = x * scale

    # Horizontal mirror with p=0.5: learns both hands.
    should_mirror = tf.random.uniform(()) > 0.5
    x = tf.cond(should_mirror, lambda: x * _MIRROR_MASK_63, lambda: x)

    return x, y


def _build_dataset(x, y, training):
    ds = tf.data.Dataset.from_tensor_slices((x, y))
    if training:
        ds = ds.shuffle(buffer_size=min(len(x), 10000), seed=SEED, reshuffle_each_iteration=True)
        ds = ds.map(_augment, num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
    return ds


def _build_model(input_dim, num_classes):
    inputs = layers.Input(shape=(input_dim,), name="landmarks")

    x = layers.Dense(256, activation="relu", kernel_regularizer=regularizers.l2(L2))(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.35)(x)

    x = layers.Dense(128, activation="relu", kernel_regularizer=regularizers.l2(L2))(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.35)(x)

    x = layers.Dense(64, activation="relu", kernel_regularizer=regularizers.l2(L2))(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.2)(x)

    x = layers.Dense(32, activation="relu", kernel_regularizer=regularizers.l2(L2))(x)
    x = layers.BatchNormalization()(x)

    outputs = layers.Dense(num_classes, activation="softmax", name="prediction")(x)

    model = models.Model(inputs, outputs, name="one_hand_classifier")
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def train_model(Xtr, ytr, Xva, yva, num_classes, epochs=EPOCHS,
                verbose=2, summary=False, checkpoint_path=None):
    """Train one dense model on integer-labeled arrays and return it.

    Shared by main() and the session-grouped cross-validator (training/
    eval_common) so the architecture and training recipe never drift between
    training and evaluation. Handles an empty validation set (used by CV, which
    scores each held-out fold separately): callbacks then monitor train accuracy.
    """
    # Weight only the classes present in this split. A class can be absent when
    # its only capture file is the held-out fold (session-grouped CV); absent
    # classes get a neutral weight so Keras never looks up a missing key.
    present = np.unique(ytr)
    weights = compute_class_weight("balanced", classes=present, y=ytr)
    class_weight = {int(c): float(w) for c, w in zip(present, weights)}
    for c in range(num_classes):
        class_weight.setdefault(c, 1.0)

    train_ds = _build_dataset(Xtr, ytr, training=True)
    val_ds   = _build_dataset(Xva, yva, training=False) if len(Xva) else None

    model = _build_model(input_dim=Xtr.shape[1], num_classes=num_classes)
    if summary:
        model.summary()

    v = 1 if verbose else 0
    monitor = "val_accuracy" if val_ds is not None else "accuracy"
    cb = [
        callbacks.EarlyStopping(monitor=monitor, mode="max", patience=35,
                                restore_best_weights=True, verbose=v),
        callbacks.ReduceLROnPlateau(
            monitor="val_loss" if val_ds is not None else "loss",
            factor=0.7, patience=8, min_lr=1e-6, verbose=v),
    ]
    # Only checkpoint to disk when asked (main() passes MODEL_OUT). CV must never
    # overwrite the real model, so it passes checkpoint_path=None.
    if checkpoint_path is not None and val_ds is not None:
        cb.append(callbacks.ModelCheckpoint(
            checkpoint_path, monitor="val_accuracy", save_best_only=True, verbose=0))

    model.fit(train_ds, validation_data=val_ds, epochs=epochs,
              callbacks=cb, class_weight=class_weight, verbose=verbose)
    return model


def main():
    np.random.seed(SEED)
    tf.random.set_seed(SEED)

    print("Loading and normalizing data...")
    x_all, y_all_str, n_files = _load_data(DATA_DIR)

    try:
        x_train, x_val, y_train_str, y_val_str = train_test_split(
            x_all, y_all_str, test_size=0.2, stratify=y_all_str, random_state=SEED,
        )
    except ValueError:
        x_train, x_val, y_train_str, y_val_str = train_test_split(
            x_all, y_all_str, test_size=0.2, random_state=SEED,
        )

    print(f"  Train: {x_train.shape[0]:>6,} samples x {x_train.shape[1]} features")
    print(f"  Val:   {x_val.shape[0]:>6,} samples x {x_val.shape[1]} features")

    encoder = LabelEncoder()
    y_train = encoder.fit_transform(y_train_str)

    val_mask = np.isin(y_val_str, encoder.classes_)
    if not val_mask.all():
        dropped = np.unique(y_val_str[~val_mask])
        print(f"  [WARN] Val labels missing from train: {dropped} -> dropped")
    x_val      = x_val[val_mask]
    y_val_str  = y_val_str[val_mask]
    y_val      = encoder.transform(y_val_str)

    num_classes = len(encoder.classes_)
    print(f"  Classes ({num_classes}): {', '.join(encoder.classes_)}\n")

    print("Building + training model...\n")
    model = train_model(x_train, y_train, x_val, y_val, num_classes,
                        summary=True, checkpoint_path=MODEL_OUT)

    model.save(MODEL_OUT)

    print("\n" + "=" * 60)
    print("Final result (best weights restored):")
    val_loss, val_acc = model.evaluate(x_val, y_val, verbose=0)
    print(f"  val_loss:     {val_loss:.4f}")
    print(f"  val_accuracy: {val_acc:.4f}  ({val_acc * 100:.2f} %)")
    print("=" * 60)

    y_pred = np.argmax(model.predict(x_val, verbose=0), axis=1)
    print("\nPer-class report (val):")
    print(classification_report(y_val, y_pred, target_names=encoder.classes_, digits=3))

    labels_dict = {str(i): name for i, name in enumerate(encoder.classes_)}
    with open(LABELS_OUT, "w", encoding="utf-8") as f:
        json.dump(labels_dict, f, indent=2, ensure_ascii=False)

    write_run_metadata(
        "letters", list(encoder.classes_),
        samples_per_class=Counter(y_all_str.tolist()),
        n_sessions=n_files,
        val_accuracy=val_acc,
        config={"epochs": EPOCHS, "batch_size": BATCH_SIZE,
                "noise_stddev": NOISE_STDDEV, "scale_jitter": SCALE_JITTER,
                "l2": L2, "seed": SEED},
    )

    print(f"\nModel saved to:  {MODEL_OUT}")
    print(f"Labels saved to: {LABELS_OUT}")


if __name__ == "__main__":
    main()
