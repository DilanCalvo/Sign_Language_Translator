"""
One-hand (letter) model training.

Input:
    CSVs captured with capture/capture_letters.py

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

import json
import os
import sys

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


MODEL_OUT  = "model/model_one_hand.h5"
LABELS_OUT = "model/labels_one_hand.json"

# CSVs captured with capture/capture_letters.py.
# Can be a single path (str) or a list of paths (list).
DATA_CSVS = [
    "data/real_capture/letters/capture_20260520_180454_A-B-C.csv",
    "data/real_capture/letters/capture_20260520_143836_D-E-F.csv",
    "data/real_capture/letters/capture_20260520_145116_G-H-I-.csv",
    "data/real_capture/letters/capture_20260520_173739_I-K-L.csv",
    "data/real_capture/letters/capture_20260520_174943_M-N-O.csv",
    "data/real_capture/letters/capture_20260520_182414_P-Q-R.csv",
    "data/real_capture/letters/capture_20260520_184531_S-T-U.csv",
    "data/real_capture/letters/capture_20260520_204902_V-W-X.csv",
    "data/real_capture/letters/capture_20260521_212857_Y.csv"
    # Add new capture sessions here:
    # "data/real_capture/letters/capture_YYYYMMDD_HHMMSS_XYZ.csv",
]

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


def _load_data(csvs) -> tuple[np.ndarray, np.ndarray]:
    paths = [csvs] if isinstance(csvs, str) else csvs
    frames = []
    for p in paths:
        if p and os.path.isfile(p):
            df = pd.read_csv(p)
            frames.append(df)
            print(f"  + {len(df):,} samples from {p}")
        elif p:
            print(f"  [WARN] File not found: {p}")

    if not frames:
        print("[ERROR] No CSV found. Check DATA_CSVS.")
        raise SystemExit(1)

    df = pd.concat(frames, ignore_index=True)
    df = df[~df["label"].isin(EXCLUDED_CLASSES)].reset_index(drop=True)
    labels = np.array(df["label"].tolist())
    raw = df.drop(columns=["label"]).to_numpy(dtype=np.float32)
    normalized = np.stack([normalize_landmarks(row) for row in raw])
    return normalized, labels


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


def main():
    np.random.seed(SEED)
    tf.random.set_seed(SEED)

    print("Loading and normalizing data...")
    x_all, y_all_str = _load_data(DATA_CSVS)

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

    weights = compute_class_weight("balanced", classes=np.arange(num_classes), y=y_train)
    class_weight = {i: float(w) for i, w in enumerate(weights)}

    train_ds = _build_dataset(x_train, y_train, training=True)
    val_ds   = _build_dataset(x_val,   y_val,   training=False)

    print("Building model...")
    model = _build_model(input_dim=x_train.shape[1], num_classes=num_classes)
    model.summary()

    cb = [
        callbacks.EarlyStopping(
            monitor="val_accuracy",
            patience=35,
            restore_best_weights=True,
            verbose=1,
        ),
        callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.7,
            patience=8,
            min_lr=1e-6,
            verbose=1,
        ),
        callbacks.ModelCheckpoint(
            MODEL_OUT,
            monitor="val_accuracy",
            save_best_only=True,
            verbose=0,
        ),
    ]

    print("\nTraining...\n")
    model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=EPOCHS,
        callbacks=cb,
        class_weight=class_weight,
        verbose=2,
    )

    model.save(MODEL_OUT)

    print("\n" + "=" * 60)
    print("Final result (best weights restored):")
    val_loss, val_acc = model.evaluate(val_ds, verbose=0)
    print(f"  val_loss:     {val_loss:.4f}")
    print(f"  val_accuracy: {val_acc:.4f}  ({val_acc * 100:.2f} %)")
    print("=" * 60)

    y_pred = np.argmax(model.predict(x_val, verbose=0), axis=1)
    print("\nPer-class report (val):")
    print(classification_report(y_val, y_pred, target_names=encoder.classes_, digits=3))

    labels_dict = {str(i): name for i, name in enumerate(encoder.classes_)}
    with open(LABELS_OUT, "w", encoding="utf-8") as f:
        json.dump(labels_dict, f, indent=2, ensure_ascii=False)

    print(f"\nModel saved to:  {MODEL_OUT}")
    print(f"Labels saved to: {LABELS_OUT}")


if __name__ == "__main__":
    main()
