"""
Real-time classification.

Loads the trained models and classifies landmarks into letters/words based
on the data produced by detector.py.

Input contract (dict from detector.py):
    {"num_hands": int, "landmarks_hand1": list[float]|None, "landmarks_hand2": list[float]|None}

Output contract:
    {
        "prediction":      str|None,   # best letter (None if confidence < threshold)
        "confidence":      float,      # letter confidence (0-1)
        "model_used":      str|None,   # "one_hand" | None
        "runner_up":       dict|None,  # {"prediction": str, "confidence": float} or None
        "word_prediction": dict|None,  # {"prediction": str, "confidence": float} or None
        "hand_is_signing": bool,       # True while the hand moves enough for a word
    }

runner_up: second letter candidate when its confidence >= 15%.
word_prediction: dynamic-sign prediction. It is None when:
  - the word model is not loaded,
  - the buffer does not yet hold WORD_SEQ_FRAMES frames,
  - hand motion is below WORD_MIN_MOTION_STD (static pose),
  - the model confidence is below WORD_CONFIDENCE_THRESHOLD,
  - or the predicted class is the negative class WORD_NULL_LABEL ("nada").
"""

import json
import os
from collections import deque

import numpy as np
import tensorflow as tf

from config import (
    MODEL_ONE_HAND_PATH   as _MODEL_ONE,
    MODEL_WORDS_PATH      as _MODEL_WORDS,
    LABELS_ONE_HAND_PATH  as _LABELS_ONE,
    LABELS_WORDS_PATH     as _LABELS_WORDS,
    WORD_SEQ_FRAMES       as _SEQ_FRAMES,
    WORD_MOTION_WINDOW    as _MOTION_WINDOW,
    WORD_NULL_LABEL       as _NULL_WORD_LABEL,
    LETTER_CONFIDENCE_THRESHOLD as CONFIDENCE_THRESHOLD,
    WORD_CONFIDENCE_THRESHOLD,
    WORD_MIN_MOTION_STD   as _MIN_MOTION_STD,
)
from src.utils import normalize_landmarks


class Classifier:
    def __init__(self):
        if not os.path.isfile(_MODEL_ONE):
            raise RuntimeError(
                f"One-hand model not found: {_MODEL_ONE}\n"
                "  Run: python training/train_letters.py"
            )

        self._model_one  = tf.keras.models.load_model(_MODEL_ONE, compile=False)
        self._labels_one = self._load_labels(_LABELS_ONE)

        # The word model is optional — a buffer of SEQ_FRAMES normalized
        # landmarks turned into mean+std (126 features) for each inference.
        self._model_words  = None
        self._labels_words = None
        self._word_buffer  = deque(maxlen=_SEQ_FRAMES)
        if os.path.isfile(_MODEL_WORDS) and os.path.isfile(_LABELS_WORDS):
            labels_candidate = self._load_labels(_LABELS_WORDS)
            if len(labels_candidate) < 2:
                # A softmax classifier with a single class always returns
                # probability 1.0 regardless of the input — useless.
                print(
                    f"  [WARN] The word model only has {len(labels_candidate)} class "
                    f"({', '.join(labels_candidate.values())}). "
                    "At least 2 words are needed for predictions to be valid.\n"
                    "         Capture more words with: python capture/capture_words.py"
                )
            else:
                self._model_words  = tf.keras.models.load_model(_MODEL_WORDS, compile=False)
                self._labels_words = labels_candidate

        # Motion level of the last analyzed frame (updated by _run_words).
        # Exposed in the classify() result as "hand_is_signing".
        self._last_motion = 0.0

        # The first TF forward pass compiles the XLA graph (200-800ms). We run
        # it here with zeros so the first real frame is instant.
        self._warmup()

        n_one   = len(self._labels_one)
        n_words = len(self._labels_words) if self._labels_words else 0
        print(f"Classifier ready  |  letters: {n_one}  |  words: {n_words}")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def classify(self, landmarks_data):
        num_hands = landmarks_data["num_hands"]
        h1 = landmarks_data["landmarks_hand1"]

        if num_hands == 0 or h1 is None:
            self._word_buffer.clear()
            self._last_motion = 0.0
            return self._empty()

        # Update the word buffer with the current frame.
        # _last_motion is updated internally in _run_words().
        word_pred = self._update_word_buffer(h1)

        # Letter classification (always uses the primary hand).
        result = self._run(self._model_one, self._labels_one, h1, "one_hand")

        result["word_prediction"] = word_pred
        # True only if the word model is loaded AND the hand moves enough.
        # When True, main.py suppresses letter predictions.
        result["hand_is_signing"] = self._last_motion >= _MIN_MOTION_STD
        return result

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _run(self, model, labels, flat, model_name):
        normalized = normalize_landmarks(flat)
        tensor = tf.constant([normalized], dtype=tf.float32)

        # Calling the model directly is 3-5x faster than model.predict()
        # for single samples in real time.
        probs = model(tensor, training=False).numpy()[0]

        # Top-2 indices sorted from highest to lowest probability.
        top2  = np.argsort(probs)[-2:][::-1]
        idx   = int(top2[0])
        idx2  = int(top2[1])

        confidence  = float(probs[idx])
        confidence2 = float(probs[idx2])

        if confidence < CONFIDENCE_THRESHOLD:
            return self._empty()

        runner_up = (
            {"prediction": labels[idx2], "confidence": confidence2}
            if confidence2 >= 0.15 else None
        )

        return {
            "prediction": labels[idx],
            "confidence": confidence,
            "model_used": model_name,
            "runner_up":  runner_up,
        }

    def _update_word_buffer(self, h1_flat):
        if self._model_words is None:
            return None
        normalized = normalize_landmarks(h1_flat)
        self._word_buffer.append(normalized)
        if len(self._word_buffer) < _SEQ_FRAMES:
            self._last_motion = 0.0
            return None
        return self._run_words()

    def _run_words(self):
        arr = np.array(list(self._word_buffer), dtype=np.float32)  # (20, 63)

        # Full std: the word model features (what it was trained on).
        std_per_coord = arr.std(axis=0)                              # (63,)

        # Motion over only the most recent frames: detects quickly when the
        # user stops the hand. Averaging all 20 frames would keep a motion
        # from 0.5s ago counting, delaying letters. A short window solves this
        # without losing word-detection accuracy (that uses the full buffer).
        recent = arr[-_MOTION_WINDOW:]                               # (8, 63)
        motion = float(recent.std(axis=0).mean())
        self._last_motion = motion

        if motion < _MIN_MOTION_STD:
            return None

        feat   = np.concatenate([arr.mean(axis=0), std_per_coord])  # (126,)
        tensor = tf.constant(feat[np.newaxis], dtype=tf.float32)
        probs  = self._model_words(tensor, training=False).numpy()[0]
        idx    = int(np.argmax(probs))
        conf   = float(probs[idx])
        if conf < WORD_CONFIDENCE_THRESHOLD:
            return None

        label = self._labels_words[idx]
        # The negative class ("nada") means "the model thinks this is not a
        # deliberate word" — we treat that prediction as None.
        if label == _NULL_WORD_LABEL:
            return None

        return {"prediction": label, "confidence": conf}

    def _warmup(self):
        self._model_one(tf.zeros((1, 63), dtype=tf.float32), training=False)
        if self._model_words is not None:
            self._model_words(tf.zeros((1, 126), dtype=tf.float32), training=False)

    @staticmethod
    def _load_labels(path):
        with open(path, "r", encoding="utf-8") as f:
            raw = json.load(f)
        # Convert keys to int for direct index access.
        return {int(k): v for k, v in raw.items()}

    @staticmethod
    def _empty():
        return {
            "prediction":      None,
            "confidence":      0.0,
            "model_used":      None,
            "runner_up":       None,
            "word_prediction": None,
            "hand_is_signing": False,
        }
