"""
Real-time classification.

Loads the trained models and classifies landmarks into letters/words based
on the data produced by detector.py.

Input contract (dict from detector.py):
    {"num_hands": int, "landmarks_hand1": list|None, "landmarks_hand2": list|None,
     "hands_by_side": {"Left": list|None, "Right": list|None},
     "shoulder_l": (x,y)|None, "shoulder_r": (x,y)|None}
Letters use landmarks_hand1 (detection order); words use hands_by_side + shoulders.

Output contract:
    {
        "prediction":      str|None,   # committed top-1 (None if conf < threshold)
        "confidence":      float,      # real top-1 confidence (0-1), even if not committed
        "model_used":      str|None,   # "one_hand" | "numbers" | None
        "top3":            list,       # [{"prediction": str, "confidence": float}] x<=3, best first
        "word_prediction": dict|None,  # {"prediction": str, "confidence": float} or None
        "hand_is_signing": bool,       # True while the hand moves enough for a word
    }

prediction vs top3: `prediction` is the committed guess (None below threshold,
used to drive the subtitle buffer and voice). `top3` is ALWAYS the raw ranked
candidates regardless of threshold, so the HUD can show alternatives when the
model is unsure (confidence < LOW_CONFIDENCE_THRESHOLD). This separates "what
we act on" from "what hints we surface".

word_prediction: dynamic-sign prediction. It is None when:
  - the word model is not loaded,
  - the buffer does not yet hold WORD_MIN_FRAMES frames,
  - hand motion is below WORD_MIN_MOTION_STD (static pose),
  - the model confidence is below WORD_CONFIDENCE_THRESHOLD,
  - or the predicted class is the negative class WORD_NULL_LABEL ("nothing").

The word model is a TEMPORAL sequence model (TCN): the rolling buffer of recent
body-anchored two-hand frames (WORD_FEATURE_DIM values each, built by
src.utils.build_word_features) is resampled to WORD_SEQ_LEN and fed as a
(1, WORD_SEQ_LEN, WORD_FEATURE_DIM) tensor — it reads the movement over time,
including WHERE the hands are relative to the body.
"""

import json
import os
from collections import deque

import numpy as np
import tensorflow as tf

from config import (
    MODEL_ONE_HAND_PATH   as _MODEL_ONE,
    MODEL_WORDS_PATH      as _MODEL_WORDS,
    MODEL_NUMBERS_PATH    as _MODEL_NUMBERS,
    LABELS_ONE_HAND_PATH  as _LABELS_ONE,
    LABELS_WORDS_PATH     as _LABELS_WORDS,
    LABELS_NUMBERS_PATH   as _LABELS_NUMBERS,
    WORD_SEQ_LEN          as _SEQ_LEN,
    WORD_FEATURE_DIM      as _FEATURE_DIM,
    WORD_BUFFER_FRAMES    as _BUFFER_FRAMES,
    WORD_MIN_FRAMES       as _MIN_FRAMES,
    WORD_MOTION_WINDOW    as _MOTION_WINDOW,
    WORD_NULL_LABEL       as _NULL_WORD_LABEL,
    LETTER_CONFIDENCE_THRESHOLD as _LETTER_THRESHOLD,
    NUMBER_CONFIDENCE_THRESHOLD as _NUMBER_THRESHOLD,
    WORD_CONFIDENCE_THRESHOLD,
    WORD_MIN_MOTION_STD   as _MIN_MOTION_STD,
    ALT_MIN_CONFIDENCE    as _ALT_MIN_CONFIDENCE,
)
from src.utils import normalize_landmarks, resample_sequence, build_word_features


class Classifier:
    def __init__(self):
        if not os.path.isfile(_MODEL_ONE):
            raise RuntimeError(
                f"One-hand model not found: {_MODEL_ONE}\n"
                "  Run: python training/train_letters.py"
            )

        self._model_one  = tf.keras.models.load_model(_MODEL_ONE, compile=False)
        self._labels_one = self._load_labels(_LABELS_ONE)

        # The word model is optional — a rolling buffer of recent normalized
        # frames, resampled to WORD_SEQ_LEN and fed to the temporal model (TCN).
        self._model_words  = None
        self._labels_words = None
        self._word_buffer  = deque(maxlen=_BUFFER_FRAMES)
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
                candidate = tf.keras.models.load_model(_MODEL_WORDS, compile=False)
                # Be honest about an incompatible model instead of crashing at
                # inference. The current model takes a 3-D input
                # (batch, time, WORD_FEATURE_DIM). Reject anything that is not a
                # sequence model OR has a different feature width (e.g. an older
                # 63-value one-hand model, or the legacy 2-D mean+std model).
                incompatible = (
                    len(candidate.input_shape) != 3
                    or candidate.input_shape[-1] != _FEATURE_DIM
                )
                if incompatible:
                    print(
                        f"  [WARN] The word model is incompatible (input "
                        f"{candidate.input_shape}; expected (None, {_SEQ_LEN}, "
                        f"{_FEATURE_DIM})). It predates the body-anchored "
                        "two-hand features.\n"
                        "         Re-capture and re-train: python capture/capture_words.py "
                        "then python training/train_words.py.\n"
                        "         Word mode is disabled until then; letters still work."
                    )
                else:
                    self._model_words  = candidate
                    self._labels_words = labels_candidate

        # The numbers model is optional and shares the letter pipeline (single
        # frame, 63 values). It stays None until the team trains 0-9; the app
        # then shows a clear notice in numbers mode instead of faking letters.
        self._model_numbers  = None
        self._labels_numbers = None
        if os.path.isfile(_MODEL_NUMBERS) and os.path.isfile(_LABELS_NUMBERS):
            labels_candidate = self._load_labels(_LABELS_NUMBERS)
            if len(labels_candidate) < 2:
                print(
                    f"  [WARN] The numbers model only has {len(labels_candidate)} class. "
                    "At least 2 are needed; ignoring it."
                )
            else:
                self._model_numbers  = tf.keras.models.load_model(_MODEL_NUMBERS, compile=False)
                self._labels_numbers = labels_candidate

        # Motion level of the last analyzed frame (updated by _run_words).
        # Exposed in the classify() result as "hand_is_signing".
        self._last_motion = 0.0

        # The first TF forward pass compiles the XLA graph (200-800ms). We run
        # it here with zeros so the first real frame is instant.
        self._warmup()

        n_one     = len(self._labels_one)
        n_words   = len(self._labels_words) if self._labels_words else 0
        n_numbers = len(self._labels_numbers) if self._labels_numbers else 0
        print(f"Classifier ready  |  letters: {n_one}  |  words: {n_words}  |  numbers: {n_numbers}")

    @property
    def has_words(self) -> bool:
        return self._model_words is not None

    @property
    def has_numbers(self) -> bool:
        return self._model_numbers is not None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def classify(self, landmarks_data, static_mode="letters"):
        """
        Classify one frame.

        static_mode selects which single-frame model produces the static
        prediction: "letters" (default) or "numbers". The word model runs
        regardless, since it also drives the "hand_is_signing" motion gate.
        """
        num_hands = landmarks_data["num_hands"]
        h1 = landmarks_data["landmarks_hand1"]

        if num_hands == 0 or h1 is None:
            self._word_buffer.clear()
            self._last_motion = 0.0
            return self._empty()

        # Update the word buffer with the current frame (needs both hands + the
        # body anchor, so it takes the whole landmarks dict, not just hand1).
        # _last_motion is updated internally in _run_words().
        word_pred = self._update_word_buffer(landmarks_data)

        # Static classification with the model selected by the active mode.
        if static_mode == "numbers":
            if self._model_numbers is not None:
                result = self._run(self._model_numbers, self._labels_numbers,
                                   h1, "numbers", _NUMBER_THRESHOLD)
            else:
                # Numbers mode requested but no model trained yet. Be honest:
                # produce an empty static result so the HUD can show a notice
                # rather than misleadingly showing letters.
                result = {"prediction": None, "confidence": 0.0,
                          "model_used": None, "top3": []}
        else:
            result = self._run(self._model_one, self._labels_one,
                               h1, "one_hand", _LETTER_THRESHOLD)

        result["word_prediction"] = word_pred
        # True only if the word model is loaded AND the hand moves enough.
        # When True, main.py suppresses letter predictions.
        result["hand_is_signing"] = self._last_motion >= _MIN_MOTION_STD
        return result

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _run(self, model, labels, flat, model_name, threshold):
        normalized = normalize_landmarks(flat)
        tensor = tf.constant([normalized], dtype=tf.float32)

        # Calling the model directly is 3-5x faster than model.predict()
        # for single samples in real time.
        probs = model(tensor, training=False).numpy()[0]

        # Ranked candidates, best first. We keep up to the top 3 (filtering out
        # near-zero noise) so the HUD can show alternatives when unsure.
        order = np.argsort(probs)[::-1]
        top3  = [
            {"prediction": labels[int(i)], "confidence": float(probs[int(i)])}
            for i in order[:3]
            if float(probs[int(i)]) >= _ALT_MIN_CONFIDENCE
        ]

        top1_conf = float(probs[int(order[0])])
        # `prediction` is only committed above the acceptance threshold; the
        # raw ranking in top3 is returned regardless of confidence.
        accepted  = top1_conf >= threshold

        return {
            "prediction": labels[int(order[0])] if accepted else None,
            "confidence": top1_conf,
            "model_used": model_name,
            "top3":       top3,
        }

    def _update_word_buffer(self, landmarks_data):
        if self._model_words is None:
            return None
        # Body-anchored, two-hand feature (slots by handedness). Hands are taken
        # from the handedness slots; the body frame from the shoulders.
        by_side = landmarks_data["hands_by_side"]
        feat = build_word_features(
            by_side["Left"], by_side["Right"],
            landmarks_data["shoulder_l"], landmarks_data["shoulder_r"],
        )
        self._word_buffer.append(feat)
        if len(self._word_buffer) < _MIN_FRAMES:
            self._last_motion = 0.0
            return None
        return self._run_words()

    def _run_words(self):
        arr = np.array(list(self._word_buffer), dtype=np.float32)  # (<=BUFFER, FEATURE_DIM)

        # Motion over only the most recent frames: detects quickly when the
        # user stops the hand. Averaging the whole buffer would keep a motion
        # from a moment ago counting, delaying the return to letters.
        recent = arr[-_MOTION_WINDOW:]
        motion = float(recent.std(axis=0).mean())
        self._last_motion = motion

        if motion < _MIN_MOTION_STD:
            return None

        # Resample the variable-length buffer to the fixed sequence length the
        # model was trained on (same helper capture used), then classify the
        # ordered sequence. This is the temporal step the old mean+std lacked.
        seq    = resample_sequence(list(self._word_buffer), _SEQ_LEN)  # (T, 63)
        tensor = tf.constant(seq[np.newaxis], dtype=tf.float32)        # (1, T, 63)
        probs  = self._model_words(tensor, training=False).numpy()[0]
        idx    = int(np.argmax(probs))
        conf   = float(probs[idx])
        if conf < WORD_CONFIDENCE_THRESHOLD:
            return None

        label = self._labels_words[idx]
        # The negative class ("nothing") means "the model thinks this is not a
        # deliberate word" — we treat that prediction as None.
        if label == _NULL_WORD_LABEL:
            return None

        return {"prediction": label, "confidence": conf}

    def _warmup(self):
        self._model_one(tf.zeros((1, 63), dtype=tf.float32), training=False)
        if self._model_words is not None:
            self._model_words(tf.zeros((1, _SEQ_LEN, _FEATURE_DIM), dtype=tf.float32), training=False)
        if self._model_numbers is not None:
            self._model_numbers(tf.zeros((1, 63), dtype=tf.float32), training=False)

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
            "top3":            [],
            "word_prediction": None,
            "hand_is_signing": False,
        }
