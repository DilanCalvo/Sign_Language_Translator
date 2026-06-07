"""
Shared utilities used across capture, training and inference.

  normalize_landmarks  — turns raw landmarks into a position/scale invariant
                         representation.
  PredictionSmoother   — sliding-window filter that stabilizes predictions.
"""

from collections import Counter, deque

import numpy as np

_WRIST = 0          # wrist landmark (used as the origin)
_MIDDLE_MCP = 9     # base of the middle finger (used as the scale reference)


def normalize_landmarks(flat):
    """
    Normalize one landmark sample.

    Accepts:
        flat: array/list of 63 floats (one hand) or 126 floats (two hands).
              Each hand is 21 consecutive points (x, y, z).

    Returns:
        np.ndarray float32 of the same size, already normalized.

    Per-hand process:
        1. Center the 21 points by subtracting the wrist position.
        2. Scale by the wrist -> middle-finger-base distance. This makes the
           model robust to large/small hands and to hands near/far from the
           camera.
    """
    arr = np.asarray(flat, dtype=np.float32)

    if arr.size == 63:
        return _normalize_single(arr)

    if arr.size == 126:
        h1 = _normalize_single(arr[:63])
        h2 = _normalize_single(arr[63:])
        return np.concatenate([h1, h2]).astype(np.float32)

    raise ValueError(
        f"Unsupported number of values: {arr.size}. Expected 63 (one hand) or 126 (two hands)."
    )


def _normalize_single(flat63):
    points = flat63.reshape(21, 3)
    centered = points - points[_WRIST]
    scale = np.linalg.norm(centered[_MIDDLE_MCP])
    if scale < 1e-6:
        scale = 1.0
    return (centered / scale).flatten().astype(np.float32)


# ---------------------------------------------------------------------------
# Temporal resampling — single source of truth for capture, training and
# inference (the same lesson as normalize_landmarks: if they disagree, the
# model sees different shapes at train vs run time).
# ---------------------------------------------------------------------------

def resample_sequence(frames, n: int) -> np.ndarray:
    """
    Resample a list/array of equal-length frame vectors to exactly `n` frames,
    evenly spaced over the original timeline.

    A recorded sign has a variable number of frames (it depends on how fast it
    was signed). The temporal word model needs a fixed length, so every take is
    resampled to WORD_SEQ_LEN here — at capture time AND at inference time —
    guaranteeing identical (n, 63) shapes everywhere.

    Args:
        frames: sequence of (D,) vectors (here D = 63 normalized landmarks).
        n:      target number of frames.

    Returns:
        np.ndarray float32 of shape (n, D).
    """
    frames = list(frames)
    if not frames:
        raise ValueError("Cannot resample an empty sequence.")
    idx = np.linspace(0, len(frames) - 1, n).round().astype(int)
    return np.stack([frames[i] for i in idx]).astype(np.float32)


def mirror_sequence(seq) -> np.ndarray:
    """
    Horizontal mirror of a one-hand landmark sequence: negate every X
    coordinate. A right-handed sign mirrored looks like the same sign performed
    left-handed, so this doubles the data for free and makes the model
    handedness-robust. Used as training augmentation (kept here so training and
    any future test-time augmentation share one definition).

    Args:
        seq: array of shape (T, 63).

    Returns:
        np.ndarray float32 of shape (T, 63) with X coordinates negated.
    """
    seq = np.asarray(seq, dtype=np.float32).copy()
    seq[:, 0::3] *= -1.0   # x is every 3rd value (x, y, z per landmark)
    return seq


# ---------------------------------------------------------------------------
# Temporal smoothing of predictions
# ---------------------------------------------------------------------------

class PredictionSmoother:
    """
    Sliding window of predictions that removes single-frame flashes.

    Usage:
        smoother = PredictionSmoother()
        smoother.update(raw_prediction)   # call every frame
        stable = smoother.get_stable()    # confirmed letter or None

    A prediction is considered stable when it appears at least `min_votes`
    times within the most recent `window` frames. None predictions (no hand
    or low confidence) count against stability, which is the desired
    behavior: if the hand leaves the frame the system waits until there is
    fresh consensus.
    """

    def __init__(self, window: int = 7, min_votes: int = 5):
        self._buffer    = deque(maxlen=window)
        self._min_votes = min_votes

    def update(self, prediction) -> None:
        self._buffer.append(prediction)

    def get_stable(self):
        """Return the winning prediction if it passes min_votes, else None."""
        if not self._buffer:
            return None
        valid = [p for p in self._buffer if p is not None]
        if not valid:
            return None
        top_pred, top_count = Counter(valid).most_common(1)[0]
        return top_pred if top_count >= self._min_votes else None

    def reset(self) -> None:
        """Clear the window. Call when the user starts a new sign."""
        self._buffer.clear()

    @property
    def raw(self):
        """Last received prediction, without smoothing."""
        return self._buffer[-1] if self._buffer else None
