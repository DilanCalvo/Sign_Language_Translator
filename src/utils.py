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


def normalize_landmarks(flat, aspect=None):
    """
    Normalize one landmark sample.

    Accepts:
        flat:   array/list of 63 floats (one hand) or 126 floats (two hands).
                Each hand is 21 consecutive points (x, y, z).
        aspect: frame width / height the landmarks came from, or None.
                MediaPipe normalizes x by the frame WIDTH and y by the frame
                HEIGHT (z scales like x, per the official docs), so the same
                physical pose yields a differently-stretched vector on a 16:9
                webcam vs a 9:16 portrait phone. When aspect is given, the
                stretch is undone ("width units": y /= aspect; x and z
                untouched) BEFORE centering/scaling, making the result
                orientation-invariant. None skips the correction — only
                legitimate for the word pipeline, whose stored training
                features predate it.

    Returns:
        np.ndarray float32 of the same size, already normalized.

    Per-hand process:
        1. Undo MediaPipe's per-axis stretch (when aspect is given).
        2. Center the 21 points by subtracting the wrist position.
        3. Scale by the wrist -> middle-finger-base distance. This makes the
           model robust to large/small hands and to hands near/far from the
           camera.
    """
    arr = np.asarray(flat, dtype=np.float32)

    if arr.size == 63:
        return _normalize_single(arr, aspect)

    if arr.size == 126:
        h1 = _normalize_single(arr[:63], aspect)
        h2 = _normalize_single(arr[63:], aspect)
        return np.concatenate([h1, h2]).astype(np.float32)

    raise ValueError(
        f"Unsupported number of values: {arr.size}. Expected 63 (one hand) or 126 (two hands)."
    )


def _normalize_single(flat63, aspect=None):
    points = flat63.reshape(21, 3)
    if aspect is not None:
        # Copy: the un-stretch must not mutate the caller's array (loaders pass
        # views into a shared matrix). It commutes with the centering below but
        # NOT with the scale norm, so it must happen first.
        points = points.copy()
        points[:, 1] /= np.float32(aspect)
    centered = points - points[_WRIST]
    scale = np.linalg.norm(centered[_MIDDLE_MCP])
    if scale < 1e-6:
        scale = 1.0
    return (centered / scale).flatten().astype(np.float32)


# ---------------------------------------------------------------------------
# Word feature builder — body-anchored, two-hand. SINGLE SOURCE OF TRUTH for the
# word pipeline (capture, training and inference must all build features here,
# the same lesson as normalize_landmarks). See config.WORD_FEATURE_DIM for the
# layout. Letters do NOT use this — they keep the plain one-hand normalize.
# ---------------------------------------------------------------------------

_HAND_SHAPE_LEN = 63          # 21 landmarks x (x, y, z)
_HAND_BLOCK_LEN = 65          # shape (63) + wrist position (x, y)
WORD_FEATURE_LEN = 130        # left block (65) + right block (65)

# Raw capture-frame layout (what capture_words.py stores on disk, packed by
# pack_word_raw / unpacked by word_features_from_raw — both HERE so the layout
# has exactly one source of truth):
#   left hand (63) + right hand (63) + shoulder_l (x, y) + shoulder_r (x, y)
#   + frame aspect (1)  =  131
# Storing RAW landmarks instead of processed features is deliberate: it lets a
# future normalization change (like the 2026-07 aspect fix) be applied
# retroactively at load time instead of forcing a full vocabulary recapture.
# Missing hand / missing shoulders are stored as zeros — MediaPipe never emits
# an exact all-zeros block, so zeros unambiguously mean "absent".
WORD_RAW_FRAME_LEN = 131


def _hand_block(hand_flat, frame, aspect):
    """
    Build one hand's 65-value block: 63 handshape + 2 body-relative wrist pos.

    hand_flat: list/array of 63 raw image-normalized landmarks, or None.
    frame: (cx, cy, scale) body frame IN WIDTH UNITS (already un-stretched),
           or None when pose was not detected.
    aspect: frame width/height — un-stretches the handshape and the raw wrist y
            so the block is camera-orientation invariant.

    A missing hand returns zeros — a distinct pattern the model reads as absent.
    When there is no body frame, the position part is 0 (NOT the raw screen
    coordinate): without an anchor we have no position signal, and leaking the
    absolute on-screen position would reintroduce exactly the distractor the
    body anchor exists to remove.
    """
    if hand_flat is None:
        return np.zeros(_HAND_BLOCK_LEN, dtype=np.float32)
    arr = np.asarray(hand_flat, dtype=np.float32)
    shape = _normalize_single(arr, aspect)         # 63, un-stretched + wrist-centered + scaled
    if frame is None:
        pos_x = pos_y = 0.0
    else:
        cx, cy, scale = frame
        wrist = arr.reshape(21, 3)[_WRIST]         # raw wrist x, y, z
        pos_x = (wrist[0] - cx) / scale
        pos_y = (wrist[1] / np.float32(aspect) - cy) / scale
    return np.concatenate([shape, [pos_x, pos_y]]).astype(np.float32)


def build_word_features(left_hand, right_hand, shoulder_l, shoulder_r, aspect):
    """
    Build the full (WORD_FEATURE_LEN,) word feature vector for one frame.

    Args:
        left_hand, right_hand: 63-value raw landmark lists, or None if that hand
            is not present. Slots are by MediaPipe handedness so the same sign
            always lands in the same slot.
        shoulder_l, shoulder_r: (x, y) of the left/right shoulder, or None if
            pose was not detected (then there is no body anchor: the wrist
            position is 0, the handshape still works).
        aspect: frame width/height. REQUIRED — MediaPipe normalizes x by frame
            width and y by height, so without undoing that stretch the same
            physical sign yields different vectors on a 16:9 webcam vs a 9:16
            portrait phone (the bug that forced the 2026-07 vocabulary
            recapture). Every caller has it: the detector emits it per frame
            (landmarks_data["frame_aspect"]) and raw capture rows store it.
            Made mandatory (not defaulted) so a forgotten call site fails
            loudly instead of silently reintroducing the geometry bug — the
            classifier's stale-model guard is width-based and cannot catch it.

    Returns:
        np.ndarray float32 of shape (WORD_FEATURE_LEN,).
    """
    if aspect is None:
        raise ValueError(
            "build_word_features requires the frame aspect ratio (width/height). "
            "Pass landmarks_data['frame_aspect'] (live) or the raw row's stored "
            "aspect (training)."
        )

    # Body frame in WIDTH UNITS: un-stretch every y before any geometry, so the
    # anchor (center + shoulder-width scale) is camera-orientation invariant.
    # hypot over both axes (not |dx| alone) keeps the scale invariant to head/
    # torso roll; after un-stretching, shoulder-y jitter enters it only at
    # ~sin(roll) weight — second order.
    frame = None
    if shoulder_l is not None and shoulder_r is not None:
        a = np.float32(aspect)
        cx = (shoulder_l[0] + shoulder_r[0]) * 0.5
        cy = (shoulder_l[1] / a + shoulder_r[1] / a) * 0.5
        scale = float(np.hypot(shoulder_l[0] - shoulder_r[0],
                               (shoulder_l[1] - shoulder_r[1]) / a))
        if scale < 1e-6:
            scale = 1.0
        frame = (cx, cy, scale)

    return np.concatenate([
        _hand_block(left_hand,  frame, aspect),
        _hand_block(right_hand, frame, aspect),
    ]).astype(np.float32)


def pack_word_raw(left_hand, right_hand, shoulder_l, shoulder_r, aspect):
    """
    Pack one frame's RAW capture data into a (WORD_RAW_FRAME_LEN,) float32 row
    (the on-disk format of capture_words.py — see the layout comment above).
    Missing hand/shoulders are stored as zeros; word_features_from_raw maps
    them back to None.
    """
    row = np.zeros(WORD_RAW_FRAME_LEN, dtype=np.float32)
    if left_hand is not None:
        row[0:63] = np.asarray(left_hand, dtype=np.float32)
    if right_hand is not None:
        row[63:126] = np.asarray(right_hand, dtype=np.float32)
    if shoulder_l is not None:
        row[126:128] = shoulder_l
    if shoulder_r is not None:
        row[128:130] = shoulder_r
    row[130] = aspect
    return row


def word_features_from_raw(row):
    """
    Turn one stored raw capture row back into the (WORD_FEATURE_LEN,) feature
    vector — the load-time counterpart of pack_word_raw. This is where the
    training loader applies the CURRENT normalization to old raw data, which is
    the whole point of storing raw: feature changes never invalidate captures.
    """
    row = np.asarray(row, dtype=np.float32)
    if row.size != WORD_RAW_FRAME_LEN:
        raise ValueError(
            f"Raw word frame must have {WORD_RAW_FRAME_LEN} values, got {row.size}."
        )
    left  = row[0:63]   if np.any(row[0:63])   else None
    right = row[63:126] if np.any(row[63:126]) else None
    sh_l  = tuple(row[126:128]) if np.any(row[126:128]) else None
    sh_r  = tuple(row[128:130]) if np.any(row[128:130]) else None
    return build_word_features(left, right, sh_l, sh_r, float(row[130]))


def mirror_word_sequence(seq) -> np.ndarray:
    """
    Horizontal mirror of a word-feature sequence (T, WORD_FEATURE_LEN).

    Mirroring a sign swaps left/right: a right-handed sign becomes the same sign
    performed left-handed. So we (1) negate every x coordinate and (2) swap the
    two hand blocks. Used as training augmentation — doubles the data and makes
    the model handedness-robust.
    """
    seq = np.asarray(seq, dtype=np.float32)
    left  = _mirror_hand_block(seq[:, :_HAND_BLOCK_LEN])
    right = _mirror_hand_block(seq[:, _HAND_BLOCK_LEN:])
    # Swap: mirrored-left becomes the right hand and vice versa.
    return np.concatenate([right, left], axis=1).astype(np.float32)


def _mirror_hand_block(block):
    """Negate x of the handshape (every 3rd value) and of the wrist position."""
    b = np.asarray(block, dtype=np.float32).copy()
    b[:, 0:_HAND_SHAPE_LEN:3] *= -1.0   # x of each of the 21 shape landmarks
    b[:, _HAND_SHAPE_LEN] *= -1.0       # wrist position x (index 63)
    return b


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
    guaranteeing identical (n, D) shapes everywhere.

    Args:
        frames: sequence of (D,) vectors (here D = WORD_FEATURE_DIM word features).
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
