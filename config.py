"""
Central configuration for the Sign Language Translator.

This file is the SINGLE source of truth for every parameter that affects the
system's behavior. Change a value here and it propagates to all modules
without touching any other file.

Sections:
  1. ACTIVE MODE       — which model the user sees right now
  2. CONFIDENCE        — when a prediction is accepted or rejected
  3. MOTION            — how a hand is detected to be signing
  4. SMOOTHING         — how long a prediction takes to appear/disappear
  5. CAPTURE           — parameters used when recording new samples
  6. PATHS             — model and label locations on disk
  7. CAMERA            — hand-detector configuration
"""

# ================================================================
#  1. ACTIVE MODE
# ================================================================

# What kind of communication the user is expected to produce.
#
#   "letters"  → Detects ASL alphabet letters (A-Z). The word model runs
#                internally but its result is not shown. Best for spelling.
#
#   "words"    → Detects dynamic signs / whole words (hola, adios, ...).
#                Letters are suppressed. Best for fluent communication with
#                a known vocabulary.
#
#   "numbers"  → Reserved. For now it behaves like "letters". It will be
#                enabled once a numbers model exists.
#
# Change this value to switch modes until a keyboard shortcut is exposed in
# the UI.

MODE = "letters"


# ================================================================
#  2. CONFIDENCE — prediction acceptance thresholds
# ================================================================

# Minimum confidence required to show a LETTER.
#
# The model returns a probability between 0 and 1. The letter is shown only
# if it passes this threshold.
#
#   Raise → fewer but more precise predictions. Good if there are many false
#           positives (letters appearing without intent).
#   Lower → more predictions but more noise. Useful if the trained model has
#           low confidences (e.g. few samples).
#
# Recommended range: 0.70 - 0.90. Calibrated default: 0.80.

LETTER_CONFIDENCE_THRESHOLD = 0.80

# Minimum confidence required to show a WORD.
#
# The word model has a softmax over N classes; the more classes, the lower
# the natural peak probabilities. With 3 classes the real ceiling is
# ~0.65-0.70 (dropout + regularization suppress it).
#
#   Raise → stricter. With the current model (few classes) values > 0.70
#           reject almost everything. Only raise it with >5 classes and many
#           false positives.
#   Lower → accepts less certain predictions. Lower to 0.40 if a new, larger
#           model produces lower confidences.
#
# Calibrate with: python training/evaluate.py (the "threshold sweep" table).
# Empirically calibrated default for a small word model: 0.50.

WORD_CONFIDENCE_THRESHOLD = 0.50


# ================================================================
#  3. MOTION — dynamic-sign detection
# ================================================================

# Number of frames accumulated to analyze a complete sign.
#
# CRITICAL WARNING: this value MUST be identical to the one used when
# capturing the data (CAPTURE_SEQ_FRAMES) and when training the model.
# Changing it here without re-capturing and re-training breaks the model.
#
#   Raise → analyzes longer gestures. Needed for slow signs or vocabulary
#           with complex movements.
#   Lower → faster response but less temporal context. Only useful for very
#           short signs and a small vocabulary.
#
# Default: 20 frames ~= 0.67s at 30fps.

WORD_SEQ_FRAMES = 20

# Number of recent frames used ONLY to decide whether the hand is moving
# right now (does not affect the model features).
#
# A short window is used to react quickly when the user stops the hand. Using
# the full buffer (20 frames) would keep a motion from 0.5s ago "activating"
# the signing state for too long, delaying letters.
#
#   Raise → slower to turn the signing state "off"; reduces false offs when
#           the user pauses briefly inside a gesture.
#   Lower → faster to return to letters; may turn off prematurely on signs
#           with pauses.
#
# Default: 8 frames ~= 0.27s at 30fps.

WORD_MOTION_WINDOW = 8

# Minimum motion standard deviation to activate the word model.
#
# Computed as the average std of each landmark coordinate over the last
# WORD_MOTION_WINDOW normalized frames.
#
#   Static pose with tremor:  std ~ 0.005 - 0.015
#   Transition between poses: std ~ 0.015 - 0.025   (ambiguous zone)
#   Clear dynamic sign:       std ~ 0.025 - 0.080
#
#   Raise → stricter. Reduces activations from moving letters or tremor.
#           Raise it if letters wrongly trigger words.
#   Lower → more sensitive. Detects subtle signs. Lower it if low-movement
#           signs are not detected.
#
# Default: 0.020 (clean static vs dynamic separation).

WORD_MIN_MOTION_STD = 0.020

# Negative-class label of the word model.
#
# When the model predicts this class it is read as "this is not a deliberate
# word" and None is returned instead of showing it. It must match exactly the
# name used when capturing the negative-class data and when training.
#
# NOTE: this string is data-bound. It must match a label inside
# model/labels_words.json. Do not change it without re-capturing data with
# the new name and re-training.

WORD_NULL_LABEL = "nada"


# ================================================================
#  4. SMOOTHING — on-screen stability and responsiveness
# ================================================================

# LETTER smoother window (frames considered for voting).
#
#   Raise → more stable prediction; slower to change letter. Reduces flicker
#           between similar letters (P/Q, U/V...).
#   Lower → faster response; may flicker more.
#
# Default: 7 frames ~= 0.23s at 30fps.

LETTER_SMOOTH_WINDOW = 7

# Minimum votes within the window to confirm a LETTER.
#
# Must be <= LETTER_SMOOTH_WINDOW. A letter must appear at least this many
# times within the last LETTER_SMOOTH_WINDOW frames.
#
#   Raise → stricter; the letter takes longer to stabilize.
#   Lower → looser; the letter appears sooner but may be noise.
#
# Rule of thumb: min_votes ~= window * 0.70.
# Default: 5 of 7.

LETTER_SMOOTH_MIN_VOTES = 5

# WORD smoother window.
#
# The word model already averages 20 frames internally, so a small window is
# enough to filter spurious predictions.
#
#   Raise → more stable but slower to confirm the sign.
#   Lower → faster response; increases false-positive risk.
#
# Default: 5 frames.

WORD_SMOOTH_WINDOW = 5

# Minimum votes within the window to confirm a WORD.
#
# Default: 3 of 5.

WORD_SMOOTH_MIN_VOTES = 3

# Lock-out frames after a word is detected.
#
# Once a sign is confirmed, new predictions are locked for this many frames.
# The detected word stays visible. This prevents the same gesture from
# repeating in bursts and gives the user time to move to the next sign.
#
#   Raise → the same sign is shown longer; harder to repeat.
#   Lower → faster cycle; may detect the same gesture twice in a row if the
#           user does not move the hand.
#
# Default: 45 frames ~= 1.5s at 30fps.

WORD_COOLDOWN_FRAMES = 45

# ASL alphabet letters that require motion to be correct.
#
# These letters cannot be detected from a single static frame (J draws a
# curve in the air; Z traces a Z). The system shows them with a "requires
# motion" warning.
#
# If a dedicated motion model for letters is trained in the future, these
# could be removed from the set.

MOTION_LETTERS = {"J", "Z"}


# ================================================================
#  5. CAPTURE — parameters used when recording new word samples
# ================================================================

# Number of sequences to record per word in capture_words.py.
#
#   Raise → more data, better model generalization. Recommended: >=100 for
#           production, >=50 for quick tests.
#   Lower → faster capture; useful for test sessions.
#
# Default: 100 sequences per word.

CAPTURE_TARGET_PER_WORD = 100

# Frames per sequence when capturing.
#
# CRITICAL WARNING: MUST equal WORD_SEQ_FRAMES. If they differ, the captured
# features will have a different dimension than the model expects and
# training will fail.
#
# It exists separately only for clarity in capture_words.py; internally it
# points to the same value as WORD_SEQ_FRAMES.

CAPTURE_SEQ_FRAMES = WORD_SEQ_FRAMES

# Directory where word-capture CSVs are saved.

CAPTURE_OUTPUT_DIR = "data/real_capture/words"


# ================================================================
#  6. PATHS — model locations on disk
# ================================================================
# You rarely need to change these. Only if you reorganize the project or use
# alternative models for experiments.

MODEL_ONE_HAND_PATH   = "model/model_one_hand.h5"
MODEL_WORDS_PATH      = "model/model_words.h5"

LABELS_ONE_HAND_PATH  = "model/labels_one_hand.json"
LABELS_WORDS_PATH     = "model/labels_words.json"

HAND_LANDMARKER_PATH  = "model/hand_landmarker.task"


# ================================================================
#  7. CAMERA — hand-detector configuration (MediaPipe)
# ================================================================

# Camera index to use. 0 = system's primary camera. Change to 1, 2... if you
# have several cameras and want a different one.

CAMERA_INDEX = 0

# Minimum confidence to detect a hand for the first time in the frame.
#
#   Raise → ignores doubtful detections; fewer initial false positives.
#   Lower → detects hands more easily in low light.
#
# Default: 0.70.

DETECTOR_MIN_DETECTION_CONFIDENCE = 0.70

# Minimum confidence to keep a detected hand on the following frames.
#
#   Raise → drops hands quickly when uncertain.
#   Lower → keeps tracking even if the hand is partially hidden.
#
# Default: 0.70.

DETECTOR_MIN_PRESENCE_CONFIDENCE = 0.70

# Minimum tracking confidence between frames (how much to trust that the hand
# stays in the same place without fully re-detecting it).
#
#   Raise → re-detects more often; more precise but slower.
#   Lower → trusts tracking more; smoother but may miss abrupt position
#           changes.
#
# Default: 0.50.

DETECTOR_MIN_TRACKING_CONFIDENCE = 0.50
