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
#   "words"    → Detects dynamic signs / whole words (ASL glosses: yes, no,
#                want, ...). Letters are suppressed. Best for fluent
#                communication with a curated vocabulary. The recognized glosses
#                can be turned into a fluent sentence by the LLM translation
#                layer (see section 10 and src/translator.py).
#
#   "numbers"  → Detects ASL digits 0-9 (static poses, same pipeline as
#                letters). Uses model/model_numbers.h5. If that model is not
#                trained yet, the app shows a clear "numbers model not trained"
#                notice instead of silently falling back to letters.
#
# Change this value to switch modes; L / W / N also switch at runtime.

MODE = "words"


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

# Minimum confidence required to show a NUMBER (0-9).
#
# Numbers are static poses classified exactly like letters (single frame,
# 63 values), so the same calibration logic applies. Kept separate so it can
# be tuned independently once a numbers model is trained.

NUMBER_CONFIDENCE_THRESHOLD = 0.60

# Low-confidence boundary for the "show alternatives" UI.
#
# This is INDEPENDENT of the acceptance thresholds above:
#   - confidence >= LETTER/NUMBER threshold  -> the prediction is committed
#     (added to the subtitle buffer and spoken).
#   - confidence <  LOW_CONFIDENCE_THRESHOLD -> the model is unsure, so the
#     HUD shows the top-3 candidates with their percentages instead of a
#     single guess, letting the user see what the model is hesitating between.
#   - in between -> the raw top-1 guess is shown small, but not committed.
#
# This decouples "what we trust enough to act on" from "what hints we surface".
# Lower it to show alternatives less often; raise it to be more cautious.

LOW_CONFIDENCE_THRESHOLD = 0.70

# Minimum confidence for a candidate to appear in the alternatives list.
#
# Candidates below this are noise and would clutter the HUD. Replaces the
# previously hardcoded runner-up cutoff.

ALT_MIN_CONFIDENCE = 0.08

# Minimum confidence required to commit a WORD.
#
# The word model has a softmax over N classes (including the negative
# "nothing" class). The negative class absorbs the probability mass of
# ambiguous frames, so genuine signs peak higher and false positives are
# already filtered by the negative class — a moderate threshold works well.
#
#   Raise → stricter; fewer false positives, may drop borderline real signs.
#   Lower → accepts less certain predictions; raise if you see misfires.
#
# Default: 0.60 (TCN sequence model with a trained negative class).

WORD_CONFIDENCE_THRESHOLD = 0.75


# ================================================================
#  3. MOTION & SEQUENCE — dynamic-sign detection
# ================================================================

# Length (in frames) of the sequence the WORD model consumes.
#
# The word model is a TEMPORAL model (TCN, 1D convolutions over time) — it
# reads the ordered sequence of normalized frames, NOT a single mean+std vector.
# This is the key change over the old model: mean+std is order-blind (it cannot
# tell "hand goes up then down" from "down then up"), which collapses as the
# vocabulary grows. Order matters in ASL, so the model must see the sequence.
#
# Recording length varies (a sign can take 0.5-1.5s); both capture and live
# inference RESAMPLE whatever was recorded to exactly this many frames
# (src.utils.resample_sequence) so every sample has the same shape (T, 63).
#
# CRITICAL WARNING: this value MUST be identical in capture, training and
# inference. It is imported everywhere from here — never hardcode it.
#
#   Raise → more temporal detail; better for complex/long signs, slower.
#   Lower → faster, less detail.
#
# Default: 32 frames (empirically validated; ~1s of signing resampled to 32).

WORD_SEQ_LEN = 32

# Number of values per frame in the WORD feature vector (body-anchored).
#
# Unlike letters (one hand, 63 values = 21 landmarks x,y,z), the word model sees
# BOTH hands AND where each hand is relative to the body. This lets it tell apart
# signs with the same handshape at different body locations (e.g. a hand at the
# chest vs. at the forehead) and signs that use two hands.
#
# Layout per frame (built by src.utils.build_word_features):
#   [ left  hand: 63 shape + 2 wrist position ]  = 65
#   [ right hand: 63 shape + 2 wrist position ]  = 65
#                                          total = 130
#
#   - "shape" = the 21 landmarks centered on the wrist and scaled by hand size
#     (the same representation letters use — pure handshape, position-invariant).
#   - "wrist position" = the wrist's (x, y) relative to the shoulder center,
#     scaled by shoulder width. This is the BODY ANCHOR — invariant to where the
#     person stands or how big they are. z is omitted (MediaPipe depth is noisy).
#   - A missing hand (one-handed sign, or hand out of frame) is all zeros, a
#     pattern the model learns to read as "that hand is absent".
#
# CRITICAL: this MUST match in capture, training and inference (imported here).
# Changing it requires re-capturing and re-training.

WORD_FEATURE_DIM = 130

# Max length of the rolling live buffer the classifier keeps while you sign.
#
# Every frame with a hand present is appended; once it holds at least
# WORD_MIN_FRAMES, the buffer is resampled to WORD_SEQ_LEN and classified. A
# value a bit larger than WORD_SEQ_LEN gives the resampler some slack so a
# slightly long sign is not clipped.
#
# Default: 45 frames ~= 1.5s at 30fps.

WORD_BUFFER_FRAMES = 45

# Minimum frames in the live buffer before the word model runs.
#
# Below this there is not enough of a gesture to classify reliably; the
# classifier returns no word prediction until the buffer fills to here.
#
# Default: 16 frames ~= 0.53s at 30fps.

WORD_MIN_FRAMES = 16

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
# Without this class a closed-set softmax labels EVERYTHING as some word (rest,
# transitions, moving letters) and never stays silent — the single most common
# cause of "it's always saying something". Capture plenty of varied non-sign
# takes for it (capture/capture_words.py rotates prompts to force variety).
#
# NOTE: this string is data-bound. It must match a label inside
# model/labels_words.json. Do not change it without re-capturing data with
# the new name and re-training. (English now, to match the ASL gloss vocabulary.)

WORD_NULL_LABEL = "nothing"


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

# Lock-out TIME after a word is detected (seconds).
#
# Once a sign is confirmed, new predictions are locked for this long. The
# detected word stays visible. This prevents the same gesture from repeating in
# bursts and gives the user a moment to move to the next sign.
#
# Time-based (not frames) on purpose: the real wait must not depend on the frame
# rate, which varies with the machine and with how many models run per frame
# (the pose model lowers FPS). At 1.0s the next sign can be made about a second
# after the previous one is recognized.
#
#   Raise → the same sign is shown longer; harder to repeat by accident.
#   Lower → faster cadence; may catch the same gesture twice if you hold it.
#
# Default: 1.0 second.

WORD_COOLDOWN_SECONDS = 2.0

# Frames of inactivity before the accumulated WORD sentence clears itself.
#
# In word mode, detected signs accumulate into a running sentence shown at the
# bottom (e.g. "yo pensar"). The sentence resets on its own after this many
# frames with no new sign, so the user just pauses to start a fresh sentence
# (no key needed).
#
#   Raise → the sentence lingers longer; good for slow signers.
#   Lower → clears sooner; risk of cutting a sentence mid-thought.
#
# Note: the per-word cooldown (WORD_COOLDOWN_SECONDS) also runs after each sign,
# so the effective gap the user has between signs includes it.
# Default: 150 frames ~= 5s at 30fps.

WORD_SENTENCE_PAUSE_FRAMES = 150

# ASL alphabet letters that require motion to be correct.
#
# These letters cannot be detected from a single static frame (J draws a
# curve in the air; Z traces a Z). The system shows them with a "requires
# motion" warning.
#
# If a dedicated motion model for letters is trained in the future, these
# could be removed from the set.

MOTION_LETTERS = {"J", "Z"}

# Minimum frames between letter additions to the overlay buffer.
#
# Once a letter is accepted into the subtitle strip, no new letter (even a
# different one) is accepted until this many frames pass. This prevents a
# single held pose from spamming the buffer.
#
#   Raise → slower accumulation; good if letters pile up too fast.
#   Lower → faster accumulation; may repeat the same letter.
#
# Default: 20 frames ~= 0.67s at 30fps.

LETTER_COOLDOWN_FRAMES = 20


# ================================================================
#  5. CAPTURE — parameters used when recording new word samples
# ================================================================

# CUMULATIVE target of takes per word, across ALL sessions, in capture_words.py.
#
# MORE IMPORTANT THAN THE COUNT: capture across MULTIPLE SESSIONS (different
# days, lighting, clothing, distance). A model trained on many takes from a
# SINGLE session memorizes that session, not the sign — it scores high in
# validation but fails live. 3 sessions x ~7 takes generalizes far better than
# 20 takes in one sitting. Re-run the script on different days; it appends until
# this cumulative total is reached, then the word is auto-skipped.
#
#   Raise → more total data (more sessions needed). Lower → fewer sessions.
#
# Set to 34: the existing "legacy" session already holds ~20 takes/word from a
# single sitting, so this leaves room for two more sessions (at the per-session
# cap of 7 below) — 20 + 7 + 7 — giving three sessions total for honest
# cross-session validation. On a clean dataset, ~21 (3 x 7) would be the default.

CAPTURE_TARGET_PER_WORD = 34

# PER-SESSION cap of takes per word in a single capture_words.py run.
#
# This enforces the multi-session discipline above: once this many takes of a
# word are recorded in ONE sitting, the capturer auto-advances to the next word.
# A single session therefore CANNOT fill the whole cumulative quota (which would
# defeat generalization). You leave, come back another day, and the per-session
# counter resets while the cumulative total keeps climbing toward
# CAPTURE_TARGET_PER_WORD.
#
#   Rule of thumb: CAPTURE_TARGET_PER_WORD / desired number of sessions.
# Default: 7 (so ~3 sessions reach the cumulative target of 20).

CAPTURE_PER_SESSION_PER_WORD = 7

# Frames the captured sequence is resampled to before saving.
#
# CRITICAL WARNING: MUST equal WORD_SEQ_LEN. Capture records a variable-length
# take, then resamples it to this fixed length so every saved sample is
# (WORD_SEQ_LEN, 63) — exactly what training and inference expect. Points to
# the same value as WORD_SEQ_LEN so they can never drift apart.

CAPTURE_SEQ_LEN = WORD_SEQ_LEN

# Directory where word-capture sequences (.npy) and the manifest are saved.
#
# Each take is one (WORD_SEQ_LEN, 63) array at seq/<gloss>_<n>.npy; manifest.csv
# records sample_id, gloss and train/val split. This is the .npy sequence format
# the TCN reads (the old flat-CSV mean+std format was removed with the old model).

CAPTURE_OUTPUT_DIR = "data/real_capture/words"


# ================================================================
#  6. PATHS — model locations on disk
# ================================================================
# You rarely need to change these. Only if you reorganize the project or use
# alternative models for experiments.

MODEL_ONE_HAND_PATH   = "model/model_one_hand.h5"
MODEL_WORDS_PATH      = "model/model_words.h5"
MODEL_NUMBERS_PATH    = "model/model_numbers.h5"

LABELS_ONE_HAND_PATH  = "model/labels_one_hand.json"
LABELS_WORDS_PATH     = "model/labels_words.json"
LABELS_NUMBERS_PATH   = "model/labels_numbers.json"

HAND_LANDMARKER_PATH  = "model/hand_landmarker.task"

# MediaPipe pose model — used ONLY by the word pipeline to anchor the hands to
# the body (see WORD_FEATURE_DIM below). Letters/numbers do not need it. If this
# file is missing the app still runs: word features fall back to no body anchor
# (the position part becomes zero), with a warning at startup.

POSE_LANDMARKER_PATH  = "model/pose_landmarker_lite.task"


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


# ================================================================
#  8. OUTPUT — voice and subtitle overlay
# ================================================================

# Master switch for speech synthesis.
#
# Set to False to run silently (overlay still works). Useful when running
# in a noisy environment or while debugging predictions.

VOICE_ENABLED = True


# ================================================================
#  9. SPEECH INPUT — voice-to-text (hearing person → deaf person)
# ================================================================

# Master switch for speech-to-text input.
#
# When True, SpeechInput loads a Whisper model at startup and the user can
# toggle the microphone with the P key. Set to False to disable the feature
# entirely (no microphone access, no Whisper model loaded).

SPEECH_INPUT_ENABLED = True

# Whisper model size. Trades accuracy for speed and disk space.
#
# The model is downloaded once to ~/.cache/huggingface/hub/ on first run.
#
#   "tiny"  (~77 MB)  — fastest; good for clear speech in quiet rooms
#   "base"  (~148 MB) — better accuracy; recommended for noisier environments
#   "small" (~488 MB) — high accuracy; ~3-5s latency on CPU without GPU
#
# "tiny" is the right default for demos and classroom use.

SPEECH_WHISPER_MODEL = "tiny"

# Whisper language hint.
#
# None = auto-detect language each clip (~0.2s overhead per transcription).
# Set to "es" for Spanish, "en" for English, etc., to skip detection and
# improve accuracy when the language is always known in advance.

SPEECH_LANGUAGE = None


# ================================================================
#  10. TRANSLATION — ASL glosses -> fluent sentence (LLM layer)
# ================================================================
# This is the second stage of the two-stage design: the recognizer outputs ASL
# GLOSSES (English keywords, citation form, e.g. "WANT DRINK NOW"); the LLM
# turns that into a natural sentence ("Quiero tomar algo ahora."). Verb
# conjugation and tense live HERE, not in the recognizer — ASL does not
# conjugate verbs, so trying to recognize conjugated forms is both
# linguistically wrong and combinatorially explosive. See src/translator.py.

# Master switch. When False, no API calls are made; pressing the translate key
# just joins the glosses as-is (the app still works fully offline).

TRANSLATION_ENABLED = True

# Target language for the fluent sentence (free text, sent to the model).

TRANSLATION_TARGET_LANGUAGE = "Spanish"

# Claude model used for translation. Glosses->sentence is a small, well-scoped
# task, so a fast, cheap model (Haiku) is a great fit. Any current model works:
#   "claude-haiku-4-5"  -> fastest + cheapest (default; ideal for this task)
#   "claude-sonnet-4-6" -> a step up in quality
#   "claude-opus-4-8"   -> most capable (overkill here)
# The API key is read from the ANTHROPIC_API_KEY environment variable.

TRANSLATION_MODEL = "claude-haiku-4-5"

# If the API is unreachable (no key, no internet, error), fall back to showing
# the raw glosses joined by spaces instead of failing. Keeps the demo robust.

TRANSLATION_OFFLINE_FALLBACK = True
