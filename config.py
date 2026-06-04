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
#  10. GRAMMAR — verb conjugation with tense support
# ================================================================

# Verb conjugation tables by tense (present, past, future).
#
# Structure:
#   "infinitive": {
#       "present": {
#           "1p_sg": "creo",          (present: yo)
#           "2p_sg": "crees",         (present: tú)
#           ...
#       },
#       "past": {
#           "1p_sg": "creí",          (past: yo)
#           "2p_sg": "creíste",       (past: tú)
#           ...
#       },
#       "future": {
#           "1p_sg": "creeré",        (future: yo)
#           "2p_sg": "creerás",       (future: tú)
#           ...
#       },
#   }
#
# ========================================================================
# HOW TO ADD A NEW VERB:
# ========================================================================
# 1. Choose an infinitive (e.g., "hablar")
# 2. Conjugate all 3 tenses × 6 forms = 18 forms total
# 3. Copy the template below and fill in:
#
#    "hablar": {
#        "present": {
#            "1p_sg": "hablo",
#            "2p_sg": "hablas",
#            "3p_sg": "habla",
#            "1p_pl": "hablamos",
#            "2p_pl": "habláis",
#            "3p_pl": "hablan",
#        },
#        "past": {
#            "1p_sg": "hablé",
#            "2p_sg": "hablaste",
#            "3p_sg": "habló",
#            "1p_pl": "hablamos",
#            "2p_pl": "hablasteis",
#            "3p_pl": "hablaron",
#        },
#        "future": {
#            "1p_sg": "hablaré",
#            "2p_sg": "hablarás",
#            "3p_sg": "hablará",
#            "1p_pl": "hablaremos",
#            "2p_pl": "hablaréis",
#            "3p_pl": "hablarán",
#        },
#    },
#
# IMPORTANT: If you forget a tense, that verb will not conjugate for
# that tense. For safety, ALWAYS provide all 3 tenses (present, past, future).
# ========================================================================

VERB_CONJUGATIONS = {
    "ser": {
        "present": {
            "1p_sg": "soy",
            "2p_sg": "eres",
            "3p_sg": "es",
            "1p_pl": "somos",
            "2p_pl": "sois",
            "3p_pl": "son",
        },
        "past": {
            "1p_sg": "fui",
            "2p_sg": "fuiste",
            "3p_sg": "fue",
            "1p_pl": "fuimos",
            "2p_pl": "fuisteis",
            "3p_pl": "fueron",
        },
        "future": {
            "1p_sg": "seré",
            "2p_sg": "serás",
            "3p_sg": "será",
            "1p_pl": "seremos",
            "2p_pl": "seréis",
            "3p_pl": "serán",
        },
    },
    "creer": {
        "present": {
            "1p_sg": "creo",
            "2p_sg": "crees",
            "3p_sg": "cree",
            "1p_pl": "creemos",
            "2p_pl": "creéis",
            "3p_pl": "creen",
        },
        "past": {
            "1p_sg": "creí",
            "2p_sg": "creíste",
            "3p_sg": "creyó",
            "1p_pl": "creímos",
            "2p_pl": "creísteis",
            "3p_pl": "creyeron",
        },
        "future": {
            "1p_sg": "creeré",
            "2p_sg": "creerás",
            "3p_sg": "creerá",
            "1p_pl": "creeremos",
            "2p_pl": "creeréis",
            "3p_pl": "creerán",
        },
    },
    "pensar": {
        "present": {
            "1p_sg": "pienso",
            "2p_sg": "piensas",
            "3p_sg": "piensa",
            "1p_pl": "pensamos",
            "2p_pl": "pensáis",
            "3p_pl": "piensan",
        },
        "past": {
            "1p_sg": "pensé",
            "2p_sg": "pensaste",
            "3p_sg": "pensó",
            "1p_pl": "pensamos",
            "2p_pl": "pensasteis",
            "3p_pl": "pensaron",
        },
        "future": {
            "1p_sg": "pensaré",
            "2p_sg": "pensarás",
            "3p_sg": "pensará",
            "1p_pl": "pensaremos",
            "2p_pl": "pensaréis",
            "3p_pl": "pensarán",
        },
    },
    "querer": {
        "present": {
            "1p_sg": "quiero",
            "2p_sg": "quieres",
            "3p_sg": "quiere",
            "1p_pl": "queremos",
            "2p_pl": "queréis",
            "3p_pl": "quieren",
        },
        "past": {
            "1p_sg": "quise",
            "2p_sg": "quisiste",
            "3p_sg": "quiso",
            "1p_pl": "quisimos",
            "2p_pl": "quisisteis",
            "3p_pl": "quisieron",
        },
        "future": {
            "1p_sg": "querré",
            "2p_sg": "querrás",
            "3p_sg": "querrá",
            "1p_pl": "querremos",
            "2p_pl": "querréis",
            "3p_pl": "querrán",
        },
    },
    "poder": {
        "present": {
            "1p_sg": "puedo",
            "2p_sg": "puedes",
            "3p_sg": "puede",
            "1p_pl": "podemos",
            "2p_pl": "podéis",
            "3p_pl": "pueden",
        },
        "past": {
            "1p_sg": "pude",
            "2p_sg": "pudiste",
            "3p_sg": "pudo",
            "1p_pl": "pudimos",
            "2p_pl": "pudisteis",
            "3p_pl": "pudieron",
        },
        "future": {
            "1p_sg": "podré",
            "2p_sg": "podrás",
            "3p_sg": "podrá",
            "1p_pl": "podremos",
            "2p_pl": "podréis",
            "3p_pl": "podrán",
        },
    },
    "tener": {
        "present": {
            "1p_sg": "tengo",
            "2p_sg": "tienes",
            "3p_sg": "tiene",
            "1p_pl": "tenemos",
            "2p_pl": "tenéis",
            "3p_pl": "tienen",
        },
        "past": {
            "1p_sg": "tuve",
            "2p_sg": "tuviste",
            "3p_sg": "tuvo",
            "1p_pl": "tuvimos",
            "2p_pl": "tuvisteis",
            "3p_pl": "tuvieron",
        },
        "future": {
            "1p_sg": "tendré",
            "2p_sg": "tendrás",
            "3p_sg": "tendrá",
            "1p_pl": "tendremos",
            "2p_pl": "tendréis",
            "3p_pl": "tendrán",
        },
    },
    "hacer": {
        "present": {
            "1p_sg": "hago",
            "2p_sg": "haces",
            "3p_sg": "hace",
            "1p_pl": "hacemos",
            "2p_pl": "hacéis",
            "3p_pl": "hacen",
        },
        "past": {
            "1p_sg": "hice",
            "2p_sg": "hiciste",
            "3p_sg": "hizo",
            "1p_pl": "hicimos",
            "2p_pl": "hicisteis",
            "3p_pl": "hicieron",
        },
        "future": {
            "1p_sg": "haré",
            "2p_sg": "harás",
            "3p_sg": "hará",
            "1p_pl": "haremos",
            "2p_pl": "haréis",
            "3p_pl": "harán",
        },
    },
    "ir": {
        "present": {
            "1p_sg": "voy",
            "2p_sg": "vas",
            "3p_sg": "va",
            "1p_pl": "vamos",
            "2p_pl": "vais",
            "3p_pl": "van",
        },
        "past": {
            "1p_sg": "fui",
            "2p_sg": "fuiste",
            "3p_sg": "fue",
            "1p_pl": "fuimos",
            "2p_pl": "fuisteis",
            "3p_pl": "fueron",
        },
        "future": {
            "1p_sg": "iré",
            "2p_sg": "irás",
            "3p_sg": "irá",
            "1p_pl": "iremos",
            "2p_pl": "iréis",
            "3p_pl": "irán",
        },
    },
}

# Pronouns mapped to grammatical forms (no changes from before).

PRONOUN_TO_FORM = {
    "yo": "1p_sg",
    "i": "1p_sg",
    "tú": "2p_sg",
    "tu": "2p_sg",
    "you": "2p_sg",
    "él": "3p_sg",
    "she": "3p_sg",
    "ella": "3p_sg",
    "usted": "3p_sg",
    "ud": "3p_sg",
    "he": "3p_sg",
    "nosotros": "1p_pl",
    "nosotras": "1p_pl",
    "we": "1p_pl",
    "vosotros": "2p_pl",
    "vosotras": "2p_pl",
    "you_all": "2p_pl",
    "ellos": "3p_pl",
    "ellas": "3p_pl",
    "they": "3p_pl",
    "ustedes": "3p_pl",
    "uds": "3p_pl",
}

# Time markers: words that indicate the tense context.
#
# When the system detects one of these words, it switches to that tense
# for all following verbs. Case-insensitive.
#
# Examples:
#   "ayer yo creer" → detects "ayer" → past tense → "ayer yo creí"
#   "mañana ellos ir" → detects "mañana" → future tense → "mañana ellos irán"

TIME_MARKERS = {
    # Past tense
    "ayer": "past",
    "anoche": "past",
    "hace": "past",
    "pasado": "past",
    "hace tiempo": "past",
    # Present tense
    "ahora": "present",
    "hoy": "present",
    "en": "present",  # "en este momento"
    "ahorita": "present",
    # Future tense
    "mañana": "future",
    "próximo": "future",
    "luego": "future",
    "después": "future",
    "pronto": "future",
}

# Default tense when no marker is detected.
DEFAULT_TENSE = "present"
