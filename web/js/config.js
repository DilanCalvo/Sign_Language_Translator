/**
 * Web-side configuration.
 *
 * Every tunable here is COPIED from config.py, which remains the single source
 * of truth for the whole system. If you change a value in config.py, change it
 * here too (and vice versa) — the comments name the original constant so the
 * pairing is greppable.
 */

// ---- Confidence (config.py section 2) ----
// Acceptance thresholds are per-mode (letters 0.80, numbers 0.60, words 0.75);
// modes.js maps each mode to its threshold. LOW_CONFIDENCE / ALT_MIN are shared.
export const LETTER_CONFIDENCE_THRESHOLD = 0.80; // LETTER_CONFIDENCE_THRESHOLD
export const NUMBER_CONFIDENCE_THRESHOLD = 0.60; // NUMBER_CONFIDENCE_THRESHOLD
export const WORD_CONFIDENCE_THRESHOLD = 0.75;   // WORD_CONFIDENCE_THRESHOLD
export const LOW_CONFIDENCE_THRESHOLD = 0.70;    // LOW_CONFIDENCE_THRESHOLD
export const ALT_MIN_CONFIDENCE = 0.08;          // ALT_MIN_CONFIDENCE

// ---- Smoothing (config.py section 4) ----
export const LETTER_SMOOTH_WINDOW = 7;           // LETTER_SMOOTH_WINDOW
export const LETTER_SMOOTH_MIN_VOTES = 5;        // LETTER_SMOOTH_MIN_VOTES
export const LETTER_COOLDOWN_FRAMES = 20;        // LETTER_COOLDOWN_FRAMES

// ---- Word pipeline (config.py section 3) — dynamic signs ----
export const WORD_SEQ_LEN = 32;                  // WORD_SEQ_LEN
export const WORD_FEATURE_DIM = 130;             // WORD_FEATURE_DIM
export const WORD_BUFFER_FRAMES = 45;            // WORD_BUFFER_FRAMES (rolling buffer max)
export const WORD_MIN_FRAMES = 16;               // WORD_MIN_FRAMES (before classifying)
export const WORD_MOTION_WINDOW = 8;             // WORD_MOTION_WINDOW
export const WORD_MIN_MOTION_STD = 0.020;        // WORD_MIN_MOTION_STD (is-signing gate)
export const WORD_SMOOTH_WINDOW = 5;             // WORD_SMOOTH_WINDOW
export const WORD_SMOOTH_MIN_VOTES = 3;          // WORD_SMOOTH_MIN_VOTES
export const WORD_COOLDOWN_SECONDS = 1.0;        // WORD_COOLDOWN_SECONDS
export const WORD_SENTENCE_PAUSE_FRAMES = 150;   // WORD_SENTENCE_PAUSE_FRAMES (auto-clear)
export const WORD_NULL_LABEL = "nothing";        // WORD_NULL_LABEL (negative class)

// ---- Hand detector (config.py section 7) ----
export const DETECTOR_MIN_DETECTION_CONFIDENCE = 0.70; // DETECTOR_MIN_DETECTION_CONFIDENCE
export const DETECTOR_MIN_PRESENCE_CONFIDENCE = 0.70;  // DETECTOR_MIN_PRESENCE_CONFIDENCE
export const DETECTOR_MIN_TRACKING_CONFIDENCE = 0.50;  // DETECTOR_MIN_TRACKING_CONFIDENCE

// Track 2 hands like the desktop detector (src/detector.py, num_hands=2). The
// word pipeline needs both hands; letters/numbers still classify only the
// primary hand ([0]), so their behavior is unchanged — this just matches the
// desktop and enables words without a second landmarker instance.
export const NUM_HANDS = 2;

// ---- Assets (served locally, never from a CDN) ----
// The .task files are the exact detectors the training data was captured with
// (copied from model/). A different detector version could produce subtly
// different landmarks than the ones the models were trained on. Both hand
// detectors are shared by all modes; the pose detector is used only by words
// (created lazily on first entry to word mode). Per-mode model + labels URLs
// live in modes.js.
export const HAND_TASK_URL = "model/hand_landmarker.task";
export const POSE_TASK_URL = "model/pose_landmarker_lite.task";

// MediaPipe Tasks Vision WASM runtime — vendored locally (web/vendor/) from
// @mediapipe/tasks-vision@0.10.14, same reasons as the .task: no CDN
// dependency at demo time, and a version pinned to what was validated.
// Resolved relative to index.html (the page), not to this module.
export const MEDIAPIPE_WASM_URL = "vendor/mediapipe/wasm";
