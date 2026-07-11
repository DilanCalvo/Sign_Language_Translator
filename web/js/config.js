/**
 * Web-side configuration.
 *
 * Every tunable here is COPIED from config.py, which remains the single source
 * of truth for the whole system. If you change a value in config.py, change it
 * here too (and vice versa) — the comments name the original constant so the
 * pairing is greppable.
 */

// ---- Confidence (config.py section 2) ----
export const LETTER_CONFIDENCE_THRESHOLD = 0.80; // LETTER_CONFIDENCE_THRESHOLD
export const LOW_CONFIDENCE_THRESHOLD = 0.70;    // LOW_CONFIDENCE_THRESHOLD
export const ALT_MIN_CONFIDENCE = 0.08;          // ALT_MIN_CONFIDENCE

// ---- Smoothing (config.py section 4) ----
export const LETTER_SMOOTH_WINDOW = 7;           // LETTER_SMOOTH_WINDOW
export const LETTER_SMOOTH_MIN_VOTES = 5;        // LETTER_SMOOTH_MIN_VOTES
export const LETTER_COOLDOWN_FRAMES = 20;        // LETTER_COOLDOWN_FRAMES

// ---- Hand detector (config.py section 7) ----
export const DETECTOR_MIN_DETECTION_CONFIDENCE = 0.70; // DETECTOR_MIN_DETECTION_CONFIDENCE
export const DETECTOR_MIN_PRESENCE_CONFIDENCE = 0.70;  // DETECTOR_MIN_PRESENCE_CONFIDENCE
export const DETECTOR_MIN_TRACKING_CONFIDENCE = 0.50;  // DETECTOR_MIN_TRACKING_CONFIDENCE

// Deliberate difference from Python: the desktop detector tracks 2 hands
// (infrastructure for the word pipeline), but letters only ever consume the
// first hand, and this page is letters-only — 1 hand costs less per frame,
// which matters on phones.
export const NUM_HANDS = 1;

// ---- Assets (served locally, never from a CDN) ----
// The .task file is the exact hand_landmarker.task the training data was
// captured with (copied from model/). A different detector version could
// produce subtly different landmarks than the ones the model was trained on.
export const MODEL_URL = "model/letters/model_weights.json";
export const LABELS_URL = "model/labels_one_hand.json";
export const HAND_TASK_URL = "model/hand_landmarker.task";

// MediaPipe Tasks Vision WASM runtime — vendored locally (web/vendor/) from
// @mediapipe/tasks-vision@0.10.14, same reasons as the .task: no CDN
// dependency at demo time, and a version pinned to what was validated.
// Resolved relative to index.html (the page), not to this module.
export const MEDIAPIPE_WASM_URL = "vendor/mediapipe/wasm";
