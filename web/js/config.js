/**
 * Web-side configuration.
 *
 * Model/threshold tunables are COPIED from config.py, which remains the single
 * source of truth for the recognizers. If you change a value in config.py,
 * change it here too (and vice versa) — the comments name the original
 * constant so the pairing is greppable.
 *
 * DELIBERATE DIVERGENCE: smoothing/cadence constants are TIME-BASED here
 * (milliseconds) while config.py counts frames — the web runs at wildly
 * variable fps (15-60), the desktop at a fixed camera rate. Each ms value is
 * the exact equivalent of the config.py frame count at the 30fps baseline.
 */

// ---- Confidence (config.py section 2) ----
// Acceptance thresholds are per-mode (letters 0.80, numbers 0.60, words 0.75);
// modes.js maps each mode to its threshold. LOW_CONFIDENCE / ALT_MIN are shared.
export const LETTER_CONFIDENCE_THRESHOLD = 0.80; // LETTER_CONFIDENCE_THRESHOLD
export const NUMBER_CONFIDENCE_THRESHOLD = 0.60; // NUMBER_CONFIDENCE_THRESHOLD
export const WORD_CONFIDENCE_THRESHOLD = 0.75;   // WORD_CONFIDENCE_THRESHOLD
export const LOW_CONFIDENCE_THRESHOLD = 0.70;    // LOW_CONFIDENCE_THRESHOLD
export const ALT_MIN_CONFIDENCE = 0.08;          // ALT_MIN_CONFIDENCE

// ---- Smoothing — TIME-BASED (deliberate web divergence, 2026-07-19) ----
// config.py counts FRAMES because the desktop camera runs at a fixed rate. The
// web runs anywhere from ~15fps (phone, pose active) to 60fps (desktop), so
// frame counts made a phone user hold every sign ~3x longer in wall-clock time.
// Every threshold below is the MILLISECOND equivalent of the config.py frame
// count at the 30fps capture baseline (the comments name the original), so
// desktop behavior is unchanged and phones commit at the same real-time pace.
// Vote minimums become a FRACTION of whatever samples landed in the window,
// with an absolute floor so 1-2 stray frames can never look "stable".
export const LETTER_SMOOTH_WINDOW_MS = 233;      // LETTER_SMOOTH_WINDOW (7 @30fps)
export const LETTER_SMOOTH_MIN_FRACTION = 5 / 7; // LETTER_SMOOTH_MIN_VOTES / window
export const LETTER_SMOOTH_MIN_SAMPLES = 3;      // floor (web-only)
export const LETTER_COOLDOWN_MS = 667;           // LETTER_COOLDOWN_FRAMES (20 @30fps)
// Fingerspelled-word close: no letter for this long -> the word is finished
// (spoken + logged). The static models have no "space" sign, so the pause (or
// the manual Space button) is what ends a word. Web-only.
export const LETTER_WORD_CLOSE_MS = 2000;
// Cap on accumulated chars (the strip shows the last 28; anything beyond this
// is dropped from the front). Web-only: unbounded growth was a slow leak.
export const CHAR_BUFFER_MAX_CHARS = 200;

// ---- Word pipeline (config.py section 3) — dynamic signs ----
export const WORD_SEQ_LEN = 32;                  // WORD_SEQ_LEN
export const WORD_FEATURE_DIM = 130;             // WORD_FEATURE_DIM
export const WORD_BUFFER_MS = 1500;              // WORD_BUFFER_FRAMES (45 @30fps)
export const WORD_MIN_BUFFER_MS = 533;           // WORD_MIN_FRAMES (16 @30fps)
export const WORD_MIN_BUFFER_SAMPLES = 8;        // floor for resample quality (web-only)
export const WORD_MOTION_WINDOW_MS = 267;        // WORD_MOTION_WINDOW (8 @30fps)
export const WORD_MIN_MOTION_STD = 0.020;        // WORD_MIN_MOTION_STD (is-signing gate)
export const WORD_SMOOTH_WINDOW_MS = 167;        // WORD_SMOOTH_WINDOW (5 @30fps)
export const WORD_SMOOTH_MIN_FRACTION = 3 / 5;   // WORD_SMOOTH_MIN_VOTES / window
export const WORD_SMOOTH_MIN_SAMPLES = 2;        // floor (web-only)
export const WORD_COOLDOWN_SECONDS = 1.0;        // WORD_COOLDOWN_SECONDS
export const WORD_SENTENCE_PAUSE_MS = 5000;      // WORD_SENTENCE_PAUSE_FRAMES (150 @30fps)
export const WORD_NULL_LABEL = "nothing";        // WORD_NULL_LABEL (negative class)

// ---- Per-frame work throttles (web-only, time-based) ----
// Shoulders barely move -> pose every ~100ms (was every 3rd frame); consecutive
// TCN inputs barely differ -> resample+TCN at most every ~66ms while signing
// (was every 2nd frame). Time-based so slow devices skip MORE work per second,
// not less, and fast devices don't burn extra inferences.
export const POSE_INTERVAL_MS = 100;
export const WORD_INFER_INTERVAL_MS = 66;

// ---- Camera capture size ----
// Mobile GPUs pay per pixel twice each frame (mirror drawImage + WebGL texture
// upload into MediaPipe), and landmark quality is unchanged at 640x360 — the
// landmarker downscales internally anyway. Desktop keeps the sharper stream.
export const CAMERA_IDEAL_DESKTOP = { width: 960, height: 540 };
export const CAMERA_IDEAL_MOBILE = { width: 640, height: 360 };

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
