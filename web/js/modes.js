/**
 * Per-mode descriptors for the recognition pipeline (letters, numbers, words).
 *
 * Only the values that DIFFER between modes live here; everything shared
 * (smoothing, detector confidences, LOW_CONFIDENCE, ALT_MIN, the .task files,
 * the MediaPipe runtime, the WORD_* constants) stays in config.js. This mirrors
 * the desktop app's L / N / W modes.
 *
 * `kind` routes the per-frame logic in pipeline.js:
 *   "static"   — one frame -> normalize -> Dense model (letters, numbers).
 *   "sequence" — rolling buffer -> body-anchored features -> TCN (words); also
 *                needs the pose detector and both hands.
 *
 * acceptThreshold is imported from config.js (the mirror of config.py) so each
 * threshold has exactly one source.
 */

import {
  LETTER_CONFIDENCE_THRESHOLD,
  NUMBER_CONFIDENCE_THRESHOLD,
  WORD_CONFIDENCE_THRESHOLD,
} from "./config.js";

export const MODES = {
  letters: {
    id: "letters",
    label: "Letters",
    kind: "static",
    modelUrl: "model/letters/model_weights.json",
    labelsUrl: "model/labels_one_hand.json",
    acceptThreshold: LETTER_CONFIDENCE_THRESHOLD,
    heading: "ASL letters, in your browser",
    footer: "Static ASL letters A–Y (J and Z need motion — desktop version roadmap). "
          + "Everything runs locally in your browser; no video leaves your device.",
  },
  numbers: {
    id: "numbers",
    label: "Numbers",
    kind: "static",
    modelUrl: "model/numbers/model_weights.json",
    labelsUrl: "model/labels_numbers.json",
    acceptThreshold: NUMBER_CONFIDENCE_THRESHOLD,
    heading: "ASL numbers, in your browser",
    footer: "Static ASL digits 0–9. "
          + "Everything runs locally in your browser; no video leaves your device.",
  },
  words: {
    id: "words",
    label: "Words",
    kind: "sequence",
    modelUrl: "model/words/model_weights.json",
    labelsUrl: "model/labels_words.json",
    acceptThreshold: WORD_CONFIDENCE_THRESHOLD,
    heading: "ASL word signs, in your browser",
    footer: "Dynamic ASL word signs (both hands + body position). Sign, then pause "
          + "to start a new sentence. Everything runs locally; no video leaves your device.",
  },
};

// The mode the page opens in (matches the desktop default of showing letters
// first for finger-spelling).
export const DEFAULT_MODE = "letters";
