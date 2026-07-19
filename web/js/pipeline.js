/**
 * Live pipeline: camera -> mirrored canvas -> MediaPipe hands -> normalize ->
 * static-pose model -> smoother -> UI. Handles the two static modes (letters
 * and numbers) behind a toggle: they share this whole pipeline and differ only
 * in the active model, labels, and acceptance threshold (see modes.js), exactly
 * like the desktop app's L / N modes. Mirrors main.py's static-sign flow:
 *
 *   raw = classify(frame)          (committed prediction: null below threshold)
 *   smoother.update(raw.prediction)
 *   stable = smoother.getStable()
 *   charBuffer.update(stable)
 *
 * CRITICAL PARITY DETAIL: the desktop app flips the frame BEFORE MediaPipe
 * (src/detector.py: cv2.flip(frame, 1)), so the whole dataset was captured on
 * mirrored video. We therefore draw the camera onto a mirrored canvas and feed
 * THAT canvas to the landmarker — the same image drives both display and
 * detection, exactly like the desktop app.
 *
 * Deliberate difference from Python: no word model here, so there is no
 * "hand_is_signing" motion gate suppressing letters while the hand moves
 * (in main.py that gate exists because the word model runs alongside).
 */

import {
  LOW_CONFIDENCE_THRESHOLD,
  ALT_MIN_CONFIDENCE,
  LETTER_SMOOTH_WINDOW_MS,
  LETTER_SMOOTH_MIN_FRACTION,
  LETTER_SMOOTH_MIN_SAMPLES,
  LETTER_COOLDOWN_MS,
  LETTER_WORD_CLOSE_MS,
  CHAR_BUFFER_MAX_CHARS,
  WORD_SEQ_LEN,
  WORD_FEATURE_DIM,
  WORD_BUFFER_MS,
  WORD_MIN_BUFFER_MS,
  WORD_MIN_BUFFER_SAMPLES,
  WORD_MOTION_WINDOW_MS,
  WORD_MIN_MOTION_STD,
  WORD_SMOOTH_WINDOW_MS,
  WORD_SMOOTH_MIN_FRACTION,
  WORD_SMOOTH_MIN_SAMPLES,
  WORD_COOLDOWN_SECONDS,
  WORD_SENTENCE_PAUSE_MS,
  WORD_NULL_LABEL,
  POSE_INTERVAL_MS,
  WORD_INFER_INTERVAL_MS,
  CAMERA_IDEAL_DESKTOP,
  CAMERA_IDEAL_MOBILE,
  DETECTOR_MIN_DETECTION_CONFIDENCE,
  DETECTOR_MIN_PRESENCE_CONFIDENCE,
  DETECTOR_MIN_TRACKING_CONFIDENCE,
  NUM_HANDS,
  HAND_TASK_URL,
  POSE_TASK_URL,
  MEDIAPIPE_WASM_URL,
} from "./config.js";
import { MODES, DEFAULT_MODE } from "./modes.js";
import {
  normalizeLandmarks,
  TimeSmoother,
  CharBuffer,
  buildWordFeatures,
  resampleSequence,
  sequenceMotion,
  WordCommitFSM,
  WordBuffer,
} from "./utils.js";
import { loadDenseModel, loadWordModel } from "./model.js";
import { speech } from "./tts.js";
import { conversation } from "./conversation.js";
import { commitFeedback } from "./ui.js";
// Vendored copy of @mediapipe/tasks-vision@0.10.14 (see web/vendor/): the app
// must not depend on a CDN being reachable at demo time, and the runtime
// version stays pinned to the one the .task files were validated with.
import {
  FilesetResolver,
  HandLandmarker,
  PoseLandmarker,
  DrawingUtils,
} from "../vendor/mediapipe/vision_bundle.mjs";

// MediaPipe pose landmark indices we anchor the hands to (src/detector.py).
const POSE_LEFT_SHOULDER = 11;
const POSE_RIGHT_SHOULDER = 12;

// ---- DOM ----
const els = {
  status: document.getElementById("status"),
  canvas: document.getElementById("view"),
  letter: document.getElementById("letter"),
  confBar: document.getElementById("conf-bar"),
  confLabel: document.getElementById("conf-label"),
  alts: document.getElementById("alternatives"),
  strip: document.getElementById("strip"),
  clearBtn: document.getElementById("clear"),
  fps: document.getElementById("fps"),
  cameraSel: document.getElementById("camera-select"),
  cameraRow: document.getElementById("camera-row"),
  camControl: document.getElementById("cam-control"),
  camBtn: document.getElementById("cam-btn"),
  camBtnLabel: document.getElementById("cam-btn-label"),
  camMenu: document.getElementById("cam-menu"),
  trackPill: document.getElementById("track-pill"),
  trackText: document.getElementById("track-text"),
  modeBtns: document.querySelectorAll("#mode-toggle button"),
  heading: document.getElementById("heading"),
  footer: document.getElementById("footer"),
  cam: document.getElementById("cam"),
  spaceBtn: document.getElementById("space-btn"),
  stage: document.getElementById("stage"),
  stageCol: document.getElementById("stage-col"),
};

/**
 * Keep the stage from towering over the page when the camera is square or
 * portrait (phone cameras, virtual cams): cap the displayed height at ~68vh
 * by capping the width at aspect * 68vh. Landscape 16:9 streams are
 * unaffected on normal screens.
 *
 * Published as a custom property because the stage bar and the action row cap
 * to the same width, so they stay flush with the video edges.
 */
function fitStageToAspect(w, h) {
  els.stageCol.style.setProperty("--stage-w", `calc(68vh * ${(w / h).toFixed(4)})`);
}
// `ctx` is the VISIBLE overlay canvas — it now draws ONLY the landmarks; the
// camera itself is shown by the native <video> (#cam), which the browser
// composites smoothly even while detectForVideo blocks the main thread.
const ctx = els.canvas.getContext("2d");
// Off-screen, mirrored copy of the frame fed to the detectors. Kept off-DOM so
// the mirror-before-detect parity is preserved without drawing the video on the
// visible canvas (which would couple video smoothness to detection speed).
const detectCanvas = document.createElement("canvas");
// GPU-backed (no willReadFrequently): MediaPipe uploads it to a WebGL texture,
// so a CPU-backed canvas would only add a copy.
const detectCtx = detectCanvas.getContext("2d");

// ---- Two status channels ----
// These used to be one element in the middle of the video, which meant the
// per-frame "waiting for hand…" / "signing…" sat on top of the user's own
// hands and flickered at frame rate. They are different kinds of information
// and now get different homes:
//
//   setStatus()   -> #status, centred OVER the video. BLOCKING states only
//                    (boot, model load, camera trouble). Covering the frame is
//                    correct here: there is nothing to see behind it.
//   setTracking() -> #track-pill, in the bar ABOVE the video. The live channel.
//                    Always present, only ever changes colour and label, so it
//                    never covers the video and never shifts the layout.

/** Bumped on every write so a flashed message knows if it is still the one showing. */
let statusToken = 0;

function setStatus(text, isError = false) {
  statusToken++;
  els.status.textContent = text;
  els.status.classList.toggle("error", isError);
  els.status.classList.toggle("hidden", !text);
}

/** Show a recoverable error over the video, then clear it if nothing replaced it. */
function flashStatus(text, ms = 4000) {
  setStatus(text, true);
  const mine = statusToken;
  setTimeout(() => { if (statusToken === mine) setStatus(""); }, ms);
}

const TRACK_LABELS = {
  busy: "Working…",
  idle: "No hand",
  ready: "Tracking",
  signing: "Signing",
};
let trackState = null;

function setTracking(state) {
  if (state === trackState) return; // called every frame — touch the DOM only on change
  trackState = state;
  els.trackPill.dataset.state = state;
  els.trackText.textContent = TRACK_LABELS[state];
}

// Display-only hysteresis. The motion gate chatters around its threshold, which
// would strobe the pill between "Signing" and "Tracking" several times a second.
// Holding the signing look briefly past the last moving frame is purely
// cosmetic — the word FSM never sees this value.
const SIGNING_HOLD_MS = 250;
let signingUntil = 0;

// ---- State ----
// The static smoother/buffer serve letters + numbers (they hold no model-
// specific state) — the same sharing the desktop app does. `activeMode` selects
// the model/labels/threshold/kind; models are lazy-loaded and cached by id.
const smoother = new TimeSmoother(
  LETTER_SMOOTH_WINDOW_MS, LETTER_SMOOTH_MIN_FRACTION, LETTER_SMOOTH_MIN_SAMPLES);
const charBuffer = new CharBuffer(
  LETTER_COOLDOWN_MS, LETTER_WORD_CLOSE_MS, CHAR_BUFFER_MAX_CHARS);

// Word (sequence) mode state: a rolling buffer of body-anchored feature frames,
// a vote/confirm/lock-out state machine, and the running sentence. Only touched
// when the active mode's kind is "sequence".
const wordSmoother = new TimeSmoother(
  WORD_SMOOTH_WINDOW_MS, WORD_SMOOTH_MIN_FRACTION, WORD_SMOOTH_MIN_SAMPLES);
const wordFsm = new WordCommitFSM(wordSmoother, WORD_COOLDOWN_SECONDS);
const wordBuffer = new WordBuffer(WORD_SENTENCE_PAUSE_MS);
// Rolling feature buffer: wordFrames[i] (Float32Array(130)) captured at
// wordFrameTimes[i] ms; entries older than WORD_BUFFER_MS age out. Evicted
// rows return to a free pool and are reused by buildWordFeatures(..., out) —
// zero steady-state allocation. The pool naturally caps at the buffer's
// frame capacity (~90 rows at 60fps).
let wordFrames = [];
let wordFrameTimes = [];
const wordRowPool = [];

/** Move every buffered row back to the pool (buffer reset, mode switch). */
function recycleWordFrames() {
  while (wordFrames.length) wordRowPool.push(wordFrames.pop());
  wordFrameTimes.length = 0;
}

// Pose is the heaviest per-frame cost in word mode, but the shoulders barely
// move, so we only run it every POSE_INTERVAL_MS and reuse the last shoulders
// in between — big compute/heat saving with negligible anchor error.
let lastPoseAt = -Infinity;
let lastShoulderL = null, lastShoulderR = null;

// The TCN is the heaviest per-frame cost in word mode, but the rolling buffer
// shifts by one frame at a time, so consecutive predictions barely differ —
// the same premise as the pose throttle above. While the hands are actively
// signing we run the (expensive) resample + TCN at most every
// WORD_INFER_INTERVAL_MS and reuse the last softmax in between. The word FSM is
// still stepped every frame with the resulting prediction, so commit timing and
// smoothing are unchanged. The cached probs reset between signing bursts (and
// when the hands leave) so a fresh burst never reuses a stale prediction.
// lastWordProbs points at wordProbsCopy, OUR buffer — model.predict's return
// is model-owned and only valid until the next predict, so it is copied at the
// cache point (one preallocated copy, no per-infer allocation).
let lastInferAt = -Infinity;
let lastWordProbs = null;
let wordProbsCopy = null;

let activeMode = MODES[DEFAULT_MODE];
const modelCache = {};   // mode id -> DenseModel | TCNModel
const labelsCache = {};  // mode id -> {0: "A", ...}
let model = null;        // active mode's model (modelCache[activeMode.id])
let labels = null;       // active mode's labels
let landmarker = null;
let poseLandmarker = null; // lazily created on first entry to word mode
let vision = null;         // FilesetResolver, shared by both landmarkers
let drawer = null;
let video = null;
let lastFrameTimes = [];

// ---- Perf instrumentation (opt-in via ?debug) ----
// Confirms which MediaPipe delegate actually loaded (GPU vs a silent CPU
// fallback) plus the real camera resolution and fps — the data needed to steer
// mobile performance work. Off unless the URL has ?debug, so normal use is
// untouched.
const DEBUG = new URLSearchParams(location.search).has("debug");
let handDelegate = "?";     // "GPU" | "CPU"
let poseDelegate = "none";  // "GPU" | "CPU" | "failed" | "none"
let debugEl = null;

// Per-stage timings (EMA, ms) so the overlay shows WHERE the frame budget
// goes on-device: mirror-draw, hand detect, pose detect, feature build, TCN.
// Sampled only under ?debug — the normal path pays nothing.
const perfMs = { draw: 0, hand: 0, pose: 0, feat: 0, tcn: 0 };
function perfSample(stage, ms) {
  perfMs[stage] = perfMs[stage] === 0 ? ms : perfMs[stage] * 0.9 + ms * 0.1;
}

function resetWordState() {
  recycleWordFrames();
  wordFsm.reset();
  wordBuffer.clear();
  lastPoseAt = -Infinity;
  lastShoulderL = null;
  lastShoulderR = null;
  lastInferAt = -Infinity;
  lastWordProbs = null;
}

els.clearBtn.addEventListener("click", () => {
  // Clear means DISCARD: the in-progress word and its pending closures go too.
  charBuffer.clear();
  resetWordState();
  els.strip.textContent = "";
});

/**
 * Route finished fingerspelled words to voice + conversation. Words close via
 * the Space button, the inactivity pause, or a mode switch; whatever the
 * trigger, they all drain through here (mirrors main.py's
 * pop_completed_words() loop).
 */
function drainSpelledWords() {
  for (const word of charBuffer.popCompletedWords()) {
    if (speech.letterMode === "word") speech.speak(word);
    conversation.addSign(word);
  }
}

els.spaceBtn.addEventListener("click", () => {
  charBuffer.space();
  els.strip.textContent = charBuffer.getText();
  drainSpelledWords();
});

// ---- Replica of src/classifier.py::_run (top-3 + acceptance threshold) ----
// `aspect` (frame width/height) lets normalizeLandmarks undo MediaPipe's
// per-axis stretch — without it a portrait phone feeds the model differently
// stretched vectors than the 16:9 data it was trained on.
// Scratch for the normalized input (reused every frame; consumed by predict
// before this function returns).
const normScratch = new Float32Array(63);

function classify(flat63, aspect) {
  const probs = model.predict(normalizeLandmarks(flat63, aspect, normScratch));
  // Single-pass top-3 (replaces spread + full sort — this runs every frame).
  let i0 = -1, i1 = -1, i2 = -1;
  for (let i = 0; i < probs.length; i++) {
    const p = probs[i];
    if (i0 < 0 || p > probs[i0]) { i2 = i1; i1 = i0; i0 = i; }
    else if (i1 < 0 || p > probs[i1]) { i2 = i1; i1 = i; }
    else if (i2 < 0 || p > probs[i2]) { i2 = i; }
  }
  const top3 = [];
  for (const i of [i0, i1, i2]) {
    if (i >= 0 && probs[i] >= ALT_MIN_CONFIDENCE) {
      top3.push({ prediction: labels[i], confidence: probs[i] });
    }
  }
  const top1Conf = probs[i0];
  return {
    prediction: top1Conf >= activeMode.acceptThreshold ? labels[i0] : null,
    confidence: top1Conf,
    top3,
  };
}

/** Flatten one MediaPipe hand (21 points) into `out` as [x,y,z,...] — the
 *  order src/detector.py builds landmarks in. `out` is a per-slot scratch
 *  (consumed within the frame, never stored). */
function flattenHand(hand, out) {
  for (let i = 0; i < 21; i++) {
    out[i * 3] = hand[i].x;
    out[i * 3 + 1] = hand[i].y;
    out[i * 3 + 2] = hand[i].z;
  }
  return out;
}
const staticFlatScratch = new Float32Array(63);
const handFlatScratch = [new Float32Array(63), new Float32Array(63)];

// ---- Static per-frame (letters, numbers) ----
function processStaticFrame(result, now, w, h) {
  const hand = result.landmarks && result.landmarks[0];
  let clsResult = null;
  if (hand) {
    drawer.drawConnectors(hand, HandLandmarker.HAND_CONNECTIONS,
                          { color: "#ffffff", lineWidth: 2 });
    drawer.drawLandmarks(hand, { color: "#ffd900", radius: 3 });
    clsResult = classify(flattenHand(hand, staticFlatScratch), w / h);
    smoother.update(clsResult.prediction, now);
    setTracking("ready");
  } else {
    smoother.update(null, now);
    setTracking("idle");
  }
  const stable = smoother.getStable();
  if (charBuffer.update(stable, now)) {
    els.strip.textContent = charBuffer.getText();
    if (speech.letterMode === "letter") speech.speak(stable, { interrupt: true });
    commitFeedback();
  }
  // A pause with no new letter finishes the fingerspelled word (the models
  // have no "space" sign) — same closing rule the Space button triggers.
  if (charBuffer.maybeAutoClose(now) !== null) {
    els.strip.textContent = charBuffer.getText();
    drainSpelledWords();
  }
  updateHud(stable, clsResult);
}

// ---- Word per-frame (dynamic signs) ----
// Mirrors classifier._update_word_buffer/_run_words + main.py's word FSM: build
// a body-anchored feature every frame, keep a rolling buffer, and only classify
// (resample -> TCN) while the hands are actually moving. Pose runs on the SAME
// mirrored canvas so shoulders and wrists share one coordinate space, and
// mirror-before-detect makes the Left/Right handedness labels match training.
function processWordFrame(result, now, w, h) {
  const aspect = w / h;
  const hands = result.landmarks || [];

  // No hand in frame: mirror the desktop, which clears the word buffer when
  // num_hands == 0 (src/classifier.py:182). Without this the previous sign's
  // frames linger in wordFrames (up to WORD_BUFFER_MS = 1.5s) and contaminate
  // the next sign — the "residue" where a new sign re-predicts the last word.
  // Still step the FSM with null so the cooldown clock advances and the locked
  // word decays, exactly like main.py's word_fsm.step(None, ...) every frame.
  // Shoulders stay cached (they are still there while the hand is off-screen),
  // and pose is skipped entirely — same early-out as the desktop's _empty().
  if (hands.length === 0) {
    recycleWordFrames();
    // Buffer just emptied -> any cached softmax is stale; force a fresh infer
    // when signing resumes.
    lastWordProbs = null;
    const { word, confidence } = wordFsm.step(null, now / 1000);
    wordBuffer.tick(now);
    els.strip.textContent = wordBuffer.getText();
    updateWordHud(word, confidence);
    signingUntil = 0; // hands gone -> drop the signing hold immediately
    setTracking("idle");
    return;
  }

  // Shoulders from pose (body anchor), throttled to every POSE_INTERVAL_MS
  // and reusing the last result in between (shoulders barely move). Runs on the
  // off-screen mirrored detection canvas, like the hand detector. Missing pose
  // -> no anchor (position 0), the same graceful degradation as the desktop.
  let shoulderL = lastShoulderL, shoulderR = lastShoulderR;
  if (poseLandmarker && now - lastPoseAt >= POSE_INTERVAL_MS) {
    lastPoseAt = now;
    const tPose = DEBUG ? performance.now() : 0;
    const pose = poseLandmarker.detectForVideo(detectCanvas, now);
    if (DEBUG) perfSample("pose", performance.now() - tPose);
    const lm = pose.landmarks && pose.landmarks[0];
    if (lm) {
      shoulderL = [lm[POSE_LEFT_SHOULDER].x, lm[POSE_LEFT_SHOULDER].y];
      shoulderR = [lm[POSE_RIGHT_SHOULDER].x, lm[POSE_RIGHT_SHOULDER].y];
    } else {
      shoulderL = shoulderR = null; // pose lost this frame -> drop the anchor
    }
    lastShoulderL = shoulderL;
    lastShoulderR = shoulderR;
  }
  if (shoulderL && shoulderR) drawShoulders(shoulderL, shoulderR, w, h);

  // Hands by handedness so the same sign always lands in the same slot
  // (parity with src/detector.py hands_by_side).
  const handed = result.handedness || result.handednesses || [];
  let leftHand = null, rightHand = null;
  for (let i = 0; i < hands.length; i++) {
    drawer.drawConnectors(hands[i], HandLandmarker.HAND_CONNECTIONS,
                          { color: "#ffffff", lineWidth: 2 });
    drawer.drawLandmarks(hands[i], { color: "#ffd900", radius: 3 });
    const side = handed[i] && handed[i][0] && handed[i][0].categoryName;
    const flat = flattenHand(hands[i], handFlatScratch[i & 1]);
    if (side === "Left") leftHand = flat;
    else if (side === "Right") rightHand = flat;
  }

  // Feature row from the recycling pool (buildWordFeatures zero-fills it);
  // rows age out of the rolling window back into the pool.
  const tFeat = DEBUG ? performance.now() : 0;
  const row = wordRowPool.pop() || new Float32Array(WORD_FEATURE_DIM);
  buildWordFeatures(leftHand, rightHand, shoulderL, shoulderR, aspect, row);
  wordFrames.push(row);
  wordFrameTimes.push(now);
  while (wordFrameTimes.length && wordFrameTimes[0] < now - WORD_BUFFER_MS) {
    wordRowPool.push(wordFrames.shift());
    wordFrameTimes.shift();
  }
  if (DEBUG) perfSample("feat", performance.now() - tFeat);

  // Classify only once the buffer has filled (enough samples AND enough
  // wall-clock span — sample count alone would fire early at high fps and
  // late at low fps) AND the hands are moving (a still pose is not a dynamic
  // sign). The FSM still steps every frame with the resulting prediction (or
  // null) so it can lock/unlock on cooldown.
  let wordPred = null;
  let isSigning = false;
  if (wordFrames.length >= WORD_MIN_BUFFER_SAMPLES
      && now - wordFrameTimes[0] >= WORD_MIN_BUFFER_MS) {
    // Motion over the last WORD_MOTION_WINDOW_MS of frames (count them from
    // the newest end — the buffer is time-ordered).
    let motionCount = 0;
    const motionCutoff = now - WORD_MOTION_WINDOW_MS;
    for (let i = wordFrameTimes.length - 1;
         i >= 0 && wordFrameTimes[i] >= motionCutoff; i--) motionCount++;
    const motion = sequenceMotion(wordFrames, motionCount);
    isSigning = motion >= WORD_MIN_MOTION_STD;
    if (isSigning) {
      // Throttle the resample + TCN to every WORD_INFER_INTERVAL_MS while
      // signing, reusing the cached softmax in between. A just-reset burst
      // (lastWordProbs === null) always infers fresh first.
      let probs = lastWordProbs;
      if (probs === null || now - lastInferAt >= WORD_INFER_INTERVAL_MS) {
        const tTcn = DEBUG ? performance.now() : 0;
        const seq = resampleSequence(wordFrames, WORD_SEQ_LEN);
        const fresh = model.predict(seq);
        // predict()'s return is model-owned (overwritten next call); cache a
        // copy in our own preallocated buffer.
        if (!wordProbsCopy) wordProbsCopy = new Float32Array(fresh.length);
        wordProbsCopy.set(fresh);
        probs = wordProbsCopy;
        lastWordProbs = probs;
        lastInferAt = now;
        if (DEBUG) perfSample("tcn", performance.now() - tTcn);
      }
      let idx = 0;
      for (let k = 1; k < probs.length; k++) if (probs[k] > probs[idx]) idx = k;
      const conf = probs[idx];
      const label = labels[idx];
      // Commit only above threshold and not the negative "nothing" class.
      if (conf >= activeMode.acceptThreshold && label !== WORD_NULL_LABEL) {
        wordPred = { prediction: label, confidence: conf };
      }
    } else {
      // Gap in signing -> drop the cached probs so the next burst re-infers
      // from the current buffer, not a stale one.
      lastWordProbs = null;
    }
  }

  const { word, confidence, isNew } = wordFsm.step(wordPred, now / 1000);
  if (isNew) {
    wordBuffer.add(word, now);
    // The commit frame — the one moment to speak, log, and flash (mirrors
    // main.py's new_word_detected block: buffer, then voice, then log).
    speech.speak(word);
    conversation.addSign(word);
    commitFeedback();
  }
  wordBuffer.tick(now);
  els.strip.textContent = wordBuffer.getText();

  updateWordHud(word, confidence);
  if (isSigning) signingUntil = now + SIGNING_HOLD_MS;
  setTracking(now < signingUntil ? "signing" : "ready");
}

/** Draw the two shoulder anchors on the (already-mirrored) canvas. */
function drawShoulders(shoulderL, shoulderR, w, h) {
  ctx.fillStyle = "#ff7b00";
  for (const s of [shoulderL, shoulderR]) {
    ctx.beginPath();
    ctx.arc(s[0] * w, s[1] * h, 6, 0, Math.PI * 2);
    ctx.fill();
  }
}

function updateWordHud(word, confidence) {
  // The current sign, kept on screen during the cooldown lock-out. No hint
  // style and no top-3 panel — words commit as whole units, not per frame.
  els.letter.classList.remove("hint");
  els.letter.textContent = word || "–";
  const conf = word ? confidence : 0;
  els.confBar.style.width = `${Math.round(conf * 100)}%`;
  els.confBar.className = conf >= 0.8 ? "good" : conf >= 0.6 ? "mid" : "low";
  els.confLabel.textContent = word ? `${Math.round(conf * 100)}%` : "";
  els.alts.classList.add("hidden");
}

// ---- UI updates ----
function updateHud(stable, result) {
  // The "hint" style separates what the system ACTS ON (a committed, stable
  // letter) from what it merely SURFACES (an uncommitted guess between the
  // display and acceptance thresholds) — same design decision as the desktop
  // HUD. They must not look identical.
  if (stable !== null) {
    els.letter.textContent = stable;
    els.letter.classList.remove("hint");
  } else if (result && result.confidence >= LOW_CONFIDENCE_THRESHOLD) {
    els.letter.textContent = result.top3.length ? result.top3[0].prediction : "–";
    els.letter.classList.add("hint");
  } else {
    els.letter.textContent = "–";
    els.letter.classList.remove("hint");
  }

  const conf = result ? result.confidence : 0;
  els.confBar.style.width = `${Math.round(conf * 100)}%`;
  els.confBar.className = conf >= 0.8 ? "good" : conf >= 0.6 ? "mid" : "low";
  els.confLabel.textContent = result ? `${Math.round(conf * 100)}%` : "";

  // Top-3 alternatives only when the model is unsure (< LOW_CONFIDENCE_THRESHOLD).
  if (result && result.confidence < LOW_CONFIDENCE_THRESHOLD && result.top3.length) {
    els.alts.innerHTML = result.top3
      .map((t) => `<span>${t.prediction} ${Math.round(t.confidence * 100)}%</span>`)
      .join("");
    els.alts.classList.remove("hidden");
  } else {
    els.alts.classList.add("hidden");
  }
}

function updateFps(now) {
  lastFrameTimes.push(now);
  if (lastFrameTimes.length > 30) lastFrameTimes.shift();
  if (lastFrameTimes.length >= 2) {
    const span = lastFrameTimes[lastFrameTimes.length - 1] - lastFrameTimes[0];
    els.fps.textContent = `${Math.round((lastFrameTimes.length - 1) * 1000 / span)} fps`;
  }
}

// Opt-in perf readout (?debug): shows the confirmed delegate for hand + pose,
// the real camera resolution, mode and fps — read directly on the phone, no
// remote debugging needed.
function updateDebug() {
  if (!debugEl) {
    debugEl = document.createElement("div");
    debugEl.id = "perf-debug";
    debugEl.style.cssText =
      "position:absolute;left:10px;bottom:30px;z-index:5;white-space:pre;"
      + "font:11px/1.4 ui-monospace,Consolas,monospace;color:#9f9;"
      + "background:rgba(0,0,0,.62);padding:5px 8px;border-radius:6px;";
    els.canvas.parentElement.appendChild(debugEl);
  }
  const res = video ? `${video.videoWidth}x${video.videoHeight}` : "?";
  const ms = (v) => v.toFixed(1).padStart(5);
  debugEl.textContent =
    `mode: ${activeMode.id}\n` +
    `hand: ${handDelegate}${ms(perfMs.hand)}ms\n` +
    `pose: ${poseDelegate}${poseDelegate === "GPU" || poseDelegate === "CPU" ? ms(perfMs.pose) + "ms" : ""}\n` +
    `draw: ${ms(perfMs.draw)}ms  feat:${ms(perfMs.feat)}ms\n` +
    `tcn:  ${ms(perfMs.tcn)}ms\n` +
    `cam:  ${res}\n` +
    `buf:  ${wordFrames.length}\n` +   // word buffer size (0 with no hand -> no residue)
    `fps:  ${els.fps.textContent || "?"}`;
}

// ---- Per-frame ----
// A crash inside the frame callback would otherwise end the
// requestVideoFrameCallback chain and freeze the app SILENTLY — the worst
// possible failure in a live demo. Errors are caught, reported, and the loop
// keeps going; only persistent failure (e.g. a dead GPU context) stops it,
// with an honest message instead of a frozen image.
let consecutiveErrors = 0;
const MAX_CONSECUTIVE_ERRORS = 30; // ~1s of solid failures at 30fps
// The hiccup notice used to be wiped by the per-frame setStatus("") in the
// process*Frame functions. Those are gone (the live channel is the pill now),
// so a recovered frame has to clear it explicitly or a single transient error
// would leave the message parked over the video forever.
let hiccupShown = false;

function onFrame(now) {
  try {
    // Re-sync canvas dims if the video track changed size mid-session — a
    // phone rotating portrait<->landscape flips videoWidth/Height. The native
    // <video> resizes itself; here we only keep the internal resolutions of the
    // off-screen detection canvas and the overlay in step with it.
    if (video.videoWidth && video.videoWidth !== detectCanvas.width) {
      detectCanvas.width = els.canvas.width = video.videoWidth;
      detectCanvas.height = els.canvas.height = video.videoHeight;
      video.style.aspectRatio = `${video.videoWidth} / ${video.videoHeight}`;
      fitStageToAspect(video.videoWidth, video.videoHeight);
    }

    const w = detectCanvas.width, h = detectCanvas.height;

    // Mirror BEFORE detection (parity with cv2.flip in src/detector.py), onto
    // the OFF-SCREEN canvas. The visible <video> is mirrored by CSS, so what is
    // shown and what is detected match.
    const tDraw = DEBUG ? performance.now() : 0;
    detectCtx.save();
    detectCtx.scale(-1, 1);
    detectCtx.drawImage(video, -w, 0, w, h);
    detectCtx.restore();

    // The overlay canvas only carries the landmarks; clear last frame's.
    ctx.clearRect(0, 0, w, h);
    if (DEBUG) perfSample("draw", performance.now() - tDraw);

    const tHand = DEBUG ? performance.now() : 0;
    const result = landmarker.detectForVideo(detectCanvas, now);
    if (DEBUG) perfSample("hand", performance.now() - tHand);
    if (activeMode.kind === "sequence") {
      processWordFrame(result, now, w, h);
    } else {
      processStaticFrame(result, now, w, h);
    }
    updateFps(now);
    if (DEBUG) updateDebug();
    consecutiveErrors = 0;
    if (hiccupShown) { setStatus(""); hiccupShown = false; }
    framesSeen++;
  } catch (err) {
    consecutiveErrors++;
    console.error("Frame processing error:", err);
    if (consecutiveErrors >= MAX_CONSECUTIVE_ERRORS) {
      setStatus("The detector failed repeatedly. Reload the page to retry.", true);
      return; // stop the loop — an honest halt beats an endless error storm
    }
    setStatus("detector hiccup — retrying…", true);
    hiccupShown = true;
  }
  scheduleNext();
}

// Per-camera-frame callback when supported (no wasted detections at 60Hz),
// otherwise plain rAF. Some environments expose requestVideoFrameCallback but
// never deliver a frame (observed in headless Chromium with a fake camera):
// a watchdog falls back to rAF so the app can never freeze silently before
// its first frame. loopGeneration invalidates any callback scheduled under a
// previous mode, so the fallback cannot leave two frame chains running.
let framesSeen = 0;
let useRvfc = false;
let loopGeneration = 0;

// One callback closure per loop generation (not per frame): scheduling ran
// 30-60x/second, so a fresh closure each frame was steady GC litter.
let frameCb = null;
let frameCbGen = -1;

function scheduleNext() {
  if (frameCbGen !== loopGeneration) {
    const gen = loopGeneration;
    frameCb = (now) => { if (gen === loopGeneration) onFrame(now); };
    frameCbGen = gen;
  }
  if (useRvfc) {
    video.requestVideoFrameCallback(frameCb);
  } else {
    requestAnimationFrame(frameCb);
  }
}

function startLoop() {
  useRvfc = !!video.requestVideoFrameCallback;
  scheduleNext();
  if (useRvfc) {
    setTimeout(() => {
      if (framesSeen === 0) {
        console.warn("requestVideoFrameCallback never fired; "
                     + "falling back to requestAnimationFrame.");
        useRvfc = false;
        loopGeneration++; // orphan any still-pending rVFC callback
        scheduleNext();
      }
    }, 1500);
  }
}

// ---- Camera selection ----
// With several cameras connected the browser picks one on its own; the
// selector lets the user pick (and keep, via localStorage) the right one.
// Mirrors the desktop app's CAMERA_INDEX in config.py, but switchable live.
const CAMERA_STORE_KEY = "asl-web.cameraDeviceId";

/**
 * Acquire (or re-acquire) the camera and attach it to the shared <video>
 * element. The frame loop keeps referencing that same element, so a camera
 * switch never needs to touch the loop. deviceId null = browser default.
 */
// Phones pay per pixel twice per frame (mirror drawImage + WebGL texture
// upload); landmark quality is unchanged at the smaller size because the
// landmarker downscales internally anyway.
const IS_MOBILE = (navigator.userAgentData && navigator.userAgentData.mobile)
  || /Android|iPhone|iPad|Mobi/i.test(navigator.userAgent);
const CAMERA_IDEAL = IS_MOBILE ? CAMERA_IDEAL_MOBILE : CAMERA_IDEAL_DESKTOP;

async function startCamera(deviceId) {
  if (video.srcObject) {
    for (const t of video.srcObject.getTracks()) t.stop();
  }
  const stream = await navigator.mediaDevices.getUserMedia({
    video: {
      ...(deviceId ? { deviceId: { exact: deviceId } } : { facingMode: "user" }),
      width: { ideal: CAMERA_IDEAL.width },
      height: { ideal: CAMERA_IDEAL.height },
    },
    audio: false,
  });
  video.srcObject = stream;
  await video.play();

  // Size the off-screen detection canvas and the overlay to the camera. The
  // native <video> takes the real aspect ratio (CSS placeholder 16:9 is only
  // pre-camera), so its displayed box matches the absolutely-positioned overlay.
  detectCanvas.width = els.canvas.width = video.videoWidth;
  detectCanvas.height = els.canvas.height = video.videoHeight;
  video.style.aspectRatio = `${video.videoWidth} / ${video.videoHeight}`;
  fitStageToAspect(video.videoWidth, video.videoHeight);
}

function currentCameraId() {
  const track = video.srcObject && video.srcObject.getVideoTracks()[0];
  return track ? track.getSettings().deviceId : null;
}

let cameras = [];

/**
 * Display name for a device. Browsers append the USB vendor:product id
 * ("HD Pro Webcam C920 (046d:082d)"), which is noise to a user and the reason
 * the old on-video picker was so wide. Labels only exist once permission is
 * granted, hence the numbered fallback.
 */
function cameraLabel(cam, i) {
  const raw = (cam.label || "").replace(/\s*\([0-9a-f]{4}:[0-9a-f]{4}\)\s*$/i, "").trim();
  return raw || `Camera ${i + 1}`;
}

function closeCamMenu() {
  els.camMenu.classList.add("hidden");
  els.camBtn.setAttribute("aria-expanded", "false");
}

/**
 * Rebuild both camera controls from the current device list: the bar button
 * (quick access, next to the video) and the Settings row (where a user goes
 * looking for a device preference). Both drive selectCamera, so the behaviour
 * cannot drift between them. Hidden entirely unless there is a real choice.
 */
async function refreshCameraList() {
  const devices = await navigator.mediaDevices.enumerateDevices();
  cameras = devices.filter((d) => d.kind === "videoinput");
  const multi = cameras.length >= 2;
  els.camControl.classList.toggle("hidden", !multi);
  els.cameraRow.hidden = !multi;
  if (!multi) return;

  const activeId = currentCameraId();

  els.cameraSel.replaceChildren();
  cameras.forEach((cam, i) => {
    const opt = document.createElement("option");
    opt.value = cam.deviceId;
    opt.textContent = cameraLabel(cam, i);
    opt.selected = cam.deviceId === activeId;
    els.cameraSel.appendChild(opt);
  });

  // Exactly two cameras is the phone case (front/back), where the only thing
  // anyone wants is to flip. A menu to choose between two options is a tap of
  // pure ceremony, so the control collapses to a one-tap toggle instead.
  const flip = cameras.length === 2;
  els.camControl.dataset.mode = flip ? "flip" : "menu";
  closeCamMenu();
  if (flip) {
    els.camBtn.title = "Switch camera";
    els.camBtnLabel.textContent = "Flip";
    els.camBtn.removeAttribute("aria-haspopup");
    els.camBtn.removeAttribute("aria-expanded");
    return;
  }

  const activeIdx = cameras.findIndex((c) => c.deviceId === activeId);
  els.camBtn.title = "Choose camera";
  els.camBtnLabel.textContent =
    activeIdx >= 0 ? cameraLabel(cameras[activeIdx], activeIdx) : "Camera";
  els.camBtn.setAttribute("aria-haspopup", "listbox");
  els.camBtn.setAttribute("aria-expanded", "false");
  els.camMenu.replaceChildren();
  cameras.forEach((cam, i) => {
    const item = document.createElement("button");
    item.type = "button";
    item.setAttribute("role", "option");
    item.setAttribute("aria-selected", cam.deviceId === activeId ? "true" : "false");
    item.textContent = cameraLabel(cam, i);
    item.addEventListener("click", () => {
      closeCamMenu();
      selectCamera(cam.deviceId);
    });
    els.camMenu.appendChild(item);
  });
}

/** Switch cameras, keeping the app alive if the new device refuses to open. */
async function selectCamera(newId) {
  if (!newId || newId === currentCameraId()) return;
  const previousId = currentCameraId();
  els.camBtn.disabled = els.cameraSel.disabled = true;
  setStatus("switching camera…");
  setTracking("busy");
  try {
    await startCamera(newId);
    localStorage.setItem(CAMERA_STORE_KEY, newId);
    // Fresh camera, fresh votes — but the spelled strip is the user's work
    // and survives the switch on purpose.
    smoother.reset();
    setStatus("");
  } catch (err) {
    // Demo safety: failing to switch must not leave the app with NO camera.
    // Restore the previous one; only if that also fails, give up honestly.
    console.error("Camera switch failed:", err);
    try {
      await startCamera(previousId);
      // Recoverable: the previous camera is live again, so the notice clears
      // itself instead of parking over a perfectly working video.
      flashStatus("could not switch camera — kept the previous one");
    } catch {
      setStatus("Camera unavailable. Reload the page to retry.", true);
    }
  } finally {
    els.camBtn.disabled = els.cameraSel.disabled = false;
    await refreshCameraList(); // re-sync both controls to whatever is actually live
  }
}

els.camBtn.addEventListener("click", () => {
  if (els.camControl.dataset.mode === "flip") {
    const other = cameras.find((c) => c.deviceId !== currentCameraId());
    if (other) selectCamera(other.deviceId);
    return;
  }
  const open = els.camMenu.classList.toggle("hidden") === false;
  els.camBtn.setAttribute("aria-expanded", open ? "true" : "false");
});
els.cameraSel.addEventListener("change", () => selectCamera(els.cameraSel.value));

// Dismiss the popover the way every menu is expected to: click away or Escape.
document.addEventListener("click", (ev) => {
  if (!els.camControl.contains(ev.target)) closeCamMenu();
});
document.addEventListener("keydown", (ev) => {
  if (ev.key === "Escape") closeCamMenu();
});

// ---- Mode loading + switching ----
// The three modes share the camera, the hand detector, and the mirror-then-
// detect rule; a switch swaps the model, labels, threshold, and per-frame logic
// (via `activeMode`). Each model is fetched once and cached, so toggling is
// instant. The word model additionally needs the pose detector (lazy).
async function loadMode(modeId) {
  const mode = MODES[modeId];
  if (!modelCache[modeId]) {
    const loadModel = mode.kind === "sequence" ? loadWordModel : loadDenseModel;
    const [m, l] = await Promise.all([
      loadModel(mode.modelUrl),
      fetch(mode.labelsUrl).then((r) => {
        if (!r.ok) throw new Error(`labels ${r.status}`);
        return r.json();
      }),
    ]);
    modelCache[modeId] = m;
    labelsCache[modeId] = l;
  }
  // Word mode needs the pose detector for the body anchor — created once here.
  if (mode.kind === "sequence") await ensurePose();
}

/** Create the pose landmarker once, on first entry to word mode. Non-fatal: a
 *  pose failure degrades word features (position anchor -> 0) but keeps the
 *  handshape working, the same graceful degradation as the desktop app. */
async function ensurePose() {
  if (poseLandmarker || !vision) return;
  const options = {
    baseOptions: { modelAssetPath: POSE_TASK_URL, delegate: "GPU" },
    runningMode: "VIDEO",
    numPoses: 1,
  };
  try {
    poseLandmarker = await PoseLandmarker.createFromOptions(vision, options);
    poseDelegate = "GPU";
  } catch {
    try {
      options.baseOptions.delegate = "CPU"; // no usable GPU delegate -> CPU
      poseLandmarker = await PoseLandmarker.createFromOptions(vision, options);
      poseDelegate = "CPU";
    } catch (err) {
      poseDelegate = "failed";
      console.warn("Pose detector unavailable; word signs will lose the body "
                   + "anchor (handshape only).", err);
    }
  }
}

function applyModeUi(mode) {
  els.heading.textContent = mode.heading;
  els.footer.textContent = mode.footer;
  // Lets the stylesheet show/hide mode-specific controls (e.g. the Space
  // button only makes sense while fingerspelling).
  document.body.dataset.modeKind = mode.kind;
  els.modeBtns.forEach((b) => {
    const on = b.dataset.mode === mode.id;
    b.classList.toggle("active", on);
    b.setAttribute("aria-pressed", on ? "true" : "false");
  });
}

async function switchMode(modeId) {
  if (modeId === activeMode.id || !MODES[modeId]) return;
  els.modeBtns.forEach((b) => (b.disabled = true));
  try {
    setStatus(`loading ${MODES[modeId].label.toLowerCase()}…`);
    setTracking("busy");
    await loadMode(modeId);
    activeMode = MODES[modeId];
    model = modelCache[modeId];
    labels = labelsCache[modeId];
    // A mode switch is not a discard: close the fingerspelled word in
    // progress so it reaches the conversation before the strip resets.
    charBuffer.space();
    drainSpelledWords();
    // Fresh mode, fresh votes and strip — mirrors main.py resetting the
    // smoother, buffer and word FSM when the desktop app changes mode.
    smoother.reset();
    charBuffer.clear();
    resetWordState();
    els.strip.textContent = "";
    applyModeUi(activeMode);
    setStatus(""); // the pill takes over the moment the loop runs a frame
  } catch (err) {
    // A failed switch must not leave the app modeless: activeMode was not
    // reassigned (the load threw first), so we stay on the previous mode.
    console.error("Mode switch failed:", err);
    applyModeUi(activeMode);
    // Recoverable — the previous mode is still running behind the notice, so
    // it clears itself rather than sitting over a working video.
    flashStatus(`could not load ${MODES[modeId].label} — kept ${activeMode.label}`);
  } finally {
    els.modeBtns.forEach((b) => (b.disabled = false));
  }
}

els.modeBtns.forEach((b) =>
  b.addEventListener("click", () => switchMode(b.dataset.mode)));

// ---- Boot ----
async function main() {
  if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
    setStatus("This browser does not support camera access. Try Chrome.", true);
    return;
  }

  try {
    setStatus("starting…");
    // Fire the independent boot work concurrently instead of in series: the
    // model weights, the MediaPipe WASM runtime (~9 MB) and camera acquisition
    // do not depend on each other, so overlapping them (and, on first visit,
    // the permission prompt with the ~17 MB of downloads) cuts time-to-first-
    // frame. The AWAIT order below preserves every dependency and the serial
    // code's exact error handling; only the START is overlapped.
    const modePromise = loadMode(activeMode.id);
    // Module-scoped so ensurePose() can reuse it when word mode is first entered.
    const visionPromise = FilesetResolver.forVisionTasks(MEDIAPIPE_WASM_URL);

    // Camera setup can begin now; the prompt/warmup overlaps the downloads.
    // Same saved-camera fallback + error semantics as the serial code, wrapped
    // in a promise. A permission error is not a device problem, so it does not
    // clear the saved choice.
    video = els.cam;          // the native <video> shown (and CSS-mirrored) in #stage
    video.playsInline = true; // iOS: play inline instead of fullscreen
    video.muted = true;
    const savedId = localStorage.getItem(CAMERA_STORE_KEY);
    const cameraPromise = (async () => {
      try {
        await startCamera(savedId);
      } catch (err) {
        if (!savedId || err.name === "NotAllowedError") throw err;
        localStorage.removeItem(CAMERA_STORE_KEY);
        await startCamera(null);
      }
    })();
    // If an earlier await below rejects we jump to catch without awaiting these;
    // mark them handled so a late rejection is not reported as "unhandled". The
    // awaits still surface the real error to the catch block.
    modePromise.catch(() => {});
    visionPromise.catch(() => {});
    cameraPromise.catch(() => {});

    // Model + labels (small): also settles the mode UI early.
    await modePromise;
    model = modelCache[activeMode.id];
    labels = labelsCache[activeMode.id];
    applyModeUi(activeMode);

    // Hand detector: needs the WASM runtime (vision), then loads its .task.
    setStatus("loading hand detector…");
    vision = await visionPromise;
    const options = {
      baseOptions: { modelAssetPath: HAND_TASK_URL, delegate: "GPU" },
      runningMode: "VIDEO",
      numHands: NUM_HANDS,
      minHandDetectionConfidence: DETECTOR_MIN_DETECTION_CONFIDENCE,
      minHandPresenceConfidence: DETECTOR_MIN_PRESENCE_CONFIDENCE,
      minTrackingConfidence: DETECTOR_MIN_TRACKING_CONFIDENCE,
    };
    try {
      landmarker = await HandLandmarker.createFromOptions(vision, options);
      handDelegate = "GPU";
    } catch {
      // Some devices have no usable GPU delegate — retry on CPU instead of dying.
      options.baseOptions.delegate = "CPU";
      landmarker = await HandLandmarker.createFromOptions(vision, options);
      handDelegate = "CPU";
    }
    drawer = new DrawingUtils(ctx);

    // Camera must be ready before the loop reads frames from it.
    setStatus("requesting camera…");
    await cameraPromise;

    // Device labels only exist after permission was granted, so the selector
    // is built now, not at page load. Refresh it if cameras (un)plug.
    await refreshCameraList();
    navigator.mediaDevices.addEventListener("devicechange", refreshCameraList);

    setStatus(""); // uncover the video; the pill reports tracking from here on
    setTracking("idle");
    startLoop();
  } catch (err) {
    if (err.name === "NotAllowedError") {
      setStatus("Camera permission denied. Allow the camera and reload.", true);
    } else if (err.name === "NotFoundError") {
      setStatus("No camera found on this device.", true);
    } else {
      setStatus(`Could not start: ${err.message}`, true);
    }
  }
}

main();

// Test hooks, only under ?debug: lets an end-to-end test drive the sign-side
// commit path (chars -> Space/auto-close -> voice + conversation) without a
// physical hand in front of the camera.
if (DEBUG) {
  window.__signexDebug = {
    charBuffer,
    conversation,
    speech,
    refreshStrip() { els.strip.textContent = charBuffer.getText(); },
    drainSpelledWords,
  };
}
