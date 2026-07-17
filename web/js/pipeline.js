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
  LETTER_SMOOTH_WINDOW,
  LETTER_SMOOTH_MIN_VOTES,
  LETTER_COOLDOWN_FRAMES,
  WORD_SEQ_LEN,
  WORD_BUFFER_FRAMES,
  WORD_MIN_FRAMES,
  WORD_MOTION_WINDOW,
  WORD_MIN_MOTION_STD,
  WORD_SMOOTH_WINDOW,
  WORD_SMOOTH_MIN_VOTES,
  WORD_COOLDOWN_SECONDS,
  WORD_SENTENCE_PAUSE_FRAMES,
  WORD_NULL_LABEL,
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
  PredictionSmoother,
  CharBuffer,
  buildWordFeatures,
  resampleSequence,
  sequenceMotion,
  WordCommitFSM,
  WordBuffer,
} from "./utils.js";
import { loadDenseModel, loadWordModel } from "./model.js";
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
  modeBtns: document.querySelectorAll("#mode-toggle button"),
  heading: document.getElementById("heading"),
  footer: document.getElementById("footer"),
};
const ctx = els.canvas.getContext("2d");

function setStatus(text, isError = false) {
  els.status.textContent = text;
  els.status.classList.toggle("error", isError);
  els.status.classList.toggle("hidden", !text);
}

// ---- State ----
// The static smoother/buffer serve letters + numbers (they hold no model-
// specific state) — the same sharing the desktop app does. `activeMode` selects
// the model/labels/threshold/kind; models are lazy-loaded and cached by id.
const smoother = new PredictionSmoother(LETTER_SMOOTH_WINDOW, LETTER_SMOOTH_MIN_VOTES);
const charBuffer = new CharBuffer(LETTER_COOLDOWN_FRAMES);

// Word (sequence) mode state: a rolling buffer of body-anchored feature frames,
// a vote/confirm/lock-out state machine, and the running sentence. Only touched
// when the active mode's kind is "sequence".
const wordSmoother = new PredictionSmoother(WORD_SMOOTH_WINDOW, WORD_SMOOTH_MIN_VOTES);
const wordFsm = new WordCommitFSM(wordSmoother, WORD_COOLDOWN_SECONDS);
const wordBuffer = new WordBuffer(WORD_SENTENCE_PAUSE_FRAMES);
let wordFrames = [];     // Float32Array(130) per frame, capped at WORD_BUFFER_FRAMES

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

function resetWordState() {
  wordFrames = [];
  wordFsm.reset();
  wordBuffer.clear();
}

els.clearBtn.addEventListener("click", () => {
  charBuffer.clear();
  resetWordState();
  els.strip.textContent = "";
});

// ---- Replica of src/classifier.py::_run (top-3 + acceptance threshold) ----
// `aspect` (frame width/height) lets normalizeLandmarks undo MediaPipe's
// per-axis stretch — without it a portrait phone feeds the model differently
// stretched vectors than the 16:9 data it was trained on.
function classify(flat63, aspect) {
  const probs = model.predict(normalizeLandmarks(flat63, aspect));
  const order = [...probs.keys()].sort((a, b) => probs[b] - probs[a]);
  const top3 = order.slice(0, 3)
    .filter((i) => probs[i] >= ALT_MIN_CONFIDENCE)
    .map((i) => ({ prediction: labels[i], confidence: probs[i] }));
  const top1Conf = probs[order[0]];
  return {
    prediction: top1Conf >= activeMode.acceptThreshold ? labels[order[0]] : null,
    confidence: top1Conf,
    top3,
  };
}

/** Flatten one MediaPipe hand (21 points) to [x,y,z,...] — the order
 *  src/detector.py builds landmarks in. */
function flattenHand(hand) {
  const flat = new Array(63);
  for (let i = 0; i < 21; i++) {
    flat[i * 3] = hand[i].x;
    flat[i * 3 + 1] = hand[i].y;
    flat[i * 3 + 2] = hand[i].z;
  }
  return flat;
}

// ---- Static per-frame (letters, numbers) ----
function processStaticFrame(result, w, h) {
  const hand = result.landmarks && result.landmarks[0];
  let clsResult = null;
  if (hand) {
    drawer.drawConnectors(hand, HandLandmarker.HAND_CONNECTIONS,
                          { color: "#ffffff", lineWidth: 2 });
    drawer.drawLandmarks(hand, { color: "#ffd900", radius: 3 });
    clsResult = classify(flattenHand(hand), w / h);
    smoother.update(clsResult.prediction);
    setStatus("");
  } else {
    smoother.update(null);
    setStatus("waiting for hand…");
  }
  const stable = smoother.getStable();
  if (charBuffer.update(stable)) {
    els.strip.textContent = charBuffer.getText();
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

  // Shoulders from pose (body anchor). Missing pose -> no anchor (position 0),
  // the same graceful degradation the desktop does when pose is unavailable.
  let shoulderL = null, shoulderR = null;
  if (poseLandmarker) {
    const pose = poseLandmarker.detectForVideo(els.canvas, now);
    const lm = pose.landmarks && pose.landmarks[0];
    if (lm) {
      shoulderL = [lm[POSE_LEFT_SHOULDER].x, lm[POSE_LEFT_SHOULDER].y];
      shoulderR = [lm[POSE_RIGHT_SHOULDER].x, lm[POSE_RIGHT_SHOULDER].y];
      drawShoulders(shoulderL, shoulderR, w, h);
    }
  }

  // Hands by handedness so the same sign always lands in the same slot
  // (parity with src/detector.py hands_by_side).
  const hands = result.landmarks || [];
  const handed = result.handedness || result.handednesses || [];
  let leftHand = null, rightHand = null;
  for (let i = 0; i < hands.length; i++) {
    drawer.drawConnectors(hands[i], HandLandmarker.HAND_CONNECTIONS,
                          { color: "#ffffff", lineWidth: 2 });
    drawer.drawLandmarks(hands[i], { color: "#ffd900", radius: 3 });
    const side = handed[i] && handed[i][0] && handed[i][0].categoryName;
    const flat = flattenHand(hands[i]);
    if (side === "Left") leftHand = flat;
    else if (side === "Right") rightHand = flat;
  }

  const feat = buildWordFeatures(leftHand, rightHand, shoulderL, shoulderR, aspect);
  wordFrames.push(feat);
  if (wordFrames.length > WORD_BUFFER_FRAMES) wordFrames.shift();

  // Classify only once the buffer has filled AND the hands are moving (a still
  // pose is not a dynamic sign). The FSM still steps every frame with the
  // resulting prediction (or null) so it can lock/unlock on cooldown.
  let wordPred = null;
  let isSigning = false;
  if (wordFrames.length >= WORD_MIN_FRAMES) {
    const motion = sequenceMotion(wordFrames, WORD_MOTION_WINDOW);
    isSigning = motion >= WORD_MIN_MOTION_STD;
    if (isSigning) {
      const seq = resampleSequence(wordFrames, WORD_SEQ_LEN);
      const probs = model.predict(seq);
      let idx = 0;
      for (let k = 1; k < probs.length; k++) if (probs[k] > probs[idx]) idx = k;
      const conf = probs[idx];
      const label = labels[idx];
      // Commit only above threshold and not the negative "nothing" class.
      if (conf >= activeMode.acceptThreshold && label !== WORD_NULL_LABEL) {
        wordPred = { prediction: label, confidence: conf };
      }
    }
  }

  const { word, confidence, isNew } = wordFsm.step(wordPred, now / 1000);
  if (isNew) wordBuffer.add(word);
  wordBuffer.tick();
  els.strip.textContent = wordBuffer.getText();

  updateWordHud(word, confidence);
  setStatus(isSigning ? "signing…" : (hands.length ? "" : "waiting for hand…"));
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
      "position:absolute;left:8px;bottom:8px;z-index:5;white-space:pre;"
      + "font:11px/1.4 ui-monospace,Consolas,monospace;color:#9f9;"
      + "background:rgba(0,0,0,.62);padding:5px 8px;border-radius:6px;";
    els.canvas.parentElement.appendChild(debugEl);
  }
  const res = video ? `${video.videoWidth}x${video.videoHeight}` : "?";
  debugEl.textContent =
    `mode: ${activeMode.id}\n` +
    `hand: ${handDelegate}\n` +
    `pose: ${poseDelegate}\n` +
    `cam:  ${res}\n` +
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

function onFrame(now) {
  try {
    // Re-sync canvas dims if the video track changed size mid-session — a
    // phone rotating portrait<->landscape flips videoWidth/Height, and a stale
    // canvas would both squash the detector input and feed classify() the
    // wrong aspect ratio.
    if (video.videoWidth && video.videoWidth !== els.canvas.width) {
      els.canvas.width = video.videoWidth;
      els.canvas.height = video.videoHeight;
      els.canvas.style.aspectRatio = `${video.videoWidth} / ${video.videoHeight}`;
    }

    const w = els.canvas.width, h = els.canvas.height;

    // Mirror BEFORE detection (parity with cv2.flip in src/detector.py).
    ctx.save();
    ctx.scale(-1, 1);
    ctx.drawImage(video, -w, 0, w, h);
    ctx.restore();

    const result = landmarker.detectForVideo(els.canvas, now);
    if (activeMode.kind === "sequence") {
      processWordFrame(result, now, w, h);
    } else {
      processStaticFrame(result, w, h);
    }
    updateFps(now);
    if (DEBUG) updateDebug();
    consecutiveErrors = 0;
    framesSeen++;
  } catch (err) {
    consecutiveErrors++;
    console.error("Frame processing error:", err);
    if (consecutiveErrors >= MAX_CONSECUTIVE_ERRORS) {
      setStatus("The detector failed repeatedly. Reload the page to retry.", true);
      return; // stop the loop — an honest halt beats an endless error storm
    }
    setStatus("detector hiccup — retrying…", true);
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

function scheduleNext() {
  const gen = loopGeneration;
  const cb = (now) => { if (gen === loopGeneration) onFrame(now); };
  if (useRvfc) {
    video.requestVideoFrameCallback(cb);
  } else {
    requestAnimationFrame(cb);
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
async function startCamera(deviceId) {
  if (video.srcObject) {
    for (const t of video.srcObject.getTracks()) t.stop();
  }
  const stream = await navigator.mediaDevices.getUserMedia({
    video: {
      ...(deviceId ? { deviceId: { exact: deviceId } } : { facingMode: "user" }),
      width: { ideal: 960 },
      height: { ideal: 540 },
    },
    audio: false,
  });
  video.srcObject = stream;
  await video.play();

  els.canvas.width = video.videoWidth;
  els.canvas.height = video.videoHeight;
  // The CSS 16:9 aspect is only a pre-camera placeholder; cameras differ
  // (and may differ between each other) — never stretch the image.
  els.canvas.style.aspectRatio = `${video.videoWidth} / ${video.videoHeight}`;
}

function currentCameraId() {
  const track = video.srcObject && video.srcObject.getVideoTracks()[0];
  return track ? track.getSettings().deviceId : null;
}

/** Populate the selector. Hidden unless there is a real choice (2+ cameras). */
async function refreshCameraList() {
  const devices = await navigator.mediaDevices.enumerateDevices();
  const cams = devices.filter((d) => d.kind === "videoinput");
  if (cams.length < 2) {
    els.cameraSel.classList.add("hidden");
    return;
  }
  const activeId = currentCameraId();
  els.cameraSel.innerHTML = "";
  cams.forEach((cam, i) => {
    const opt = document.createElement("option");
    opt.value = cam.deviceId;
    // Labels are only exposed once camera permission is granted; the numbered
    // fallback covers browsers that still withhold them.
    opt.textContent = cam.label || `Camera ${i + 1}`;
    opt.selected = cam.deviceId === activeId;
    els.cameraSel.appendChild(opt);
  });
  els.cameraSel.classList.remove("hidden");
}

els.cameraSel.addEventListener("change", async () => {
  const previousId = currentCameraId();
  const newId = els.cameraSel.value;
  els.cameraSel.disabled = true;
  setStatus("switching camera…");
  try {
    await startCamera(newId);
    localStorage.setItem(CAMERA_STORE_KEY, newId);
    // Fresh camera, fresh votes — but the spelled strip is the user's work
    // and survives the switch on purpose.
    smoother.reset();
    setStatus("waiting for hand…");
  } catch (err) {
    // Demo safety: failing to switch must not leave the app with NO camera.
    // Restore the previous one; only if that also fails, give up honestly.
    console.error("Camera switch failed:", err);
    try {
      await startCamera(previousId);
      els.cameraSel.value = previousId;
      setStatus("could not switch camera — kept the previous one", true);
    } catch {
      setStatus("Camera unavailable. Reload the page to retry.", true);
    }
  } finally {
    els.cameraSel.disabled = false;
  }
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
    await loadMode(modeId);
    activeMode = MODES[modeId];
    model = modelCache[modeId];
    labels = labelsCache[modeId];
    // Fresh mode, fresh votes and strip — mirrors main.py resetting the
    // smoother, buffer and word FSM when the desktop app changes mode.
    smoother.reset();
    charBuffer.clear();
    resetWordState();
    els.strip.textContent = "";
    applyModeUi(activeMode);
    setStatus("waiting for hand…");
  } catch (err) {
    // A failed switch must not leave the app modeless: activeMode was not
    // reassigned (the load threw first), so we stay on the previous mode.
    console.error("Mode switch failed:", err);
    applyModeUi(activeMode);
    setStatus(`could not load ${MODES[modeId].label} — kept ${activeMode.label}`, true);
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
    setStatus("loading model…");
    await loadMode(activeMode.id);
    model = modelCache[activeMode.id];
    labels = labelsCache[activeMode.id];
    applyModeUi(activeMode);

    setStatus("loading hand detector…");
    // Module-scoped so ensurePose() can reuse it when word mode is first entered.
    vision = await FilesetResolver.forVisionTasks(MEDIAPIPE_WASM_URL);
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

    setStatus("requesting camera…");
    video = document.createElement("video");
    video.playsInline = true; // iOS: play inline instead of fullscreen
    video.muted = true;

    // Prefer the camera the user picked last time; if it is gone (unplugged)
    // fall back to the default instead of dying. A permission error is not a
    // device problem, so it does not clear the saved choice.
    const savedId = localStorage.getItem(CAMERA_STORE_KEY);
    try {
      await startCamera(savedId);
    } catch (err) {
      if (!savedId || err.name === "NotAllowedError") throw err;
      localStorage.removeItem(CAMERA_STORE_KEY);
      await startCamera(null);
    }

    // Device labels only exist after permission was granted, so the selector
    // is built now, not at page load. Refresh it if cameras (un)plug.
    await refreshCameraList();
    navigator.mediaDevices.addEventListener("devicechange", refreshCameraList);

    setStatus("waiting for hand…");
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
