/**
 * Live pipeline: camera -> mirrored canvas -> MediaPipe hands -> normalize ->
 * letters model -> smoother -> UI. Mirrors main.py's letters flow:
 *
 *   raw = classify(frame)          (committed prediction: null below threshold)
 *   smoother.update(raw.prediction)
 *   stable = smoother.getStable()
 *   letterBuffer.update(stable)
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
  LETTER_CONFIDENCE_THRESHOLD,
  LOW_CONFIDENCE_THRESHOLD,
  ALT_MIN_CONFIDENCE,
  LETTER_SMOOTH_WINDOW,
  LETTER_SMOOTH_MIN_VOTES,
  LETTER_COOLDOWN_FRAMES,
  DETECTOR_MIN_DETECTION_CONFIDENCE,
  DETECTOR_MIN_PRESENCE_CONFIDENCE,
  DETECTOR_MIN_TRACKING_CONFIDENCE,
  NUM_HANDS,
  MODEL_URL,
  LABELS_URL,
  HAND_TASK_URL,
  MEDIAPIPE_WASM_URL,
} from "./config.js";
import { normalizeLandmarks, PredictionSmoother, LetterBuffer } from "./utils.js";
import { loadLettersModel } from "./model.js";
import {
  FilesetResolver,
  HandLandmarker,
  DrawingUtils,
} from "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14";

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
};
const ctx = els.canvas.getContext("2d");

function setStatus(text, isError = false) {
  els.status.textContent = text;
  els.status.classList.toggle("error", isError);
  els.status.classList.toggle("hidden", !text);
}

// ---- State ----
const smoother = new PredictionSmoother(LETTER_SMOOTH_WINDOW, LETTER_SMOOTH_MIN_VOTES);
const letterBuffer = new LetterBuffer(LETTER_COOLDOWN_FRAMES);
let model = null;
let labels = null;   // {0: "A", ...}
let landmarker = null;
let drawer = null;
let video = null;
let lastFrameTimes = [];

els.clearBtn.addEventListener("click", () => {
  letterBuffer.clear();
  els.strip.textContent = "";
});

// ---- Replica of src/classifier.py::_run (top-3 + acceptance threshold) ----
function classify(flat63) {
  const probs = model.predict(normalizeLandmarks(flat63));
  const order = [...probs.keys()].sort((a, b) => probs[b] - probs[a]);
  const top3 = order.slice(0, 3)
    .filter((i) => probs[i] >= ALT_MIN_CONFIDENCE)
    .map((i) => ({ prediction: labels[i], confidence: probs[i] }));
  const top1Conf = probs[order[0]];
  return {
    prediction: top1Conf >= LETTER_CONFIDENCE_THRESHOLD ? labels[order[0]] : null,
    confidence: top1Conf,
    top3,
  };
}

// ---- UI updates ----
function updateHud(stable, result) {
  if (stable !== null) {
    els.letter.textContent = stable;
  } else if (result && result.confidence >= LOW_CONFIDENCE_THRESHOLD) {
    // Between thresholds: show the raw guess small (uncommitted), like the HUD.
    els.letter.textContent = result.top3.length ? result.top3[0].prediction : "–";
  } else {
    els.letter.textContent = "–";
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

// ---- Per-frame ----
function onFrame(now) {
  const w = els.canvas.width, h = els.canvas.height;

  // Mirror BEFORE detection (parity with cv2.flip in src/detector.py).
  ctx.save();
  ctx.scale(-1, 1);
  ctx.drawImage(video, -w, 0, w, h);
  ctx.restore();

  const result = landmarker.detectForVideo(els.canvas, now);
  const hand = result.landmarks && result.landmarks[0];

  let clsResult = null;
  if (hand) {
    drawer.drawConnectors(hand, HandLandmarker.HAND_CONNECTIONS,
                          { color: "#ffffff", lineWidth: 2 });
    drawer.drawLandmarks(hand, { color: "#ffd900", radius: 3 });

    // Flatten x,y,z — same order as src/detector.py builds landmarks_hand1.
    const flat = new Array(63);
    for (let i = 0; i < 21; i++) {
      flat[i * 3] = hand[i].x;
      flat[i * 3 + 1] = hand[i].y;
      flat[i * 3 + 2] = hand[i].z;
    }
    clsResult = classify(flat);
    smoother.update(clsResult.prediction);
    setStatus("");
  } else {
    smoother.update(null);
    setStatus("waiting for hand…");
  }

  const stable = smoother.getStable();
  if (letterBuffer.update(stable)) {
    els.strip.textContent = letterBuffer.getText();
  }
  updateHud(stable, clsResult);
  updateFps(now);
  scheduleNext();
}

function scheduleNext() {
  // Per-camera-frame callback when supported (no wasted detections at 60Hz),
  // otherwise plain rAF.
  if (video.requestVideoFrameCallback) {
    video.requestVideoFrameCallback((now) => onFrame(now));
  } else {
    requestAnimationFrame((now) => onFrame(now));
  }
}

// ---- Boot ----
async function main() {
  if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
    setStatus("This browser does not support camera access. Try Chrome.", true);
    return;
  }

  try {
    setStatus("loading model…");
    [model, labels] = await Promise.all([
      loadLettersModel(MODEL_URL),
      fetch(LABELS_URL).then((r) => {
        if (!r.ok) throw new Error(`labels ${r.status}`);
        return r.json();
      }),
    ]);

    setStatus("loading hand detector…");
    const vision = await FilesetResolver.forVisionTasks(MEDIAPIPE_WASM_URL);
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
    } catch {
      // Some devices have no usable GPU delegate — retry on CPU instead of dying.
      options.baseOptions.delegate = "CPU";
      landmarker = await HandLandmarker.createFromOptions(vision, options);
    }
    drawer = new DrawingUtils(ctx);

    setStatus("requesting camera…");
    const stream = await navigator.mediaDevices.getUserMedia({
      video: { facingMode: "user", width: { ideal: 960 }, height: { ideal: 540 } },
      audio: false,
    });
    video = document.createElement("video");
    video.srcObject = stream;
    video.playsInline = true; // iOS: play inline instead of fullscreen
    video.muted = true;
    await video.play();

    els.canvas.width = video.videoWidth;
    els.canvas.height = video.videoHeight;
    // The CSS 16:9 aspect is only a pre-camera placeholder; the real camera
    // may be 4:3 (typical on phones) and must not be stretched.
    els.canvas.style.aspectRatio = `${video.videoWidth} / ${video.videoHeight}`;

    setStatus("waiting for hand…");
    scheduleNext();
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
