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
// Vendored copy of @mediapipe/tasks-vision@0.10.14 (see web/vendor/): the app
// must not depend on a CDN being reachable at demo time, and the runtime
// version stays pinned to the one the .task file was validated with.
import {
  FilesetResolver,
  HandLandmarker,
  DrawingUtils,
} from "../vendor/mediapipe/vision_bundle.mjs";

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
    prediction: top1Conf >= LETTER_CONFIDENCE_THRESHOLD ? labels[order[0]] : null,
    confidence: top1Conf,
    top3,
  };
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
      clsResult = classify(flat, w / h);
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
