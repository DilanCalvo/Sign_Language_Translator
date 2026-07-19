/**
 * UI glue for everything that is not the recognition pipeline: the
 * conversation panel (chat, REC/export/clear), the hearing side's compose bar
 * (typing + dictation), the big on-stage message, the help/settings dialogs,
 * the voice controls, and the commit feedback flash.
 *
 * pipeline.js imports { commitFeedback } from here (and this module for its
 * side effects); this module never imports pipeline.js — the recognition loop
 * talks to the world through speech/conversation, keeping the dependency
 * one-way.
 */

import { speech } from "./tts.js";
import { SpeechInput } from "./stt.js";
import { conversation, exportLog, SOURCE_SIGN } from "./conversation.js";

const $ = (id) => document.getElementById(id);

const stage = $("stage");
const strip = $("strip");
const chatList = $("chat-list");
const chatEmpty = $("chat-empty");
const recBtn = $("rec-btn");
const recBadge = $("rec-badge");
const exportBtn = $("export-btn");
const clearLogBtn = $("clear-log-btn");
const composeInput = $("compose-input");
const sendBtn = $("send-btn");
const sttBtn = $("stt-btn");
const bigMessage = $("big-message");
const ttsBtn = $("tts-btn");
const settingsBtn = $("settings-btn");
const helpBtn = $("help-btn");
const helpDialog = $("help-dialog");
const settingsDialog = $("settings-dialog");
const ttsEnabledInput = $("tts-enabled");
const ttsVoiceSelect = $("tts-voice");

// ==========================================================================
// Commit feedback: a visible (and, on Android, tactile) pulse the instant a
// sign is accepted. On phones — where recognition inevitably feels slower —
// knowing EXACTLY when a sign landed is half the UX.
// ==========================================================================

export function commitFeedback() {
  stage.classList.remove("commit-flash");
  void stage.offsetWidth; // restart the animation even mid-flash
  stage.classList.add("commit-flash");
  // Secondary cue right where the text lands: the caption blinks white.
  strip.animate(
    [{ color: "#ffffff" }, { color: "#ffd900" }],
    { duration: 350, easing: "ease-out" },
  );
  if (navigator.vibrate) navigator.vibrate(30);
}
stage.addEventListener("animationend", (ev) => {
  if (ev.animationName === "commit-ring") stage.classList.remove("commit-flash");
});

// ==========================================================================
// Chat panel
// ==========================================================================

const pad2 = (n) => String(n).padStart(2, "0");

function appendMessage(entry) {
  chatEmpty.classList.add("hidden");

  const li = document.createElement("div");
  li.className = `msg ${entry.source === SOURCE_SIGN ? "sign" : "hearing"}`;

  const meta = document.createElement("div");
  meta.className = "meta";
  const who = document.createElement("span");
  who.className = "who";
  who.textContent = entry.source === SOURCE_SIGN ? "Signs" : "Hearing";
  const time = document.createElement("time");
  const d = entry.when;
  time.textContent = `${pad2(d.getHours())}:${pad2(d.getMinutes())}:${pad2(d.getSeconds())}`;
  meta.append(who, time);
  if (entry.recorded) {
    const tape = document.createElement("span");
    tape.className = "tape";
    tape.textContent = "●";
    tape.title = "In the recorded transcript";
    meta.append(tape);
  }

  const text = document.createElement("div");
  text.className = "text";
  text.textContent = entry.text; // textContent only — never markup

  li.append(meta, text);

  // Stick to the newest message unless the user scrolled up to read history.
  const nearBottom =
    chatList.scrollHeight - chatList.scrollTop - chatList.clientHeight < 60;
  chatList.appendChild(li);
  if (nearBottom) chatList.scrollTop = chatList.scrollHeight;
}

function refreshExportButton() {
  exportBtn.disabled = conversation.recordedEntries().length === 0;
}

conversation.subscribe((event, payload) => {
  if (event === "add") {
    appendMessage(payload);
    refreshExportButton();
  } else if (event === "clear") {
    chatList.replaceChildren(chatEmpty);
    chatEmpty.classList.remove("hidden");
    refreshExportButton();
  } else if (event === "recording") {
    recBtn.setAttribute("aria-pressed", payload ? "true" : "false");
    recBtn.lastChild.textContent = payload ? "Recording" : "Rec";
    recBadge.classList.toggle("hidden", !payload);
  }
});

recBtn.addEventListener("click", () => {
  conversation.setRecording(!conversation.recording);
});

exportBtn.addEventListener("click", () => {
  exportLog();
});

// Clear is destructive: arm on the first tap, fire on the second.
let clearArmTimer = null;
clearLogBtn.addEventListener("click", () => {
  if (clearLogBtn.dataset.armed) {
    clearTimeout(clearArmTimer);
    delete clearLogBtn.dataset.armed;
    clearLogBtn.style.color = "";
    clearLogBtn.title = "Clear the conversation";
    conversation.clear();
    return;
  }
  clearLogBtn.dataset.armed = "1";
  clearLogBtn.style.color = "var(--err)";
  clearLogBtn.title = "Tap again to clear the whole conversation";
  clearArmTimer = setTimeout(() => {
    delete clearLogBtn.dataset.armed;
    clearLogBtn.style.color = "";
    clearLogBtn.title = "Clear the conversation";
  }, 2500);
});

// ==========================================================================
// Hearing side: typing + dictation + the big on-stage message
// ==========================================================================

let bigMessageTimer = null;

function showBigMessage(text) {
  bigMessage.textContent = text;
  bigMessage.classList.remove("hidden");
  clearTimeout(bigMessageTimer);
  bigMessageTimer = setTimeout(() => bigMessage.classList.add("hidden"), 8000);
}
bigMessage.addEventListener("click", () => bigMessage.classList.add("hidden"));

function sendHearingMessage(text) {
  text = (text || "").trim();
  if (!text) return;
  conversation.addHearing(text);
  showBigMessage(text); // the signer reads it without leaving the camera
}

sendBtn.addEventListener("click", () => {
  sendHearingMessage(composeInput.value);
  composeInput.value = "";
  composeInput.focus();
});
composeInput.addEventListener("keydown", (ev) => {
  if (ev.key === "Enter" && !ev.isComposing) {
    sendHearingMessage(composeInput.value);
    composeInput.value = "";
  }
});

// Dictation: shown only where SpeechRecognition exists; a failed engine only
// ever costs the mic button, never the typing path.
if (SpeechInput.supported) {
  const stt = new SpeechInput({
    lang: "en-US",
    onInterim: (text) => { composeInput.value = text; },
    onFinal: (text) => {
      composeInput.value = "";
      sendHearingMessage(text);
    },
    onState: (state) => {
      sttBtn.classList.toggle("listening", state === "listening");
      if (state === "error") {
        composeInput.placeholder = "Mic unavailable — type instead";
        setTimeout(() => { composeInput.placeholder = "Type to the signer…"; }, 3000);
      }
    },
  });
  sttBtn.classList.remove("hidden");
  sttBtn.addEventListener("click", () => stt.toggle());
}

// ==========================================================================
// Voice controls (toolbar toggle + settings dialog)
// ==========================================================================

function syncTtsUi() {
  ttsBtn.setAttribute("aria-pressed", speech.enabled ? "true" : "false");
  ttsEnabledInput.checked = speech.enabled;
}

ttsBtn.addEventListener("click", () => {
  speech.setEnabled(!speech.enabled);
  syncTtsUi();
});
ttsEnabledInput.addEventListener("change", () => {
  speech.setEnabled(ttsEnabledInput.checked);
  syncTtsUi();
});

function populateVoices() {
  const voices = speech.voices;
  ttsVoiceSelect.replaceChildren();
  if (voices.length === 0) {
    const opt = document.createElement("option");
    opt.textContent = "No voices on this device";
    opt.disabled = true;
    ttsVoiceSelect.appendChild(opt);
    ttsVoiceSelect.disabled = true;
    return;
  }
  ttsVoiceSelect.disabled = false;
  // English voices first (the glosses are English), then everything else.
  const en = voices.filter((v) => v.lang && v.lang.startsWith("en"));
  const rest = voices.filter((v) => !v.lang || !v.lang.startsWith("en"));
  for (const v of [...en, ...rest]) {
    const opt = document.createElement("option");
    opt.value = v.voiceURI;
    opt.textContent = `${v.name} (${v.lang})`;
    opt.selected = v.voiceURI === speech.activeVoiceURI;
    ttsVoiceSelect.appendChild(opt);
  }
}
speech.onVoicesChanged = populateVoices;
populateVoices();

ttsVoiceSelect.addEventListener("change", () => {
  speech.setVoice(ttsVoiceSelect.value);
});

for (const radio of document.querySelectorAll("input[name='tts-letter-mode']")) {
  radio.checked = radio.value === speech.letterMode;
  radio.addEventListener("change", () => {
    if (radio.checked) speech.setLetterMode(radio.value);
  });
}
syncTtsUi();

// ==========================================================================
// Dialogs
// ==========================================================================

helpBtn.addEventListener("click", () => helpDialog.showModal());
settingsBtn.addEventListener("click", () => {
  populateVoices(); // late-loading voices (iOS) land here too
  settingsDialog.showModal();
});
for (const btn of document.querySelectorAll(".dialog-close")) {
  btn.addEventListener("click", () => $(btn.dataset.close).close());
}
// Tap on the backdrop closes (the dialog element itself is the backdrop hit).
for (const dlg of [helpDialog, settingsDialog]) {
  dlg.addEventListener("click", (ev) => {
    if (ev.target === dlg) dlg.close();
  });
}

