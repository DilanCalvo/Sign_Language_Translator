/**
 * Voice output — the web counterpart of src/voice.py, built on the Web Speech
 * API (speechSynthesis). Same product decisions as the desktop app: speaking
 * never blocks recognition, and what gets spoken is decided by the caller
 * (pipeline hooks on the commit frames).
 *
 * Web-specific realities this module absorbs:
 *  - getVoices() is often EMPTY at load and fills in later (voiceschanged).
 *    iOS sometimes never fires the event, so a short poll backs it up.
 *  - iOS/Safari require the first utterance to start inside a user gesture;
 *    a silent unlock utterance on the first pointerdown covers every later
 *    programmatic speak().
 *  - Chrome occasionally wedges the queue in a paused state; resume() before
 *    each speak un-sticks it.
 *
 * Prefs (localStorage, "asl-web." namespace): tts.enabled, tts.voiceURI,
 * tts.letterMode ("word" = speak fingerspelled words when they close,
 * "letter" = speak each committed letter; default "word").
 */

import { prefGet, prefSet } from "./prefs.js";

class Speech {
  constructor() {
    this.supported = typeof window !== "undefined" && "speechSynthesis" in window;
    this.enabled = prefGet("tts.enabled", true);
    this.letterMode = prefGet("tts.letterMode", "word");
    this.voices = [];
    this._voice = null;
    this._savedVoiceURI = prefGet("tts.voiceURI", null);
    this.onVoicesChanged = null; // settings UI hook (repopulate the picker)
  }

  init() {
    if (!this.supported) return;
    const refresh = () => {
      const list = speechSynthesis.getVoices();
      if (list.length === 0) return;
      this.voices = list;
      this._pickVoice();
      if (this.onVoicesChanged) this.onVoicesChanged();
    };
    refresh();
    speechSynthesis.addEventListener("voiceschanged", refresh);
    // iOS: voiceschanged is unreliable — poll for up to ~3s as a backup.
    let tries = 0;
    const poll = setInterval(() => {
      if (this.voices.length > 0 || ++tries > 10) clearInterval(poll);
      else refresh();
    }, 300);
    // First user gesture unlocks synthesis on iOS/Safari for all later calls.
    document.addEventListener("pointerdown", () => this._unlock(), { once: true });
  }

  /** Saved voice if still present -> en-US -> any en-* -> platform default. */
  _pickVoice() {
    const byURI = this._savedVoiceURI
      && this.voices.find((v) => v.voiceURI === this._savedVoiceURI);
    this._voice = byURI
      || this.voices.find((v) => v.lang === "en-US")
      || this.voices.find((v) => v.lang && v.lang.startsWith("en"))
      || this.voices.find((v) => v.default)
      || this.voices[0]
      || null;
  }

  _unlock() {
    if (!this.supported) return;
    try {
      const u = new SpeechSynthesisUtterance(" ");
      u.volume = 0;
      speechSynthesis.speak(u);
    } catch { /* unlock is best-effort */ }
  }

  /**
   * Speak `text` (non-blocking). `interrupt` cancels anything already
   * queued/speaking first — used for per-letter feedback, which must stay
   * snappy rather than queue up behind itself.
   */
  speak(text, { interrupt = false } = {}) {
    if (!this.supported || !this.enabled || !text) return;
    try {
      speechSynthesis.resume(); // un-wedge Chrome's occasional paused state
      if (interrupt || speechSynthesis.pending) speechSynthesis.cancel();
      const u = new SpeechSynthesisUtterance(String(text));
      if (this._voice) {
        u.voice = this._voice;
        u.lang = this._voice.lang;
      } else {
        u.lang = "en-US"; // glosses are English even before voices load
      }
      u.rate = 1.05;
      speechSynthesis.speak(u);
    } catch { /* a failed utterance must never break recognition */ }
  }

  get activeVoiceURI() {
    return this._voice ? this._voice.voiceURI : null;
  }

  setEnabled(on) {
    this.enabled = !!on;
    prefSet("tts.enabled", this.enabled);
    if (!on && this.supported) {
      try { speechSynthesis.cancel(); } catch { /* already idle */ }
    }
  }

  setVoice(voiceURI) {
    this._savedVoiceURI = voiceURI;
    prefSet("tts.voiceURI", voiceURI);
    this._pickVoice();
  }

  setLetterMode(mode) {
    this.letterMode = mode === "letter" ? "letter" : "word";
    prefSet("tts.letterMode", this.letterMode);
  }
}

export const speech = new Speech();
speech.init();
