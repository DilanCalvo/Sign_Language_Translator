/**
 * Speech input for the hearing side — the web counterpart of the desktop's
 * push-to-talk Whisper (src/speech_input.py), built on the Web Speech API's
 * SpeechRecognition. One tap starts a single phrase; the final transcript is
 * delivered through onFinal and recognition stops on its own.
 *
 * Availability is genuinely spotty (no Firefox; Chrome's engine is
 * server-backed and needs a network), so the UI must feature-detect via
 * `SpeechInput.supported` and simply hide the mic when absent — typing is
 * always available as the reliable path.
 */

const SR = typeof window !== "undefined"
  ? (window.SpeechRecognition || window.webkitSpeechRecognition)
  : null;

export class SpeechInput {
  static supported = !!SR;

  /**
   * @param {object} handlers
   *   onFinal(text)    — a finished transcript (send it)
   *   onInterim(text)  — live partial transcript (preview it)
   *   onState(state)   — "listening" | "idle" | "error"
   */
  constructor({ lang = "en-US", onFinal, onInterim, onState } = {}) {
    this._onFinal = onFinal || (() => {});
    this._onInterim = onInterim || (() => {});
    this._onState = onState || (() => {});
    this.listening = false;
    if (!SpeechInput.supported) return;

    this._rec = new SR();
    this._rec.lang = lang;
    this._rec.continuous = false;      // one phrase per tap
    this._rec.interimResults = true;
    this._rec.maxAlternatives = 1;

    this._rec.onresult = (ev) => {
      let interim = "";
      let final = "";
      for (let i = ev.resultIndex; i < ev.results.length; i++) {
        const r = ev.results[i];
        if (r.isFinal) final += r[0].transcript;
        else interim += r[0].transcript;
      }
      if (interim) this._onInterim(interim.trim());
      if (final.trim()) this._onFinal(final.trim());
    };
    this._rec.onend = () => {
      this.listening = false;
      this._onState("idle");
    };
    this._rec.onerror = (ev) => {
      this.listening = false;
      // "no-speech"/"aborted" are normal outcomes of an idle mic, not errors.
      this._onState(ev.error === "no-speech" || ev.error === "aborted"
        ? "idle" : "error");
    };
  }

  start() {
    if (!SpeechInput.supported || this.listening) return;
    try {
      this._rec.start();
      this.listening = true;
      this._onState("listening");
    } catch {
      this._onState("error");
    }
  }

  stop() {
    if (!SpeechInput.supported || !this.listening) return;
    try { this._rec.stop(); } catch { /* already stopping */ }
  }

  toggle() {
    if (this.listening) this.stop();
    else this.start();
  }
}
