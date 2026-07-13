import time

import cv2

# Load variables from a local .env (e.g. ANTHROPIC_API_KEY for the translation
# layer) into the environment BEFORE the modules that read them. Optional and
# silent: if python-dotenv is not installed or there is no .env, the app still
# runs and the translator simply stays in offline/fallback mode.
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

from config import (
    MODE,
    LETTER_SMOOTH_WINDOW, LETTER_SMOOTH_MIN_VOTES,
    WORD_SMOOTH_WINDOW, WORD_SMOOTH_MIN_VOTES,
    WORD_COOLDOWN_SECONDS,
    MOTION_LETTERS,
    LOW_CONFIDENCE_THRESHOLD,
    VOICE_ENABLED,
    SPEECH_INPUT_ENABLED,
    SPEECH_WHISPER_MODEL,
    SPEECH_LANGUAGE,
    TRANSLATION_ENABLED,
    TRANSLATION_TARGET_LANGUAGE,
    TRANSLATION_MODEL,
    TRANSLATION_OFFLINE_FALLBACK,
)
from src.classifier import Classifier
from src.conversation_log import ConversationLog
from src.detector import Detector
from src.overlay import LetterBuffer, SpeechBuffer, WordBuffer
from src.speech_input import SpeechInput, SpeechState
from src.translator import Translator
from src.utils import PredictionSmoother
from src.voice import VoiceOutput

# BGR colors
_WHITE  = (255, 255, 255)
_BLACK  = (0,   0,   0)
_GREEN  = (0,   220, 0)
_YELLOW = (0,   210, 255)
_GRAY   = (160, 160, 160)
_RED    = (60,  60,  220)
_CYAN   = (220, 220, 0)

_MODE_LABELS = {
    "letters": "LETTERS",
    "words":   "WORDS",
    "numbers": "NUMBERS",
}


class WordCommitFSM:
    """Vote -> confirm -> lock-out state machine for dynamic word signs.

    Owns the word smoother and the cooldown clock so the main loop holds no
    loose word state. step() returns (word, confidence, is_new): is_new is True
    only on the frame a fresh sign is committed (the moment to speak/log it).
    During the cooldown the last committed word stays on screen and no new word
    is accepted.
    """

    def __init__(self, smoother, cooldown_seconds):
        self._smoother = smoother
        self._cooldown = cooldown_seconds
        self.reset()

    def reset(self):
        self._smoother.reset()
        self._locked = None
        self._locked_conf = 0.0
        self._cooldown_until = 0.0

    def step(self, word_pred, now):
        if now < self._cooldown_until:
            self._smoother.reset()
            return self._locked, self._locked_conf, False

        self._locked = None
        self._smoother.update(word_pred["prediction"] if word_pred is not None else None)
        stable = self._smoother.get_stable()
        if stable is not None:
            self._locked = stable
            self._locked_conf = word_pred["confidence"] if word_pred is not None else 0.0
            self._cooldown_until = now + self._cooldown
            return stable, self._locked_conf, True
        return None, 0.0, False


def _draw_confidence_bar(frame, x, y, width, confidence):
    """Horizontal bar proportional to confidence. Green > 80%, yellow > 60%, red otherwise."""
    filled = int(width * confidence)
    if confidence >= 0.80:
        color = _GREEN
    elif confidence >= 0.60:
        color = _YELLOW
    else:
        color = _RED
    cv2.rectangle(frame, (x, y), (x + width, y + 10), (50, 50, 50), -1)
    cv2.rectangle(frame, (x, y), (x + filled, y + 10), color, -1)
    cv2.rectangle(frame, (x, y), (x + width, y + 10), _GRAY, 1)


def _draw_alternatives(frame, top3):
    """
    Centered panel listing the top candidates when the model is unsure.

    Shown only in the low-confidence regime (< LOW_CONFIDENCE_THRESHOLD): the
    model has no single trustworthy guess, so we surface what it is hesitating
    between instead of committing to one letter.
    """
    h, w = frame.shape[:2]
    title = "not sure - did you mean:"
    cv2.putText(frame, title, (w // 2 - 130, h // 2 - 50),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, _YELLOW, 1, cv2.LINE_AA)
    for i, cand in enumerate(top3):
        line = f"{i + 1}.  {cand['prediction'].upper()}    {cand['confidence'] * 100:.0f}%"
        y = h // 2 - 10 + i * 36
        cv2.putText(frame, line, (w // 2 - 110 + 3, y + 3),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, _BLACK, 4, cv2.LINE_AA)
        color = _WHITE if i == 0 else _GRAY
        cv2.putText(frame, line, (w // 2 - 110, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2, cv2.LINE_AA)


def _draw_translation(frame, text, status):
    """
    Draw the LLM-translated sentence as a cyan banner above the word subtitle.

    status: "translating" shows a spinner-ish hint; "" hides the bar unless
    there is text to show (the last translation lingers until the next one).
    """
    if not text and status != "translating":
        return
    h, w = frame.shape[:2]
    bar_h = 50
    y1 = h - 96          # sits just above the bottom word-subtitle bar
    y0 = y1 - bar_h
    roi = frame[y0:y1, 0:w]
    tint = roi.copy(); tint[:] = (60, 40, 10)   # dark teal/navy
    cv2.addWeighted(tint, 0.6, roi, 0.4, 0, roi)
    frame[y0:y1, 0:w] = roi

    label = "translating..." if status == "translating" else "translation:"
    cv2.putText(frame, label, (12, y0 + 18),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, _CYAN, 1, cv2.LINE_AA)
    if text:
        scale, thick = 0.8, 2
        (tw, _), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, thick)
        while tw > w - 24 and scale > 0.45:
            scale -= 0.05
            (tw, _), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, thick)
        cv2.putText(frame, text, (12, y1 - 12),
                    cv2.FONT_HERSHEY_SIMPLEX, scale, _WHITE, thick, cv2.LINE_AA)


def _draw_hud(frame, stable, raw_result, stable_word=None, stable_word_conf=0.0,
              hand_is_signing=False, mode="letters", num_hands=0,
              speech_enabled=False, numbers_unavailable=False,
              words_unavailable=False):
    h, w = frame.shape[:2]

    top3       = raw_result["top3"]
    confidence = raw_result["confidence"]
    raw_top    = top3[0]["prediction"] if top3 else None

    show_static = not hand_is_signing and not numbers_unavailable

    # -----------------------------------------------------------------------
    # Mode selected but its model is missing: be honest about it instead of
    # showing a blank screen (letters are suppressed in words mode, so without
    # this notice a missing word model would look like a frozen app).
    # -----------------------------------------------------------------------
    unavailable_msg = None
    if numbers_unavailable:
        unavailable_msg  = "Numbers model not trained yet"
        unavailable_hint = "capture 0-9 and train model/model_numbers.h5"
    elif words_unavailable:
        unavailable_msg  = "Word model not trained yet"
        unavailable_hint = "recapture: python capture/capture_words.py, then train_words.py"
    if unavailable_msg:
        (mw, _), _ = cv2.getTextSize(unavailable_msg, cv2.FONT_HERSHEY_SIMPLEX, 0.9, 2)
        cv2.putText(frame, unavailable_msg, ((w - mw) // 2, h // 2 - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, _YELLOW, 2, cv2.LINE_AA)
        (hw, _), _ = cv2.getTextSize(unavailable_hint, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.putText(frame, unavailable_hint, ((w - hw) // 2, h // 2 + 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, _GRAY, 1, cv2.LINE_AA)

    # -----------------------------------------------------------------------
    # Raw per-frame top guess (top-right, small). Shown once the guess clears
    # the low-confidence floor, even before it is committed.
    # -----------------------------------------------------------------------
    if show_static and raw_top is not None and confidence >= LOW_CONFIDENCE_THRESHOLD:
        raw_text = raw_top.upper()
        (tw, _), _ = cv2.getTextSize(raw_text, cv2.FONT_HERSHEY_SIMPLEX, 1.1, 2)
        lx = max(w // 2, w - tw - 12)
        cv2.putText(frame, "raw:", (lx, 24),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, _GRAY, 1, cv2.LINE_AA)
        cv2.putText(frame, raw_text, (lx, 52),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.1, _GRAY, 2, cv2.LINE_AA)

    # -----------------------------------------------------------------------
    # Stable committed prediction (center, large). Suppressed while signing.
    # -----------------------------------------------------------------------
    if show_static and stable is not None:
        text  = stable.upper()
        scale = 5.0
        thick = 8
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, thick)
        x = (w - tw) // 2
        y = th + 20

        # Shadow for legibility over any background
        cv2.putText(frame, text, (x + 3, y + 3),
                    cv2.FONT_HERSHEY_SIMPLEX, scale, _BLACK, thick + 4, cv2.LINE_AA)
        cv2.putText(frame, text, (x, y),
                    cv2.FONT_HERSHEY_SIMPLEX, scale, _WHITE, thick, cv2.LINE_AA)

        # Motion-letter indicator
        if stable in MOTION_LETTERS:
            warn = "requires motion"
            cv2.putText(frame, warn, (max(4, x - 20), y + 45),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, _YELLOW, 1, cv2.LINE_AA)

        # Confidence bar centered below the letter
        bar_w = 160
        bar_x = (w - bar_w) // 2
        bar_y = y + 55
        _draw_confidence_bar(frame, bar_x, bar_y, bar_w, confidence)
        conf_label = f"{confidence * 100:.1f}%"
        (cl_w, _), _ = cv2.getTextSize(conf_label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
        cl_x = min(bar_x + bar_w + 8, w - cl_w - 4)
        cv2.putText(frame, conf_label, (cl_x, bar_y + 9),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, _GREEN, 1, cv2.LINE_AA)

    # -----------------------------------------------------------------------
    # Low-confidence: no committed guess -> show the top-3 alternatives so the
    # user can see (and re-sign toward) what the model is weighing.
    # -----------------------------------------------------------------------
    elif show_static and confidence < LOW_CONFIDENCE_THRESHOLD and top3:
        _draw_alternatives(frame, top3)

    # -----------------------------------------------------------------------
    # Current detected word — centered feedback with confidence bar. The
    # accumulated sentence is drawn separately at the bottom by
    # WordBuffer.draw_subtitle, so this stays in the upper-center to avoid
    # colliding with it.
    # -----------------------------------------------------------------------
    if stable_word is not None:
        word_text = stable_word.upper()
        font_scale = 2.2
        thick = 4
        (tw, th), _ = cv2.getTextSize(word_text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thick)
        wx = max(8, (w - tw) // 2)
        wy = th + 40
        cv2.putText(frame, word_text, (wx + 2, wy + 2),
                    cv2.FONT_HERSHEY_SIMPLEX, font_scale, _BLACK, thick + 4, cv2.LINE_AA)
        cv2.putText(frame, word_text, (wx, wy),
                    cv2.FONT_HERSHEY_SIMPLEX, font_scale, _GREEN, thick, cv2.LINE_AA)

        # Confidence bar for the detected word (below the text).
        bar_w = 160
        bar_x = (w - bar_w) // 2
        bar_y = wy + 24
        _draw_confidence_bar(frame, bar_x, bar_y, bar_w, stable_word_conf)
        conf_label = f"{stable_word_conf * 100:.0f}%"
        cv2.putText(frame, conf_label, (bar_x + bar_w + 8, bar_y + 9),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, _GREEN, 1, cv2.LINE_AA)

    # -----------------------------------------------------------------------
    # Active mode + hand indicator (top-left).
    # -----------------------------------------------------------------------
    mode_text  = f"MODE: {_MODE_LABELS.get(mode, mode.upper())}"
    chip_color = _GREEN if mode == "words" else _CYAN if mode == "letters" else _YELLOW
    cv2.putText(frame, mode_text, (12, 26),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, chip_color, 1, cv2.LINE_AA)
    (mt_w, _), _ = cv2.getTextSize(mode_text, cv2.FONT_HERSHEY_SIMPLEX, 0.65, 1)
    dot_color = _GREEN if num_hands > 0 else (60, 60, 180)
    cv2.circle(frame, (12 + mt_w + 10, 20), 5, dot_color, -1, cv2.LINE_AA)

    # Small microphone hint below mode indicator when speech input is active.
    if speech_enabled:
        cv2.putText(frame, "P: mic", (12, 46),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.40, _GRAY, 1, cv2.LINE_AA)


def main():
    # Local mutable mode; updated at runtime by L / W / N keyboard shortcuts.
    mode = MODE

    detector      = Detector()
    classifier    = Classifier()
    smoother      = PredictionSmoother(window=LETTER_SMOOTH_WINDOW, min_votes=LETTER_SMOOTH_MIN_VOTES)
    word_fsm      = WordCommitFSM(
        PredictionSmoother(window=WORD_SMOOTH_WINDOW, min_votes=WORD_SMOOTH_MIN_VOTES),
        WORD_COOLDOWN_SECONDS,
    )
    letter_buffer = LetterBuffer()
    word_buffer   = WordBuffer()
    speech_buffer = SpeechBuffer()
    conv_log      = ConversationLog()
    voice         = VoiceOutput() if VOICE_ENABLED else None

    speech_input: SpeechInput | None = None
    if SPEECH_INPUT_ENABLED:
        speech_input = SpeechInput(
            model_size=SPEECH_WHISPER_MODEL,
            language=SPEECH_LANGUAGE,
        )

    translator = Translator(
        model=TRANSLATION_MODEL,
        target_language=TRANSLATION_TARGET_LANGUAGE,
        enabled=TRANSLATION_ENABLED,
        offline_fallback=TRANSLATION_OFFLINE_FALLBACK,
    )

    # Translation display state (the LLM-produced sentence + its status).
    translation_text   = ""
    translation_status = ""   # "translating" while a request is in flight

    print("Camera started.")
    print("  Q: quit  |  L: letters  |  W: words  |  N: numbers")
    print("  E: export conversation log  |  T: translate signed sentence")
    if not translator.online and TRANSLATION_ENABLED:
        print(f"  [translate] offline mode ({translator.reason}); T joins glosses as-is")
    if speech_input is not None:
        print("  P: toggle microphone  (push-to-talk speech recognition)")

    while True:
        frame, landmarks_data = detector.get_frame()

        if frame is None:
            print("Failed to read frame.")
            break

        static_mode     = "numbers" if mode == "numbers" else "letters"
        raw_result      = classifier.classify(landmarks_data, static_mode=static_mode)
        hand_is_signing = raw_result["hand_is_signing"]

        # Numbers mode with no trained model: show a notice, skip prediction.
        numbers_unavailable = (mode == "numbers" and not classifier.has_numbers)
        # Same honesty for words mode: without a trained word model the screen
        # would otherwise just stay blank (letters are suppressed in this mode).
        words_unavailable = (mode == "words" and not classifier.has_words)

        # ----- Static signs (active in "letters" and "numbers" modes) -----
        if mode in ("letters", "numbers"):
            if hand_is_signing:
                smoother.reset()
            else:
                smoother.update(raw_result["prediction"])
            stable = smoother.get_stable()
            if letter_buffer.update(stable) and stable and voice:
                voice.speak(stable)
            # Completed spelled words go into the conversation log.
            for finished in letter_buffer.pop_completed_words():
                conv_log.add_sign(finished)
        else:
            smoother.reset()
            stable = None

        # ----- Words with cooldown (active only in "words" mode) -----
        if mode == "words":
            stable_word, word_conf, new_word_detected = word_fsm.step(
                raw_result["word_prediction"], time.time())
            if new_word_detected:
                # Add the sign to the running sentence and speak/log it.
                spoken = word_buffer.add(stable_word)
                if spoken:
                    if voice:
                        voice.speak(spoken)
                    conv_log.add_sign(spoken)
            word_buffer.tick()
        else:
            word_fsm.reset()
            stable_word = None
            word_conf = 0.0

        # In "words" mode always suppress the raw letter display.
        effective_signing = hand_is_signing or (mode == "words")

        # ----- Speech-to-text transcript -----
        if speech_input is not None:
            transcript = speech_input.get_transcript()
            if transcript:
                speech_buffer.add(transcript)
                conv_log.add_speech(transcript)

        # ----- Translation result (LLM gloss -> sentence) -----
        ready = translator.poll()
        if ready is not None:
            translation_text   = ready
            translation_status = ""
            if voice:
                voice.speak(ready)
            conv_log.add("Traduccion", ready)
        elif not translator.busy:
            translation_status = ""

        # ----- Draw -----
        _draw_hud(
            frame, stable, raw_result, stable_word, word_conf,
            effective_signing, mode,
            landmarks_data["num_hands"],
            speech_enabled=(speech_input is not None),
            numbers_unavailable=numbers_unavailable,
            words_unavailable=words_unavailable,
        )
        if mode in ("letters", "numbers"):
            letter_buffer.draw_subtitle(frame)
        elif mode == "words":
            _draw_translation(frame, translation_text, translation_status)
            word_buffer.draw_subtitle(frame)

        speech_status = speech_input.state.name.lower() if speech_input else ""
        speech_buffer.draw(frame, speech_status)

        cv2.imshow("Sign Language Translator", frame)

        # ----- Keyboard -----
        key = cv2.waitKey(1) & 0xFF

        if key == ord('q'):
            break

        elif key in (ord('l'), ord('w'), ord('n')):
            mode = {ord('l'): "letters", ord('w'): "words", ord('n'): "numbers"}[key]
            smoother.reset()
            word_fsm.reset()
            letter_buffer.clear()
            word_buffer.clear()
            translation_text   = ""
            translation_status = ""

        elif key == ord('e'):
            path = conv_log.export()
            if path:
                print(f"[LOG] Conversation exported ({len(conv_log)} entries) -> {path}")
            else:
                print("[LOG] Nothing to export yet.")

        elif key == ord('t'):
            # Translate the signed sentence so far (the gloss buffer) into a
            # fluent sentence via the LLM layer. Result arrives asynchronously
            # and is picked up by translator.poll() above.
            glosses = word_buffer.get_words()
            if glosses:
                if translator.submit(glosses):
                    translation_status = "translating"
                # If submit returned False but we are offline, the fallback was
                # already queued and poll() will pick it up next frame.
            else:
                print("[translate] No signs to translate yet.")

        elif key == ord('p') and speech_input is not None:
            if speech_input.state == SpeechState.IDLE:
                speech_input.start_recording()
            elif speech_input.state == SpeechState.LISTENING:
                speech_input.stop_recording()

    # ----- Cleanup -----
    detector.release()
    if voice:
        voice.stop()
    if speech_input is not None:
        speech_input.shutdown()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
