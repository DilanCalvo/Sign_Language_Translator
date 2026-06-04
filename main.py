import cv2

from config import (
    MODE,
    LETTER_SMOOTH_WINDOW, LETTER_SMOOTH_MIN_VOTES,
    WORD_SMOOTH_WINDOW, WORD_SMOOTH_MIN_VOTES,
    WORD_COOLDOWN_FRAMES,
    MOTION_LETTERS,
    VOICE_ENABLED,
    SPEECH_INPUT_ENABLED,
    SPEECH_WHISPER_MODEL,
    SPEECH_LANGUAGE,
)
from src.classifier import Classifier
from src.detector import Detector
from src.overlay import LetterBuffer, SpeechBuffer
from src.speech_input import SpeechInput, SpeechState
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


def _draw_hud(frame, stable, raw_result, stable_word=None,
              hand_is_signing=False, mode="letters", num_hands=0,
              speech_enabled=False):
    h, w = frame.shape[:2]

    raw_pred   = raw_result["prediction"]
    confidence = raw_result["confidence"]
    runner_up  = raw_result["runner_up"]

    # -----------------------------------------------------------------------
    # Raw per-frame prediction (top-right, small).
    # Suppressed while the hand is signing — word model takes priority.
    # -----------------------------------------------------------------------
    if raw_pred is not None and not hand_is_signing:
        raw_text = raw_pred.upper()
        (tw, _), _ = cv2.getTextSize(raw_text, cv2.FONT_HERSHEY_SIMPLEX, 1.1, 2)
        lx = max(w // 2, w - tw - 12)
        cv2.putText(frame, "raw:", (lx, 24),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, _GRAY, 1, cv2.LINE_AA)
        cv2.putText(frame, raw_text, (lx, 52),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.1, _GRAY, 2, cv2.LINE_AA)

    # -----------------------------------------------------------------------
    # Stable letter prediction (center, large). Suppressed while signing.
    # -----------------------------------------------------------------------
    if stable is not None and not hand_is_signing:
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

        # Runner-up: second candidate when it carries meaningful weight
        if runner_up is not None:
            ru_text = f"  /{runner_up['prediction'].upper()}  {runner_up['confidence']*100:.0f}%"
            cv2.putText(frame, ru_text, (bar_x - 10, bar_y + 32),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, _YELLOW, 1, cv2.LINE_AA)

    # -----------------------------------------------------------------------
    # Detected word — centered bottom strip with semi-transparent background.
    # -----------------------------------------------------------------------
    if stable_word is not None:
        word_text = stable_word.upper()
        font_scale = 2.0
        thick = 3
        (tw, th), _ = cv2.getTextSize(word_text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thick)
        bar_h = th + 40
        y0 = h - bar_h
        roi = frame[y0:h, 0:w]
        dark = roi.copy()
        dark[:] = (0, 20, 0)
        cv2.addWeighted(dark, 0.60, roi, 0.40, 0, roi)
        frame[y0:h, 0:w] = roi
        wx = max(8, (w - tw) // 2)
        wy = h - 20
        cv2.putText(frame, word_text, (wx + 2, wy + 2),
                    cv2.FONT_HERSHEY_SIMPLEX, font_scale, _BLACK, thick + 4, cv2.LINE_AA)
        cv2.putText(frame, word_text, (wx, wy),
                    cv2.FONT_HERSHEY_SIMPLEX, font_scale, _GREEN, thick, cv2.LINE_AA)

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

    # Tense selector; updated at runtime by T (past) / R (present) / F (future).
    tense = "present"

    detector      = Detector()
    classifier    = Classifier()
    smoother      = PredictionSmoother(window=LETTER_SMOOTH_WINDOW, min_votes=LETTER_SMOOTH_MIN_VOTES)
    word_smoother = PredictionSmoother(window=WORD_SMOOTH_WINDOW, min_votes=WORD_SMOOTH_MIN_VOTES)
    letter_buffer = LetterBuffer()
    speech_buffer = SpeechBuffer()
    voice         = VoiceOutput() if VOICE_ENABLED else None

    speech_input: SpeechInput | None = None
    if SPEECH_INPUT_ENABLED:
        speech_input = SpeechInput(
            model_size=SPEECH_WHISPER_MODEL,
            language=SPEECH_LANGUAGE,
        )

    # Word cooldown state.
    locked_word     = None
    cooldown_frames = 0

    print("Camera started.")
    print("  Q: quit  |  L: letters  |  W: words  |  N: numbers")
    print("  T: past tense  |  R: present tense (default)  |  F: future tense")
    if speech_input is not None:
        print("  P: toggle microphone  (push-to-talk speech recognition)")

    while True:
        frame, landmarks_data = detector.get_frame()

        if frame is None:
            print("Failed to read frame.")
            break

        raw_result      = classifier.classify(landmarks_data)
        hand_is_signing = raw_result["hand_is_signing"]

        # ----- Letters (active in "letters" and "numbers" modes) -----
        if mode in ("letters", "numbers"):
            if hand_is_signing:
                smoother.reset()
            else:
                smoother.update(raw_result["prediction"])
            stable = smoother.get_stable()
            if letter_buffer.update(stable) and stable and voice:
                voice.speak(stable)
        else:
            smoother.reset()
            stable = None

        # ----- Words with cooldown (active only in "words" mode) -----
        if mode == "words":
            word_raw          = raw_result["word_prediction"]
            new_word_detected = False
            if cooldown_frames > 0:
                cooldown_frames -= 1
                word_smoother.reset()
                stable_word = locked_word
                if cooldown_frames == 0:
                    locked_word = None
            else:
                word_smoother.update(word_raw["prediction"] if word_raw is not None else None)
                stable_word = word_smoother.get_stable()
                if stable_word is not None:
                    locked_word       = stable_word
                    cooldown_frames   = WORD_COOLDOWN_FRAMES
                    new_word_detected = True
            if new_word_detected and voice:
                voice.speak(stable_word)
        else:
            word_smoother.reset()
            stable_word = None

        # In "words" mode always suppress the raw letter display.
        effective_signing = hand_is_signing or (mode == "words")

        # ----- Speech-to-text transcript -----
        if speech_input is not None:
            transcript = speech_input.get_transcript()
            if transcript:
                speech_buffer.add(transcript)

        # ----- Draw -----
        _draw_hud(
            frame, stable, raw_result, stable_word, effective_signing, mode,
            landmarks_data["num_hands"],
            speech_enabled=(speech_input is not None),
        )
        if mode in ("letters", "numbers"):
            letter_buffer.draw_subtitle(frame, override_tense=tense)

        speech_status = speech_input.state.name.lower() if speech_input else ""
        speech_buffer.draw(frame, speech_status)

        cv2.imshow("Sign Language Translator", frame)

        # ----- Keyboard -----
        key = cv2.waitKey(1) & 0xFF

        if key == ord('q'):
            break

        elif key == ord('l'):
            mode = "letters"
            smoother.reset()
            word_smoother.reset()
            letter_buffer.clear()
            locked_word     = None
            cooldown_frames = 0

        elif key == ord('w'):
            mode = "words"
            smoother.reset()
            word_smoother.reset()
            letter_buffer.clear()
            locked_word     = None
            cooldown_frames = 0

        elif key == ord('n'):
            mode = "numbers"
            smoother.reset()
            word_smoother.reset()
            letter_buffer.clear()
            locked_word     = None
            cooldown_frames = 0

        # ----- Tense selection (T=past, R=present, F=future) -----
        elif key == ord('t'):
            tense = "past"
            print("[TENSE] Past (creí, creyó, creyeron)")

        elif key == ord('r'):
            tense = "present"
            print("[TENSE] Present (creo, cree, creen)")

        elif key == ord('f'):
            tense = "future"
            print("[TENSE] Future (creeré, creerá, creerán)")

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
