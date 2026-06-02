import cv2

from config import (
    MODE,
    LETTER_SMOOTH_WINDOW, LETTER_SMOOTH_MIN_VOTES,
    WORD_SMOOTH_WINDOW, WORD_SMOOTH_MIN_VOTES,
    WORD_COOLDOWN_FRAMES,
    MOTION_LETTERS,
)
from src.classifier import Classifier
from src.detector import Detector
from src.utils import PredictionSmoother

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


def _draw_hud(frame, stable, raw_result, stable_word=None, hand_is_signing=False, mode="letters"):
    h, w = frame.shape[:2]

    raw_pred   = raw_result["prediction"]
    confidence = raw_result["confidence"]
    runner_up  = raw_result["runner_up"]
    model_used = raw_result["model_used"]

    # -----------------------------------------------------------------------
    # Raw per-frame prediction (top-right, small).
    # Suppressed while the hand is signing — the word model takes priority and
    # we do not want letter noise on screen.
    # -----------------------------------------------------------------------
    if raw_pred is not None and not hand_is_signing:
        raw_text = raw_pred.upper()
        cv2.putText(frame, "frame:", (w - 145, 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, _GRAY, 1, cv2.LINE_AA)
        cv2.putText(frame, raw_text, (w - 70, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.6, _GRAY, 2, cv2.LINE_AA)

    # -----------------------------------------------------------------------
    # Stable letter prediction (center, large).
    # Suppressed while the hand is signing.
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
            cv2.putText(frame, warn, (x - 20, y + 45),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, _YELLOW, 1, cv2.LINE_AA)

        # Confidence bar centered below the letter
        bar_w = 160
        bar_x = (w - bar_w) // 2
        bar_y = y + 55
        _draw_confidence_bar(frame, bar_x, bar_y, bar_w, confidence)
        conf_label = f"{confidence * 100:.1f}%"
        cv2.putText(frame, conf_label, (bar_x + bar_w + 8, bar_y + 9),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, _GREEN, 1, cv2.LINE_AA)

        # Runner-up: second candidate when it carries meaningful weight
        if runner_up is not None:
            ru_text = f"  /{runner_up['prediction'].upper()}  {runner_up['confidence']*100:.0f}%"
            cv2.putText(frame, ru_text, (bar_x - 10, bar_y + 32),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, _YELLOW, 1, cv2.LINE_AA)

    # -----------------------------------------------------------------------
    # Detected word (bottom-left, green)
    # -----------------------------------------------------------------------
    if stable_word is not None:
        word_text = stable_word.upper()
        cv2.putText(frame, "WORD:", (12, h - 55),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, _GRAY, 1, cv2.LINE_AA)
        # Shadow
        cv2.putText(frame, word_text, (14, h - 18),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.8, _BLACK, 6, cv2.LINE_AA)
        cv2.putText(frame, word_text, (12, h - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.8, _GREEN, 3, cv2.LINE_AA)

    # -----------------------------------------------------------------------
    # Active mode (top-left)
    # -----------------------------------------------------------------------
    mode_text = f"MODE: {_MODE_LABELS.get(mode, mode.upper())}"
    chip_color = _GREEN if mode == "words" else _CYAN if mode == "letters" else _YELLOW
    cv2.putText(frame, mode_text, (12, 22),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, chip_color, 1, cv2.LINE_AA)


def main():
    detector      = Detector()
    classifier    = Classifier()
    smoother      = PredictionSmoother(window=LETTER_SMOOTH_WINDOW, min_votes=LETTER_SMOOTH_MIN_VOTES)
    word_smoother = PredictionSmoother(window=WORD_SMOOTH_WINDOW, min_votes=WORD_SMOOTH_MIN_VOTES)

    # Word cooldown state.
    # locked_word: word kept on screen during the cooldown.
    # cooldown_frames: countdown; while > 0, no new word is accepted.
    locked_word     = None
    cooldown_frames = 0

    print("Camera started. Press 'q' to quit.")

    while True:
        frame, landmarks_data = detector.get_frame()

        if frame is None:
            print("Failed to read frame.")
            break

        raw_result      = classifier.classify(landmarks_data)
        hand_is_signing = raw_result["hand_is_signing"]

        # ----- Letters (active in "letters" and "numbers" modes) -----
        if MODE in ("letters", "numbers"):
            if hand_is_signing:
                smoother.reset()
            else:
                smoother.update(raw_result["prediction"])
            stable = smoother.get_stable()
        else:
            smoother.reset()
            stable = None

        # ----- Words with cooldown (active only in "words" mode) -----
        if MODE == "words":
            word_raw = raw_result["word_prediction"]
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
                    locked_word     = stable_word
                    cooldown_frames = WORD_COOLDOWN_FRAMES
        else:
            word_smoother.reset()
            stable_word = None

        # In "words" mode always suppress the raw letter display
        effective_signing = hand_is_signing or (MODE == "words")
        _draw_hud(frame, stable, raw_result, stable_word, effective_signing, MODE)

        cv2.imshow("Sign Language Translator", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    detector.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
