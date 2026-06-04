# Sign Language Translator

Real-time American Sign Language (ASL) recognition from your webcam. The app
detects hand signs, turns them into on-screen text, and is built to grow into
a full sign-to-text-and-voice translator. The goal is to remove communication
barriers between deaf and hearing people without either side needing to know
sign language beforehand.

It works out of the box: clone, install, run. Pre-trained models are included,
so you can try it immediately — and you can also capture your own signs and
retrain it to make it yours.

> **Tech:** Python · MediaPipe · TensorFlow/Keras · OpenCV · scikit-learn

![Sign Language Translator detecting the letter A in real time](assets/screenshot.png)

*Live detection: MediaPipe draws the 21 hand landmarks, the model predicts the
letter, and a confidence bar confirms it.*

---

## How it works

The pipeline runs per camera frame:

```
Camera frame
  → MediaPipe detects the hand and extracts 21 landmarks (x, y, z) = 63 numbers
  → normalize_landmarks (wrist-centered + scale-invariant)
  → a small dense neural network classifies the sign
  → temporal smoothing removes single-frame flicker
  → the stable prediction is drawn on screen
```

The model learns the **geometry** of the hand (joint coordinates), not its
appearance. That makes it robust to lighting, skin tone, background, and
distance to the camera.

There are two models:

| Model | Input | Detects |
|---|---|---|
| **Letters** (`model_one_hand.h5`) | 63 landmark values (one frame) | Static ASL letters |
| **Words** (`model_words.h5`) | 126 values (mean + std over 20 frames) | Dynamic signs / whole words |

The words model uses 20 frames of motion, so it can recognize signs that
require movement, which a single static frame cannot capture.

---

## Quick start

You need **Python 3.10+** and a webcam.

```bash
# 1. Clone
git clone <your-repo-url>
cd Sign_Language_Translator

# 2. Create and activate a virtual environment
python -m venv venv
venv\Scripts\activate          # Windows
# source venv/bin/activate     # Linux / macOS

# 3. Install dependencies
pip install -r requirements.txt

# 4. Run
python main.py
```

### Keyboard shortcuts

| Key | Action |
|---|---|
| **Q** | Quit the application |
| **L** | Switch to letters mode |
| **W** | Switch to words mode |
| **N** | Switch to numbers mode |
| **T** | Force past tense (conjugation) |
| **R** | Force present tense (conjugation, default) |
| **F** | Force future tense (conjugation) |
| **P** | Toggle speech-to-text microphone (if enabled) |

The pre-trained models are already in the repo, so the app runs right after
installing — no dataset download or training required.

---

## Using the app

The active mode is set in [config.py](config.py) with the `MODE` variable:

| `MODE` | What it does |
|---|---|
| `"letters"` | Shows static letters (A–Y). Best for finger-spelling. |
| `"words"` | Shows whole-word signs (the default). Best for fluent signs. |
| `"numbers"` | Reserved — behaves like `"letters"` until a numbers model exists. |

On screen you will see the detected letter/word, a confidence bar (green > 80%,
yellow > 60%, red below), and the current mode. `config.py` is heavily
commented — every threshold and timing value is documented there.

**What the included models currently recognize:**

- **Letters:** A–Y as static poses, except **J** and **Z** (these need motion
  and are flagged with a "requires motion" hint).
- **Words:** `hola`, `adios`, `Yo` (plus a negative "nothing" class). The word
  labels are in Spanish because they are the signs the team captured — you can
  capture and train any vocabulary you want (see below).

---

## Make it your own (capture + train)

The whole project is a **capture → train → use** pipeline. You can replace the
included data with your own signs and retrain both models from scratch. This
is the recommended path if you want the recognition tuned to *your* hands.

### 1. Capture signs from your webcam

```bash
# Letters (static poses)
python capture/capture_letters.py

# Words (dynamic signs, recorded as short sequences)
python capture/capture_words.py
```

Edit the `LABELS` / `WORDS` list at the top of each script to choose what to
capture. Each session saves a timestamped CSV under `data/real_capture/`.

### 2. Point the training scripts at your CSVs

Add the generated CSV paths to the `DATA_CSVS` list in
[training/train_letters.py](training/train_letters.py) or
[training/train_words.py](training/train_words.py).

### 3. Train

```bash
python training/train_letters.py     # produces model/model_one_hand.h5
python training/train_words.py       # produces model/model_words.h5
```

### 4. (Optional) Evaluate

```bash
python training/evaluate.py
```

This prints per-class accuracy, top confusions, and a confidence-threshold
sweep, and saves confusion-matrix PNGs to `model/`.

---

## Project structure

```
Sign_Language_Translator/
├── main.py                  # Entry point — real-time translator with tense selector
├── config.py                # Single source of truth for all parameters (includes verb conjugations)
├── requirements.txt         # Dependencies
│
├── src/
│   ├── detector.py          # Camera + MediaPipe hand landmarks (21 per hand)
│   ├── classifier.py        # Loads models and classifies in real time
│   ├── utils.py             # normalize_landmarks + PredictionSmoother
│   ├── voice.py             # Speech synthesis (Windows SAPI5, offline)
│   ├── overlay.py           # LetterBuffer (accumulation + conjugation) + SpeechBuffer (transcripts)
│   ├── grammar.py           # Verb conjugation system (deterministic, no ML)
│   └── speech_input.py      # Speech-to-text (Whisper, push-to-talk with P key)
│
├── capture/
│   ├── capture_letters.py   # Capture static letter samples
│   └── capture_words.py     # Capture dynamic word sequences
│
├── training/
│   ├── train_letters.py     # Train the letter model
│   ├── train_words.py       # Train the word model
│   └── evaluate.py          # Metrics + confusion matrices
│
├── scripts/
│   └── export_grammar.py    # Export conjugation tables to JSON (for web)
│
├── model/
│   ├── model_one_hand.h5    # Letter classification model
│   ├── model_words.h5       # Word/dynamic-sign classification model
│   ├── labels_one_hand.json # Letter labels (A–Y)
│   ├── labels_words.json    # Word labels (hola, adios, Yo, nada)
│   ├── hand_landmarker.task # MediaPipe hand detection model
│   └── grammar.json         # Conjugation tables (auto-generated, for web)
│
├── data/real_capture/       # Sample captured CSVs (landmarks)
│   ├── letters/
│   └── words/
│
├── ROADMAP.md               # Detailed roadmap + architecture decisions + scalability analysis
├── HOW_TO_ADD_VERBS.md      # Guide: add new Spanish verbs with all 3 tenses
├── GRAMMAR_WEB_PLAN.md      # Grammar system + web migration plan
├── GRAMMAR_SHORT.md         # 1-page summary of conjugation
└── CLAUDE.md                # Internal project context (Spanish)
```

---

## What is and isn't in the repo

**Included (small, so the app runs immediately):**

- Trained models: `model/*.h5` and the MediaPipe `hand_landmarker.task`
- Label maps: `model/labels_*.json`
- Sample captured datasets: `data/real_capture/letters` and `.../words`

**Not included (see [.gitignore](.gitignore)):**

- The Python virtual environment (`venv/`) — recreate it with the steps above.
- Large raw image datasets (`data/asl_alphabet/`, `data/processed/`) — the
  current models are trained directly from the captured landmark CSVs, so no
  bulky image dataset is needed.

Everything required to run **and** to retrain is in the repository.

---

## Documentation

**For roadmap, architecture, and future improvements:**

- **[ROADMAP.md](ROADMAP.md)** — Detailed roadmap with phases, architecture decisions, and analysis of future scalability solutions (how to handle 500+ verbs efficiently, when to migrate to web, etc.)

**For verb conjugation and tenses:**

- **[HOW_TO_ADD_VERBS.md](HOW_TO_ADD_VERBS.md)** — Step-by-step guide to add new Spanish verbs with all 3 tenses (present, past, future)
- **[GRAMMAR_WEB_PLAN.md](GRAMMAR_WEB_PLAN.md)** — Detailed grammar architecture, web migration plan, and improvement roadmap
- **[GRAMMAR_SHORT.md](GRAMMAR_SHORT.md)** — Quick 1-page summary of conjugation system

**For system design:**

- **[CLAUDE.md](CLAUDE.md)** — Internal project context (Spanish, for team use)

---

## Current Status (MVP)

✅ **Implemented:**
- Real-time hand detection (MediaPipe)
- Letter recognition (A–Y static)
- Word / dynamic-sign recognition (4 words)
- Verb conjugation (8 verbs, 3 tenses: present, past, future)
- Speech synthesis (Windows SAPI5, offline)
- Speech-to-text (Whisper, push-to-talk with P key)
- Letter accumulation + text overlay with auto-conjugation
- Tense auto-detection (ayer → past, mañana → future)
- Capture + training pipeline

🔜 **Short-term (Phase 2):**
- Expand vocabulary (50+ verbs)
- Tiempos compuestos (present perfect: "he creído")
- Web MVP (React + tensorflowjs)

💡 **Future (Phase 3+):**
- Refactor to pattern-based conjugation (scalable to 500+ verbs)
- Multi-language support (English, ASL British)
- Mobile app (iOS / Android)
- Subjuntivo, condicional
- Full web platform with API

See **[ROADMAP.md](ROADMAP.md)** for detailed phases and decision rationale. |

---

## Notes & limitations

- **J and Z** require motion and are not recognized as static poses yet; the UI
  flags them.
- **Word vocabulary is small** (a few signs). More captured samples per word
  make it noticeably more reliable.
- Recognition quality depends on lighting and camera position. Capturing your
  own samples in your usual setup improves accuracy.

---

## License

Educational project. Feel free to explore, fork, and build on it.
