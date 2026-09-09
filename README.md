# Sign Language Translator

Real-time American Sign Language (ASL) recognition from a camera. The app
detects hand signs, turns them into on-screen text and speech, and lets the
hearing side answer back — a two-way conversation tool for people who don't
share a language. The goal is to remove the communication barrier between deaf
and hearing people without either side needing to know sign language
beforehand.

It comes in two flavours, both running **100% locally** — no server, no
account, and no video ever leaves the device:

- **Web app** (`web/`) — one page, no install, works on desktop and phone.
- **Desktop app** (`main.py`) — Python/OpenCV, plus the capture and training
  pipeline used to build the models.

Pre-trained models are included, so you can clone, install, and run. You can
also capture your own signs and retrain everything to make it yours.

> **Tech:** Python · MediaPipe · TensorFlow/Keras · OpenCV · scikit-learn ·
> vanilla JavaScript (web)

![Sign Language Translator detecting the letter A in real time](assets/screenshot.png)

*Live detection: MediaPipe draws the 21 hand landmarks, the model predicts the
letter, and a confidence bar confirms it.*

---

## How it works

The pipeline runs per camera frame:

```
Camera frame
  → MediaPipe detects the hand and extracts 21 landmarks (x, y, z) = 63 numbers
  → normalize_landmarks (aspect-corrected + wrist-centered + scale-invariant)
  → a neural network classifies the sign
  → temporal smoothing removes single-frame flicker
  → the stable prediction is drawn on screen, spoken, and logged
```

The models learn the **geometry** of the hand (joint coordinates), not its
appearance. That makes them robust to lighting, skin tone, background, and
distance to the camera.

### Three models

| Model | Input | Recognizes | Trained on |
|---|---|---|---|
| **Letters** (`model_one_hand.h5`) | 63 landmark values (one frame) | 24 static letters, **A–Y without J/Z** | 3 750 samples · 9 sessions |
| **Numbers** (`model_numbers.h5`) | 63 landmark values (one frame) | Digits **0–9** | 1 500 samples · 4 sessions |
| **Words** (`model_words.h5`) | a sequence of 32 frames × 130 body-anchored values | 18 dynamic word signs + a negative class | 668 takes · 11 sessions |

Numbers are a **separate model on purpose**: several ASL digits collide with
letters (2=V, 6=W, 9=F), so one closed set would have to choose between them.

The words model is a **temporal sequence model** (a TCN — stacked 1-D
convolutions over time). It reads the *ordered* movement of the hands, so it can
tell signs apart by *how* the hands move, not just the average pose. This is the
key to scaling the vocabulary: a movement summary that ignores order collapses
once two signs share a similar average shape.

Each word frame holds **both hands** (handshape) plus **where each hand is
relative to the shoulders** (a body anchor, via MediaPipe pose). That lets the
model separate signs with the same handshape at different body locations (hand
at the chest vs. the forehead) and signs that genuinely use two hands. Letters
and numbers stay one-hand and are unaffected.

The current word vocabulary is: *drink, eat, finished, good, hello, help, let,
me, more, need, no, please, sorry, thanks, that, want, yes, you* — plus a
negative `nothing` class that is never displayed. That class is essential:
without it a closed-set classifier labels *everything* as some word and never
stays quiet.

### Two-stage design: recognize, then translate

Recognition outputs **glosses** — English keywords in citation form
(`WANT DRINK NOW`). A second stage sends those to an LLM (Claude) which produces
a fluent sentence with correct grammar and tense (*"Quiero tomar algo ahora."*).
Conjugation and tense live in the translation layer, not the recognizer: ASL
doesn't conjugate verbs, so teaching the recognizer conjugated forms would be
both wrong and explosive. Press **T** in words mode to translate the signed
sentence. It works offline too (falls back to the raw glosses), and it is
currently a **desktop-only** feature.

---

## Quick start — web app

No install, no build step. Serve the folder and open it:

```bash
cd web
python -m http.server 8000
# open http://localhost:8000
```

The camera works on `http://localhost` (secure-context exception); any other
host needs HTTPS. `netlify.toml` at the repo root is preconfigured, so
connecting the repo in Netlify deploys `web/` on every push.

**What the web app does**

- **Letters / Numbers / Words** behind a single toggle — the same three modes as
  the desktop app.
- **Speaks recognized signs out loud** (Web Speech), with a voice picker and an
  on/off toggle; letters mode can speak whole words or every letter.
- **Two-way conversation panel**: the hearing side types or dictates, and their
  message pops up large over the camera so the signer reads it without leaving
  the frame. Everything from both sides lands in one chat log.
- **REC + Export**: keeps a transcript and downloads it as TXT + CSV, in the
  exact same format the desktop app writes to `logs/`.
- Confidence bar, top-3 candidates when the model is unsure, camera picker,
  help and settings dialogs, and a `?debug` overlay with per-stage timings.

Details, architecture and the parity rules live in
[web/README.md](web/README.md). **`web/js/utils.test.html` is the test gate** —
it replays fixtures generated from real capture data and must be all green
before deploying.

---

## Quick start — desktop app

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
| **N** | Switch to numbers mode |
| **W** | Switch to words mode |
| **E** | Export the conversation log (TXT + CSV to `logs/`) |
| **T** | Translate the signed sentence to fluent text (words mode) |
| **P** | Toggle the speech-to-text microphone |

The translation layer uses the Claude API. Set `ANTHROPIC_API_KEY` in your
environment to enable it; without a key the app still runs and **T** just joins
the recognized glosses. Tune it in [config.py](config.py) section 10.

The starting mode is `MODE` in [config.py](config.py) (`"letters"`, `"numbers"`
or `"words"`); L/N/W switch it live. On screen you get the detected sign, a
confidence bar (green > 80%, yellow > 60%, red below), the accumulated subtitle
bar, and — when the model is unsure (< 70%) — the top-3 candidates.
`config.py` is heavily commented: every threshold and timing value is
documented there and is the single source of truth for the whole project.

---

## Make it your own (capture → train → use)

The project is a full **capture → train → use** pipeline. There is no external
dataset: every model is trained on landmarks captured with a webcam, so you can
replace the included data with your own signs and retrain from scratch. This is
the recommended path if you want recognition tuned to *your* hands.

### 1. Capture from your webcam

```bash
python capture/capture_letters.py    # static letter poses  -> CSV
python capture/capture_numbers.py    # static digit poses   -> CSV
python capture/capture_words.py      # dynamic signs        -> .npy + manifest
```

Edit the `LABELS` / `WORDS` list at the top of each script to choose what to
capture. Letters and numbers save timestamped CSVs; word captures save one
`.npy` sequence per take plus a `manifest.csv` under `data/real_capture/words/`.

All capture formats are **self-describing about camera geometry**: every sample
records the frame's aspect ratio (an `aspect` column for letters/numbers, a
per-frame value in the word `.npy`). MediaPipe normalizes landmarks per axis, so
this is what lets a model trained on a 16:9 webcam work on a portrait phone —
and it means you can capture from any camera. Word takes additionally store
**raw** landmarks, with feature building deferred to training time, so future
changes to normalization or sequence length never invalidate captured data.

**Capture across multiple sessions.** Re-run the scripts on different days
(different lighting, clothing, distance) — they *append*. A model trained on
many takes from a single session memorizes that session and fails live; varied
sessions are what make it generalize. This is the single most important factor
for reliable recognition.

### 2. Train

Every trainer discovers its own data — no paths to edit. The word trainer reads
`data/real_capture/words/manifest.csv`; the letter and number trainers pick up
every CSV in their capture directory.

```bash
python training/train_letters.py     # -> model/model_one_hand.h5
python training/train_numbers.py     # -> model/model_numbers.h5
python training/train_words.py       # -> model/model_words.h5 (TCN)
```

Each run writes a record to `runs/<model>_<timestamp>.json` — vocabulary,
sample counts, session count, hyperparameters, git SHA and validation accuracy —
so you can always tell which data produced which model.

### 3. Evaluate

Validation accuracy from a training run is **optimistic** when train and
validation takes come from the same session. The evaluators offer a leak-free
alternative: k-fold cross-validation **grouped by session**, so no session is
ever split across train and test.

```bash
python training/evaluate.py --cv 5           # words   (groups by session_id)
python training/evaluate_letters.py --cv 5   # letters (groups by CSV file)
python training/evaluate_numbers.py --cv 5   # numbers (groups by CSV file)
```

Without `--cv` they run a fast holdout. All three print a per-class report and a
confusion matrix, and reuse the trainers' own `train_model(...)`, so evaluation
can never drift from training.

### 4. Push the new model to the web app

One command per mode exports the weights, regenerates the parity fixtures and
copies the labels:

```bash
python tools/update_web.py letters
python tools/update_web.py numbers
python tools/update_web.py words
```

Then open `web/js/utils.test.html` and confirm **all tests pass** before
deploying. Skipping this leaves the deployed site silently serving the old
model.

---

## Project structure

```
Sign_Language_Translator/
├── main.py                  # Entry point — real-time desktop translator
├── config.py                # Single source of truth for all parameters
├── requirements.txt         # Dependencies
│
├── src/
│   ├── detector.py          # Camera + MediaPipe hand/pose landmarks
│   ├── classifier.py        # Loads the models and classifies in real time
│   ├── utils.py             # normalize_landmarks + word features + smoothing
│   ├── voice.py             # Speech synthesis (Windows SAPI5, offline)
│   ├── overlay.py           # LetterBuffer + WordBuffer + SpeechBuffer (subtitles)
│   ├── conversation_log.py  # Exportable conversation log (TXT + CSV)
│   ├── speech_input.py      # Speech-to-text (Whisper, push-to-talk with P)
│   └── translator.py        # ASL glosses -> fluent sentence (Claude API, T key)
│
├── capture/
│   ├── capture_letters.py   # Static letter samples (CSV)
│   ├── capture_numbers.py   # Static digit samples (CSV)
│   └── capture_words.py     # Dynamic word sequences (.npy + manifest)
│
├── training/
│   ├── train_letters.py     # Letter model
│   ├── train_numbers.py     # Number model
│   ├── train_words.py       # Word model (temporal TCN)
│   ├── run_log.py           # Writes runs/<model>_<timestamp>.json
│   ├── eval_common.py       # Shared report + session-grouped CV
│   └── evaluate*.py         # One evaluator per model
│
├── tools/                   # Web export: weights, fixtures, labels
├── web/                     # Browser app (see web/README.md)
├── docs/                    # IDEA.md (vision) · TECNICO.md · DESIGN_BRIEF.md
├── model/                   # Trained models, labels, MediaPipe .task files
├── runs/                    # One JSON record per training run
└── data/real_capture/       # letters/ + numbers/ (CSV) + words/ (.npy + manifest)
```

---

## What is and isn't in the repo

**Included (small, so the app runs immediately):**

- Trained models (`model/*.h5`) and the MediaPipe `.task` detectors
- Label maps (`model/labels_*.json`)
- The captured datasets under `data/real_capture/`
- The web app with its own copy of the models and a vendored MediaPipe runtime

**Not included (see [.gitignore](.gitignore)):**

- The Python virtual environment (`venv/`) — recreate it with the steps above
- Exported conversation logs (`logs/`) — per-session user data, not source
- Secrets: `.env` is ignored; `.env.example` is the committable template

Everything required to run **and** to retrain is in the repository.

---

## Current status

✅ **Working:**

- Real-time hand and body detection (MediaPipe)
- Letters A–Y (no J/Z), digits 0–9, and 18 dynamic word signs
- Confidence bar + top-3 alternatives when the model is unsure (< 70%)
- Speech synthesis — desktop (SAPI5, offline) and web (Web Speech)
- Speech-to-text — desktop (Whisper, push-to-talk) and web (dictation)
- Accumulated subtitles and a two-way conversation log, exportable to TXT + CSV
- Gloss → fluent sentence translation via the Claude API (desktop, **T**)
- Capture, training and session-grouped evaluation pipelines for all 3 models
- Web app with all three modes, a conversation panel, and a parity test gate

🔜 **Next:**

- Gloss → sentence translation in the web app
- Dynamic J / Z letters (they need motion, so they belong to the word model)
- A larger word vocabulary
- Adding words from inside the app, without external scripts
- Correction feedback to improve the model from real use

---

## Notes & limitations

- **J and Z** require motion and are not recognized as static poses; the UI
  flags them.
- **The word vocabulary is small** (18 signs). More captured samples across more
  sessions is what makes it noticeably more reliable.
- **Validation accuracy is not live accuracy.** Use the session-grouped `--cv`
  evaluators for an honest number, and test with a person who wasn't recorded.
- Recognition quality depends on lighting and camera position. In words mode the
  **shoulders must be visible** — the signs are anchored to the body.
- On phones the web app runs at roughly 15–24 fps in words mode; that is the
  practical MediaPipe ceiling on mobile, not a bug.

---

## Team

Dilan Calvo · Adrián Durán · Nazareth Solís — COTEPECOS, Web Development, 2026.

## License

Educational project. Feel free to explore, fork, and build on it.
