# Sign Language Translator

Real-time American Sign Language (ASL) recognition from a camera. The app
detects hand signs and turns them into on-screen text, speech, and — for
whole-word signs — fluent translated sentences. The goal is to remove
communication barriers between deaf and hearing people without either side
needing to know sign language beforehand.

The project ships as **two independent apps that share the same models and
math**:

- **Desktop (Python)** — `main.py` + OpenCV + MediaPipe, the full-featured
  version: letters, numbers, words, offline text-to-speech, speech-to-text
  (Whisper), and an LLM translation layer (gloss → fluent sentence).
- **Web (`web/`)** — a build-step-free vanilla-JS app that runs entirely in
  the browser (no backend, no video ever leaves the device): the same three
  recognition modes, voice in/out via the Web Speech API, and a two-way
  conversation panel with exportable transcripts. See [web/README.md](web/README.md).
  The one desktop feature not yet ported is the LLM gloss→sentence
  translation step.

It works out of the box: clone, install, run. Pre-trained models are included
for all three modes, so you can try it immediately — and you can also capture
your own signs and retrain to make it yours.

> **Tech:** Python · MediaPipe · TensorFlow/Keras · OpenCV · scikit-learn ·
> vanilla JS (web) · Claude API (translation) · Whisper (speech-to-text)

![Sign Language Translator detecting the letter A in real time](assets/screenshot.png)

*Live detection: MediaPipe draws the 21 hand landmarks, the model predicts the
letter, and a confidence bar confirms it.*

---

## How it works

The pipeline runs per camera frame:

```
Camera frame
  → MediaPipe detects hands (+ shoulders, in words mode) and extracts landmarks
  → normalize_landmarks (aspect-corrected, wrist-centered, scale-invariant)
  → a neural network classifies the sign (dense net for letters/numbers, a TCN for words)
  → temporal smoothing removes single-frame flicker
  → the stable prediction is drawn on screen, spoken aloud, and logged
```

The model learns the **geometry** of the hand (joint coordinates), not its
appearance. That makes it robust to lighting, skin tone, background, and
distance to the camera.

There are three models:

| Model | Input | Detects | Classes |
|---|---|---|---|
| **Letters** (`model_one_hand.h5`) | 63 landmark values (one frame) | Static ASL letters | 24 (A–Y, no J/Z) |
| **Numbers** (`model_numbers.h5`) | 63 landmark values (one frame) | Digits | 10 (0–9) |
| **Words** (`model_words.h5`) | a sequence of 32 frames × 130 body-anchored values | Dynamic signs / whole words | 19 (18 glosses + `nothing`) |

Letters and numbers use separate models (rather than one shared classifier)
because some digits and letters share a near-identical handshape (`2`≈V,
`6`≈W, `9`≈F) and would otherwise collide.

The words model is a **temporal sequence model** (a TCN — stacked 1-D
convolutions over time). It reads the *ordered* movement of the hands, so it
can tell signs apart by *how* the hands move, not just the average pose. Each
frame holds **both hands** (handshape) plus **where each hand is relative to
the shoulders** (a body anchor, via MediaPipe pose). That lets the model
separate signs with the same handshape at different body locations (hand at
the chest vs. the forehead) and signs that genuinely use two hands.

### Two-stage design: recognize, then translate

Recognition outputs **glosses** — English keywords in citation form
(`WANT DRINK NOW`). A second stage sends those to an LLM (Claude) which
produces a fluent sentence with correct grammar and tense (*"Quiero tomar
algo ahora."*). Conjugation and tense live in the translation layer, not the
recognizer: ASL doesn't conjugate verbs, so teaching the recognizer conjugated
forms would be both wrong and explosive. Press **T** in words mode to
translate the signed sentence. Works offline too (falls back to the raw
glosses). This stage exists on desktop only for now — see the web roadmap in
[web/README.md](web/README.md).

---

## Quick start

### Desktop (Python)

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

The pre-trained models are already in the repo, so the app runs right after
installing — no dataset download or training required.

### Keyboard shortcuts

| Key | Action |
|---|---|
| **Q** | Quit the application |
| **L** | Switch to letters mode |
| **W** | Switch to words mode |
| **N** | Switch to numbers mode |
| **E** | Export the conversation log (TXT + CSV to `logs/`) |
| **T** | Translate the signed sentence to fluent text (words mode) |
| **P** | Toggle speech-to-text microphone (if enabled) |

The translation layer uses the Claude API. Set `ANTHROPIC_API_KEY` in your
environment (see [.env.example](.env.example)) to enable it; without a key
the app still runs and **T** just joins the recognized glosses. Tune it in
[config.py](config.py) section 10.

### Web

```bash
cd web
python -m http.server 8000
# open http://localhost:8000
```

No install step — it's static JS. The camera requires `localhost` or HTTPS.
Full architecture, parity rules with the desktop app, and deploy notes
(Netlify, via [netlify.toml](netlify.toml)) are in [web/README.md](web/README.md).

---

## Using the app

The active mode is set in [config.py](config.py) with the `MODE` variable
(default: `"words"`); the web app has an on-screen Letters/Numbers/Words
toggle instead.

| `MODE` | What it does |
|---|---|
| `"letters"` | Static letters (A–Y). Best for finger-spelling. |
| `"words"` | Whole-word signs (the default). Best for fluent signs. |
| `"numbers"` | Digits 0–9. |

On screen you will see the detected letter/word, a confidence bar (green > 80%,
yellow > 60%, red below), the current mode, and — when the model is unsure
(< 70%) — the top-3 candidates so you can see what it is weighing. `config.py`
is heavily commented — every threshold and timing value is documented there.

**What the included models currently recognize:**

- **Letters:** A–Y as static poses, except **J** and **Z** (these need motion
  and are flagged with a "requires motion" hint).
- **Numbers:** 0–9 as static poses.
- **Words:** 18 conversational ASL glosses (`drink, eat, finished, good,
  hello, help, let, me, more, need, no, please, sorry, thanks, that, want,
  yes, you`) plus a negative `nothing` class that keeps the model silent when
  you are not signing. The `nothing` class is essential — without it a
  closed-set classifier labels *everything* as some word and never stays
  quiet. You capture and train whatever vocabulary you want (see below).

---

## Make it your own (capture + train)

The whole project is a **capture → train → use** pipeline. You can replace the
included data with your own signs and retrain any of the three models from
scratch. This is the recommended path if you want the recognition tuned to
*your* hands.

### 1. Capture signs from your webcam

```bash
python capture/capture_letters.py   # static letters
python capture/capture_numbers.py   # static digits
python capture/capture_words.py     # dynamic word signs (recorded as sequences)
```

Edit the `LABELS` / `WORDS` list at the top of each script to choose what to
capture. Letter/number captures save timestamped CSVs; word captures save one
`.npy` sequence per take plus a `manifest.csv` under `data/real_capture/words/`.
If a word capture session's manifest is ever lost or out of sync, rebuild it
from the `.npy` files on disk with `python capture/rebuild_manifest.py`.

All capture formats are **self-describing about camera geometry**: every
sample records the frame's aspect ratio (CSV `aspect` column for
letters/numbers; per-frame value in the word `.npy`). MediaPipe normalizes
landmarks per-axis, so this is what lets models trained on a 16:9 webcam work
on a portrait phone — and it means you can capture from any camera. Word
takes additionally store RAW landmarks (feature building happens at training
time), so future normalization changes never invalidate captured data.

**Capture across multiple sessions.** Re-run the capture script on different
days (different lighting, clothing, distance). The script *appends*, so this
builds a multi-session dataset. A model trained on many takes from a single
session memorizes that session and fails live — varied sessions are what make
it generalize. This is the single most important factor for reliable
recognition, and it's currently the biggest gap in the included letters
dataset (captured in a single session, unlike numbers and words below).

### 2. Train

Every trainer discovers its data automatically — no paths to edit. The word
trainer reads `data/real_capture/words/manifest.csv`; the letter and number
trainers pick up every CSV in their capture directory.

```bash
python training/train_letters.py     # produces model/model_one_hand.h5
python training/train_numbers.py     # produces model/model_numbers.h5
python training/train_words.py       # produces model/model_words.h5 (TCN)
```

Each run appends a record to `runs/<model>_<timestamp>.json` (vocab, sessions,
sample counts, validation accuracy, git SHA) so results are comparable over
time. To check generalization honestly (not just holdout accuracy from the
same session), evaluate with a session-grouped cross-validation:

```bash
python training/evaluate.py --cv 5           # words
python training/evaluate_letters.py --cv 5   # letters (needs 2+ sessions per letter)
python training/evaluate_numbers.py --cv 5   # numbers
```

### 3. Update the web app

After retraining, sync the new model into `web/` (export weights, regenerate
parity fixtures, copy labels):

```bash
python tools/update_web.py letters   # or numbers / words
```

Then open `web/js/utils.test.html` and confirm **ALL TESTS PASS** before
deploying — this is a manual gate by design, see [web/README.md](web/README.md).

---

## Project structure

```
Sign_Language_Translator/
├── main.py                  # Entry point — real-time translator (desktop)
├── config.py                # Single source of truth for all parameters
├── requirements.txt         # Python dependencies
├── .env.example             # ANTHROPIC_API_KEY template
├── netlify.toml             # Deploy config for the web app
│
├── src/
│   ├── detector.py          # Camera + MediaPipe hand/pose landmarks
│   ├── classifier.py        # Loads models and classifies in real time
│   ├── utils.py             # normalize_landmarks + build_word_features + resample_sequence
│   ├── voice.py             # Speech synthesis (Windows SAPI5, offline)
│   ├── overlay.py           # LetterBuffer + WordBuffer + SpeechBuffer (subtitle bars)
│   ├── conversation_log.py  # Exportable conversation log (TXT + CSV)
│   ├── speech_input.py      # Speech-to-text (Whisper, push-to-talk with P key)
│   └── translator.py        # ASL glosses -> fluent sentence (Claude API, T key)
│
├── capture/                 # capture_letters/numbers/words.py + rebuild_manifest.py
├── training/                # train_letters/numbers/words.py + evaluate*.py
├── tools/                   # export_model_json.py, make_fixtures.py, update_web.py
│                             #   (bridge trained models -> web/)
│
├── model/                   # model_*.h5, labels_*.json, MediaPipe .task files
├── data/real_capture/       # letters/ + numbers/ (CSV) + words/ (seq/*.npy + manifest.csv)
├── runs/                    # One JSON per training run (metrics, vocab, git SHA)
├── docs/                    # IDEA.md (pitch) + TECNICO.md (deep technical reference, Spanish)
│
└── web/                     # Standalone browser app — see web/README.md
    ├── index.html, css/
    ├── js/                  # pipeline, model, utils, tts, stt, conversation, ui, config
    ├── model/                # exported weights + labels + .task files
    ├── fixtures/             # parity fixtures (desktop <-> web output must match)
    └── vendor/               # MediaPipe WASM + fonts, vendored (no CDN)
```

---

## What is and isn't in the repo

**Included (small, so the app runs immediately):**

- Trained models: `model/*.h5` and the MediaPipe `.task` files
- Label maps: `model/labels_*.json`
- Captured datasets: `data/real_capture/{letters,numbers,words}`
- The exported web build: `web/model/`, `web/fixtures/`

**Not included (see [.gitignore](.gitignore)):**

- The Python virtual environment (`venv/`) — recreate it with the steps above.
- Large raw image datasets — the models are trained directly from captured
  landmark CSVs/sequences, so no bulky image dataset is needed.

Everything required to run **and** to retrain is in the repository.

---

## Current status

✅ **Implemented (desktop):**
- Real-time hand + pose detection (MediaPipe)
- Letter recognition (A–Y static, 24 classes)
- Number recognition (0–9, 99% val accuracy)
- Word / dynamic-sign recognition (18 glosses + `nothing`, 98% val accuracy)
- Gloss → fluent sentence translation (Claude API, offline fallback)
- Confidence bar + top-3 alternatives when the model is unsure (< 70%)
- Speech synthesis (Windows SAPI5, offline) and speech-to-text (Whisper, push-to-talk)
- Letter / word accumulation with subtitle overlay
- Exportable conversation log (E key → TXT + CSV)
- Full capture + training + evaluation pipeline

✅ **Implemented (web, see [web/README.md](web/README.md)):**
- Same three modes (Letters / Numbers / Words), same models, running 100% client-side
- Voice output (Web Speech synthesis) and dictation (Web Speech recognition)
- Two-way conversation panel, REC + TXT/CSV export (byte-compatible with desktop logs)
- Netlify deploy config

🔜 **Next:**
- Dynamic J / Z letters (require motion, not yet recognized as static poses)
- Gloss → sentence translation on the web (currently desktop-only)
- Expand the word vocabulary further
- A second capture session for letters, to make cross-session validation possible

---

## Notes & limitations

- **J and Z** require motion and are not recognized as static poses yet; the
  UI flags them.
- **Letters were captured in a single session** — unlike numbers (4 sessions)
  and words (11 sessions), so leak-free cross-session validation isn't
  meaningful yet for that model. Capturing a second session is the fix.
- Recognition quality depends on lighting and camera position. Capturing your
  own samples in your usual setup improves accuracy.
- Offline text-to-speech (SAPI5) is Windows-only on desktop; the web app uses
  the browser's Web Speech API instead and works cross-platform.

---

## License

Educational project. Feel free to explore, fork, and build on it.
