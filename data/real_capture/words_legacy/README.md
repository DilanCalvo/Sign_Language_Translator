# Legacy word captures (archived 2026-07-12)

These .npy files store ALREADY-PROCESSED (32, 130) feature vectors built
BEFORE the aspect-ratio correction landed in `build_word_features`
(see `LEGACY_CAPTURE_ASPECT` in config.py and docs/TECNICO.md).

They are geometrically inconsistent with the corrected pipeline and CANNOT
be retro-corrected (the raw landmarks were never stored — that lesson is why
the new capture format in capture/capture_words.py stores raw (n, 131) frames).

Do NOT train on these. They are kept for reference only, together with the
model they belong to (`model/legacy/model_words.h5` + labels). Recapture the
vocabulary with `python capture/capture_words.py`, which writes the new
self-describing raw format into `data/real_capture/words/`.
