"""
Generate parity fixtures for the web port (letters, numbers, or words).

The JS reimplementation of the feature math and the exported model must produce
EXACTLY what Python produces — a mismatch is a silent bug (the model still runs,
it just predicts garbage). These fixtures turn that silent failure into a loud
one: web/js/utils.test.html replays them in the browser and shows PASS/FAIL.

Two families share this generator:
  - letters/numbers (static): raw 63 -> normalize -> softmax. Inputs are real
    capture CSV rows; outputs normalize_fixtures.json + model_fixtures.json.
  - words (sequence): raw (n,131) takes -> build_word_features per frame ->
    resample to WORD_SEQ_LEN -> TCN softmax. Inputs are real .npy takes; outputs
    feature_fixtures.json + resample_fixtures.json + model_fixtures.json.

Usage (project venv, from the repo root):
    venv/Scripts/python tools/make_fixtures.py [letters|numbers|words]
"""

import csv
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from config import LEGACY_CAPTURE_ASPECT, WORD_SEQ_LEN  # noqa: E402
from src.utils import (  # noqa: E402
    normalize_landmarks,
    build_word_features,
    word_features_from_raw,
    resample_sequence,
    WORD_RAW_FRAME_LEN,
    WORD_FEATURE_LEN,
)

# Per-mode capture source, trained model, and web output dir. Keep in sync with
# tools/export_model_json.py and tools/update_web.py.
MODES = {
    "letters": {
        "csv_dir": REPO / "data" / "real_capture" / "letters",
        "model":   REPO / "model" / "model_one_hand.h5",
        "out_dir": REPO / "web" / "fixtures",
    },
    "numbers": {
        "csv_dir": REPO / "data" / "real_capture" / "numbers",
        "model":   REPO / "model" / "model_numbers.h5",
        "out_dir": REPO / "web" / "fixtures" / "numbers",
    },
    "words": {
        "data_dir": REPO / "data" / "real_capture" / "words",
        "model":    REPO / "model" / "model_words.h5",
        "out_dir":  REPO / "web" / "fixtures" / "words",
    },
}

SEED = 42


def collect_real_rows(csv_dir):
    """One (label, raw63, aspect) per CLASS, sampled across ALL capture CSVs.

    Per-class (not per-file) coverage on purpose: every class present in the
    capture data gets a parity case, so a bug that only shows up in some part
    of the class range cannot slip through the gate. Deterministic (fixed
    seed) so reruns produce identical fixtures for identical data.

    The aspect comes from the CSV's own "aspect" column when present
    (self-describing captures); legacy files fall back to
    config.LEGACY_CAPTURE_ASPECT — the same rule the training loader applies.
    """
    rng = np.random.default_rng(SEED)
    by_label = {}
    for path in sorted(csv_dir.glob("*.csv")):
        with open(path, encoding="utf-8") as f:
            rows = list(csv.reader(f))
        header, rows = rows[0], rows[1:]
        has_aspect = "aspect" in header
        for row in rows:
            if row:
                aspect = float(row[64]) if has_aspect else LEGACY_CAPTURE_ASPECT
                by_label.setdefault(row[0], []).append((row, aspect))
    cases = []
    for label in sorted(by_label):
        row, aspect = by_label[label][int(rng.integers(len(by_label[label])))]
        cases.append((label, [float(v) for v in row[1:64]], aspect))
    return cases


def edge_cases():
    """Synthetic inputs that exercise the scale guard (scale < 1e-6 -> 1.0).
    Aspect 1.0 = square frame, a no-op stretch that still walks the aspect path.
    """
    zeros = [0.0] * 63
    # All 21 points identical: centering yields zeros, scale is 0 -> guard path.
    collapsed = [0.4, 0.5, 0.1] * 21
    return [("edge_all_zeros", zeros, 1.0), ("edge_collapsed_hand", collapsed, 1.0)]


def invariance_cases(real, n=3):
    """THE regression gate for the aspect-ratio bug.

    Property: one physical hand pose, rendered by cameras of different aspect
    ratios, must normalize to the SAME vector. Built from real rows: undo the
    row's own stretch to recover geometry-true points, re-stretch them as a
    16:9 camera and as a 9:16 portrait phone would report them, and expect
    both to normalize to what the original row normalizes to. If someone ever
    breaks the un-stretch (in Python or in the JS port), these cases fail.
    """
    cases = []
    for label, raw, aspect in real[:n]:
        expected = normalize_landmarks(raw, aspect=aspect)
        true_pts = np.asarray(raw, dtype=np.float32).reshape(21, 3).copy()
        true_pts[:, 1] /= np.float32(aspect)  # geometry-true (width units)
        for cam_aspect in (16 / 9, 9 / 16):
            stretched = true_pts.copy()
            stretched[:, 1] *= np.float32(cam_aspect)
            case_input = [float(v) for v in stretched.flatten()]
            # Server-side assert: the gate must hold in Python before it is
            # ever handed to the browser test page.
            got = normalize_landmarks(case_input, aspect=cam_aspect)
            assert np.allclose(got, expected, atol=1e-5), (
                f"aspect invariance broken for {label} at {cam_aspect:.2f}")
            cases.append((
                f"invariance_{label}_{'16x9' if cam_aspect > 1 else '9x16'}",
                case_input, cam_aspect, expected,
            ))
    return cases


# ---------------------------------------------------------------------------
# Words (sequence) fixtures. Structurally different from the static path: real
# inputs are raw .npy takes (n, 131), and the JS must reproduce build_word_features
# (per frame), resample_sequence (variable length -> WORD_SEQ_LEN) and the TCN
# forward pass. Three fixture files isolate each stage so a failure localizes.
# ---------------------------------------------------------------------------

# y-coordinate positions inside one (131,) raw word frame: every landmark's y in
# both hands, plus each shoulder's y. Used to re-stretch a frame to another
# aspect for the invariance gate (x/z and the aspect slot are left untouched).
_WORD_Y_INDICES = ([3 * i + 1 for i in range(21)]           # left hand ys
                   + [63 + 3 * i + 1 for i in range(21)]    # right hand ys
                   + [127, 129])                            # shoulder_l/r ys


def collect_word_takes(data_dir):
    """One (gloss, raw_take) per class, sampled deterministically across the
    manifest. Each take is a real (n_frames, WORD_RAW_FRAME_LEN) array."""
    manifest = data_dir / "manifest.csv"
    seq_dir = data_dir / "seq"
    if not manifest.is_file():
        sys.exit(f"[ERROR] No manifest at {manifest}")
    rng = np.random.default_rng(SEED)
    by_gloss = {}
    with open(manifest, encoding="utf-8") as f:
        for r in csv.DictReader(f):
            by_gloss.setdefault(r["gloss"], []).append(r["sample_id"])
    takes = []
    for gloss in sorted(by_gloss):
        ids = by_gloss[gloss]
        sample_id = ids[int(rng.integers(len(ids)))]
        path = seq_dir / f"{sample_id}.npy"
        if not path.is_file():
            continue
        take = np.load(path).astype(np.float32)
        if take.ndim != 2 or take.shape[1] != WORD_RAW_FRAME_LEN or len(take) < 2:
            continue
        takes.append((gloss, take))
    return takes


def _restretch_word_frame(frame, new_aspect):
    """Same physical frame as if a camera of `new_aspect` had reported it:
    undo the frame's own y-stretch and re-apply the new one. Zeros (absent
    hand/shoulders) stay zero, so 'absent' is preserved."""
    frame = np.asarray(frame, dtype=np.float32).copy()
    factor = np.float32(new_aspect) / np.float32(frame[130])
    for j in _WORD_Y_INDICES:
        frame[j] *= factor
    frame[130] = new_aspect
    return frame


def build_word_fixtures(data_dir, model_path, out_dir):
    takes = collect_word_takes(data_dir)
    if not takes:
        sys.exit(f"[ERROR] No usable word takes under {data_dir}")

    # --- 1. Feature fixtures: one representative frame per gloss. The middle
    # frame is mid-sign, so hands are present and the body-anchor path runs. ---
    feature_fixtures, invariance = [], []
    for gloss, take in takes:
        frame = take[len(take) // 2]
        expected = word_features_from_raw(frame)
        feature_fixtures.append({
            "label": gloss,
            "input": [float(v) for v in frame],
            "expected": [float(v) for v in expected],
        })

    # --- 2. Aspect invariance: the same frame at 16:9 and 9:16 must featurize
    # identically (the word-side of the aspect gate). Asserted in Python first. ---
    for gloss, take in takes[:3]:
        frame = take[len(take) // 2]
        expected = word_features_from_raw(frame)
        for cam_aspect in (16 / 9, 9 / 16):
            stretched = _restretch_word_frame(frame, cam_aspect)
            got = word_features_from_raw(stretched)
            assert np.allclose(got, expected, atol=1e-5), (
                f"word aspect invariance broken for {gloss} at {cam_aspect:.2f}")
            invariance.append({
                "label": f"invariance_{gloss}_{'16x9' if cam_aspect > 1 else '9x16'}",
                "input": [float(v) for v in stretched],
                "expected": [float(v) for v in expected],
            })

    # --- 3. Resample index math: variable length -> WORD_SEQ_LEN. The JS must
    # pick the SAME frame indices Python does (pure index-picking). ---
    resample_fixtures = []
    for n in sorted({len(t) for _, t in takes} | {WORD_SEQ_LEN, 16, 45}):
        idx = np.linspace(0, n - 1, WORD_SEQ_LEN).round().astype(int)
        resample_fixtures.append({"n": int(n), "indices": [int(i) for i in idx]})

    out_dir.mkdir(parents=True, exist_ok=True)
    for name, data in (("feature_fixtures", feature_fixtures + invariance),
                       ("resample_fixtures", resample_fixtures)):
        with open(out_dir / f"{name}.json", "w", encoding="utf-8") as f:
            json.dump(data, f)
    print(f"[words] feature fixtures  -> {out_dir / 'feature_fixtures.json'}  "
          f"({len(feature_fixtures)} + {len(invariance)} invariance)")
    print(f"[words] resample fixtures -> {out_dir / 'resample_fixtures.json'}  "
          f"({len(resample_fixtures)} cases)")

    # --- 4. Model fixtures: the FULL live chain per gloss. Store the raw take;
    # the JS resamples -> featurizes -> runs the TCN and must match this softmax.
    # This is the gate that catches a wrong causal/dilated conv. ---
    import tensorflow as tf
    model = tf.keras.models.load_model(model_path, compile=False)
    model_fixtures = []
    for gloss, take in takes:
        raw32 = resample_sequence(take, WORD_SEQ_LEN)               # (T, 131)
        seq = np.stack([word_features_from_raw(f) for f in raw32])  # (T, 130)
        probs = model(tf.constant(seq[np.newaxis], dtype=tf.float32),
                      training=False).numpy()[0]
        model_fixtures.append({
            "label": gloss,
            "raw_take": [[float(v) for v in row] for row in take],
            "expected_probs": [float(p) for p in probs],
        })
    with open(out_dir / "model_fixtures.json", "w", encoding="utf-8") as f:
        json.dump(model_fixtures, f)
    print(f"[words] model fixtures    -> {out_dir / 'model_fixtures.json'}  "
          f"({len(model_fixtures)} cases, full resample->featurize->TCN chain)")


def main():
    mode = sys.argv[1] if len(sys.argv) > 1 else "letters"
    if mode not in MODES:
        sys.exit(f"[ERROR] Unknown mode '{mode}'. Use one of: {', '.join(MODES)}")

    if mode == "words":
        cfg = MODES["words"]
        if not cfg["data_dir"].is_dir():
            sys.exit(f"[ERROR] No word capture dir at {cfg['data_dir']}")
        build_word_fixtures(cfg["data_dir"], cfg["model"], cfg["out_dir"])
        return

    csv_dir = MODES[mode]["csv_dir"]
    model_path = MODES[mode]["model"]
    out_dir = MODES[mode]["out_dir"]

    if not csv_dir.is_dir():
        sys.exit(f"[ERROR] No capture CSVs at {csv_dir}")

    real = collect_real_rows(csv_dir)
    if not real:
        sys.exit(f"[ERROR] No rows found in {csv_dir}")

    norm_fixtures = []
    for label, raw, aspect in real + edge_cases():
        normalized = normalize_landmarks(raw, aspect=aspect)
        norm_fixtures.append({
            "label": label,
            "input": raw,
            "aspect": aspect,
            "expected": [float(v) for v in normalized],
        })
    for label, case_input, cam_aspect, expected in invariance_cases(real):
        norm_fixtures.append({
            "label": label,
            "input": case_input,
            "aspect": cam_aspect,
            "expected": [float(v) for v in expected],
        })

    out_dir.mkdir(parents=True, exist_ok=True)
    norm_path = out_dir / "normalize_fixtures.json"
    with open(norm_path, "w", encoding="utf-8") as f:
        json.dump(norm_fixtures, f)
    print(f"[{mode}] normalize fixtures -> {norm_path}  ({len(norm_fixtures)} cases: "
          f"{', '.join(c['label'] for c in norm_fixtures)})")

    # Model fixtures: what the .h5 says for the normalized real rows. The
    # exported browser model must reproduce these probs (tolerance ~1e-4).
    import tensorflow as tf
    model = tf.keras.models.load_model(model_path, compile=False)
    model_fixtures = []
    for label, raw, aspect in real:
        normalized = normalize_landmarks(raw, aspect=aspect)
        probs = model(tf.constant([normalized], dtype=tf.float32),
                      training=False).numpy()[0]
        model_fixtures.append({
            "label": label,
            "input": [float(v) for v in normalized],
            "expected_probs": [float(p) for p in probs],
        })
    model_fixtures_path = out_dir / "model_fixtures.json"
    with open(model_fixtures_path, "w", encoding="utf-8") as f:
        json.dump(model_fixtures, f)
    print(f"[{mode}] model fixtures     -> {model_fixtures_path}  ({len(model_fixtures)} cases)")


if __name__ == "__main__":
    main()
