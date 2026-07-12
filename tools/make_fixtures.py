"""
Generate parity fixtures for the web port.

The JS reimplementation of normalize_landmarks and the TF.js-converted letters
model must produce EXACTLY what Python produces — a mismatch is a silent bug
(the model still runs, it just predicts garbage). These fixtures turn that
silent failure into a loud one: web/js/utils.test.html replays them in the
browser and shows PASS/FAIL per case.

Outputs (consumed by web/js/utils.test.html):
    web/fixtures/normalize_fixtures.json  raw 63-value row -> normalized 63
    web/fixtures/model_fixtures.json      normalized 63    -> softmax probs

Inputs are REAL capture rows (data/real_capture/letters/*.csv) spread across
letters and sessions, plus synthetic edge cases for the scale guard.

Usage (project venv, from the repo root):
    venv/Scripts/python tools/make_fixtures.py
"""

import csv
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from config import LEGACY_CAPTURE_ASPECT  # noqa: E402
from src.utils import normalize_landmarks  # noqa: E402

CSV_DIR = REPO / "data" / "real_capture" / "letters"
OUT_DIR = REPO / "web" / "fixtures"
MODEL_PATH = REPO / "model" / "model_one_hand.h5"

SEED = 42


def collect_real_rows():
    """One (label, raw63, aspect) per LETTER, sampled across ALL capture CSVs.

    Per-letter (not per-file) coverage on purpose: every class present in the
    capture data gets a parity case, so a bug that only shows up in some part
    of the class range cannot slip through the gate. Deterministic (fixed
    seed) so reruns produce identical fixtures for identical data.

    The aspect comes from the CSV's own "aspect" column when present
    (self-describing captures); legacy files fall back to
    config.LEGACY_CAPTURE_ASPECT — the same rule the training loader applies.
    """
    rng = np.random.default_rng(SEED)
    by_label = {}
    for path in sorted(CSV_DIR.glob("*.csv")):
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


def main():
    if not CSV_DIR.is_dir():
        sys.exit(f"[ERROR] No capture CSVs at {CSV_DIR}")

    real = collect_real_rows()
    if not real:
        sys.exit(f"[ERROR] No rows found in {CSV_DIR}")

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

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    norm_path = OUT_DIR / "normalize_fixtures.json"
    with open(norm_path, "w", encoding="utf-8") as f:
        json.dump(norm_fixtures, f)
    print(f"normalize fixtures -> {norm_path}  ({len(norm_fixtures)} cases: "
          f"{', '.join(c['label'] for c in norm_fixtures)})")

    # Model fixtures: what the .h5 says for the normalized real rows. The
    # TF.js-converted model must reproduce these probs (tolerance ~1e-4).
    import tensorflow as tf
    model = tf.keras.models.load_model(MODEL_PATH, compile=False)
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
    model_path = OUT_DIR / "model_fixtures.json"
    with open(model_path, "w", encoding="utf-8") as f:
        json.dump(model_fixtures, f)
    print(f"model fixtures     -> {model_path}  ({len(model_fixtures)} cases)")


if __name__ == "__main__":
    main()
