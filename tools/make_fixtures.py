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

from src.utils import normalize_landmarks  # noqa: E402

CSV_DIR = REPO / "data" / "real_capture" / "letters"
OUT_DIR = REPO / "web" / "fixtures"
MODEL_PATH = REPO / "model" / "model_one_hand.h5"

SEED = 42


def collect_real_rows():
    """One raw row per LETTER, sampled across ALL capture CSVs.

    Per-letter (not per-file) coverage on purpose: every class present in the
    capture data gets a parity case, so a bug that only shows up in some part
    of the class range cannot slip through the gate. Deterministic (fixed
    seed) so reruns produce identical fixtures for identical data.
    """
    rng = np.random.default_rng(SEED)
    by_label = {}
    for path in sorted(CSV_DIR.glob("*.csv")):
        with open(path, encoding="utf-8") as f:
            for row in list(csv.reader(f))[1:]:  # skip header
                if row:
                    by_label.setdefault(row[0], []).append(row)
    cases = []
    for label in sorted(by_label):
        rows = by_label[label]
        row = rows[int(rng.integers(len(rows)))]
        cases.append((label, [float(v) for v in row[1:64]]))
    return cases


def edge_cases():
    """Synthetic inputs that exercise the scale guard (scale < 1e-6 -> 1.0)."""
    zeros = [0.0] * 63
    # All 21 points identical: centering yields zeros, scale is 0 -> guard path.
    collapsed = [0.4, 0.5, 0.1] * 21
    return [("edge_all_zeros", zeros), ("edge_collapsed_hand", collapsed)]


def main():
    if not CSV_DIR.is_dir():
        sys.exit(f"[ERROR] No capture CSVs at {CSV_DIR}")

    real = collect_real_rows()
    if not real:
        sys.exit(f"[ERROR] No rows found in {CSV_DIR}")

    norm_fixtures = []
    for label, raw in real + edge_cases():
        normalized = normalize_landmarks(raw)
        norm_fixtures.append({
            "label": label,
            "input": raw,
            "expected": [float(v) for v in normalized],
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
    for label, raw in real:
        normalized = normalize_landmarks(raw)
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
