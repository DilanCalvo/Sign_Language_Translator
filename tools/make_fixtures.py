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

ROWS_PER_FILE = 1     # spread cases across sessions/letters, not one big file
MAX_REAL_CASES = 10
SEED = 42


def collect_real_rows():
    """One raw row per CSV (different letter each time when possible)."""
    rng = np.random.default_rng(SEED)
    cases, seen_labels = [], set()
    for path in sorted(CSV_DIR.glob("*.csv"))[:MAX_REAL_CASES]:
        with open(path, encoding="utf-8") as f:
            rows = list(csv.reader(f))[1:]  # skip header
        if not rows:
            continue
        # Prefer a label we have not covered yet so cases span the alphabet.
        fresh = [r for r in rows if r[0] not in seen_labels] or rows
        row = fresh[int(rng.integers(len(fresh)))]
        seen_labels.add(row[0])
        cases.append((row[0], [float(v) for v in row[1:64]]))
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
