"""
Propagate a retrained letters model to the web version, in one command.

The web app does NOT read model/model_one_hand.h5 directly — it reads a
static export (web/model/letters/model_weights.json) plus a copy of the
labels file. Nothing regenerates those automatically after
training/train_letters.py runs, so a retrain that isn't followed by this
script leaves the deployed web app silently serving the OLD model: no error,
no warning, just wrong predictions. That is the failure mode this script
exists to remove.

What it does, in order, aborting immediately if any step fails:
    1. tools/export_model_json.py   -> web/model/letters/model_weights.json
    2. tools/make_fixtures.py       -> web/fixtures/*.json (against the NEW model)
    3. copy model/labels_one_hand.json -> web/model/labels_one_hand.json

What it deliberately does NOT do (still a manual step, on purpose — this
script touches files, it does not silently declare success):
    Open web/js/utils.test.html (served over http, not file://) and confirm
    "ALL N TESTS PASS" before trusting the deployed model. The script prints
    the exact command for this as its last line.

Usage (project venv, from the repo root), after training/train_letters.py:
    venv/Scripts/python tools/update_web_letters.py
"""

import shutil
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
MODEL_PATH = REPO / "model" / "model_one_hand.h5"
LABELS_SRC = REPO / "model" / "labels_one_hand.json"
LABELS_DST = REPO / "web" / "model" / "labels_one_hand.json"


def run_step(description, script_name):
    print(f"\n==> {description}")
    result = subprocess.run([sys.executable, str(REPO / "tools" / script_name)])
    if result.returncode != 0:
        sys.exit(
            f"\n[ABORTED] {script_name} failed (exit {result.returncode}).\n"
            "The web files were NOT fully updated — do not deploy until this "
            "is fixed and the script runs clean."
        )


def main():
    if not MODEL_PATH.is_file():
        sys.exit(f"[ERROR] No model at {MODEL_PATH}. Train it first: "
                 "python training/train_letters.py")

    run_step("Exporting model weights for the browser", "export_model_json.py")
    run_step("Regenerating parity fixtures against the new model", "make_fixtures.py")

    print(f"\n==> Copying labels ({LABELS_SRC.name})")
    LABELS_DST.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(LABELS_SRC, LABELS_DST)
    print(f"    {LABELS_SRC} -> {LABELS_DST}")

    print(
        "\n" + "=" * 64 +
        "\nWeb files updated. ONE step is still manual, on purpose:\n"
        "  1. From web/, run:  python -m http.server 8000\n"
        "  2. Open:            http://localhost:8000/js/utils.test.html\n"
        "  3. Confirm it says: ALL <N> TESTS PASS\n"
        "Do not redeploy until step 3 is green.\n" + "=" * 64
    )


if __name__ == "__main__":
    main()
