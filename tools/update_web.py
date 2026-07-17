"""
Propagate a retrained model (letters, numbers, or words) to the web version,
in one command.

The web app does NOT read the .h5 models directly — it reads a static export
(web/model/<mode>/model_weights.json) plus a copy of the labels file. Nothing
regenerates those automatically after training runs, so a retrain that isn't
followed by this script leaves the deployed web app silently serving the OLD
model: no error, no warning, just wrong predictions. That is the failure mode
this script exists to remove.

What it does, in order, aborting immediately if any step fails:
    1. tools/export_model_json.py <mode>  -> web/model/<mode>/model_weights.json
    2. tools/make_fixtures.py <mode>      -> web/fixtures/[<mode>/]*.json (NEW model)
    3. copy model/labels_<mode>.json      -> web/model/labels_<mode>.json
    4. (words only) copy the pose .task into web/ if it is not there yet — the
       word pipeline needs MediaPipe pose for the body anchor. Copied only when
       missing (it is a large, static asset, not a per-retrain artifact).

What it deliberately does NOT do (still a manual step, on purpose — this
script touches files, it does not silently declare success):
    Open web/js/utils.test.html (served over http, not file://) and confirm
    "ALL N TESTS PASS" before trusting the deployed model. The one test page
    covers ALL modes, so run it once after updating any. The script prints the
    exact command for this as its last line.

Usage (project venv, from the repo root), after the matching trainer:
    venv/Scripts/python tools/update_web.py [letters|numbers|words]

`mode` defaults to "letters".
"""

import shutil
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent

# Per-mode trained model + labels. The export/fixtures paths live in the two
# scripts this orchestrates (export_model_json.py, make_fixtures.py); here we
# only need the source model (existence check), the labels copy, and (words
# only) any large static assets to copy once. Keep the mode set in sync with
# those scripts.
MODES = {
    "letters": {
        "model":      REPO / "model" / "model_one_hand.h5",
        "labels_src": REPO / "model" / "labels_one_hand.json",
        "labels_dst": REPO / "web" / "model" / "labels_one_hand.json",
        "train_cmd":  "python training/train_letters.py",
    },
    "numbers": {
        "model":      REPO / "model" / "model_numbers.h5",
        "labels_src": REPO / "model" / "labels_numbers.json",
        "labels_dst": REPO / "web" / "model" / "labels_numbers.json",
        "train_cmd":  "python training/train_numbers.py",
    },
    "words": {
        "model":      REPO / "model" / "model_words.h5",
        "labels_src": REPO / "model" / "labels_words.json",
        "labels_dst": REPO / "web" / "model" / "labels_words.json",
        "train_cmd":  "python training/train_words.py",
        # Copied once when missing (5.5 MB static asset, not a retrain artifact).
        "copy_if_missing": [
            (REPO / "model" / "pose_landmarker_lite.task",
             REPO / "web" / "model" / "pose_landmarker_lite.task"),
        ],
    },
}


def run_step(description, script_name, mode):
    print(f"\n==> {description}")
    result = subprocess.run(
        [sys.executable, str(REPO / "tools" / script_name), mode])
    if result.returncode != 0:
        sys.exit(
            f"\n[ABORTED] {script_name} {mode} failed (exit {result.returncode}).\n"
            "The web files were NOT fully updated — do not deploy until this "
            "is fixed and the script runs clean."
        )


def main():
    mode = sys.argv[1] if len(sys.argv) > 1 else "letters"
    if mode not in MODES:
        sys.exit(f"[ERROR] Unknown mode '{mode}'. Use one of: {', '.join(MODES)}")
    cfg = MODES[mode]

    if not cfg["model"].is_file():
        sys.exit(f"[ERROR] No model at {cfg['model']}. Train it first: "
                 f"{cfg['train_cmd']}")

    run_step(f"[{mode}] Exporting model weights for the browser",
             "export_model_json.py", mode)
    run_step(f"[{mode}] Regenerating parity fixtures against the new model",
             "make_fixtures.py", mode)

    print(f"\n==> [{mode}] Copying labels ({cfg['labels_src'].name})")
    cfg["labels_dst"].parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(cfg["labels_src"], cfg["labels_dst"])
    print(f"    {cfg['labels_src']} -> {cfg['labels_dst']}")

    for src, dst in cfg.get("copy_if_missing", []):
        if dst.is_file():
            continue
        if not src.is_file():
            sys.exit(f"[ERROR] Required asset missing: {src}")
        print(f"\n==> [{mode}] Copying static asset ({src.name}, first time)")
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(src, dst)
        print(f"    {src} -> {dst}")

    print(
        "\n" + "=" * 64 +
        f"\nWeb files updated for '{mode}'. ONE step is still manual, on purpose:\n"
        "  1. From web/, run:  python -m http.server 8000\n"
        "  2. Open:            http://localhost:8000/js/utils.test.html\n"
        "  3. Confirm it says: ALL <N> TESTS PASS  (covers all modes)\n"
        "Do not redeploy until step 3 is green.\n" + "=" * 64
    )


if __name__ == "__main__":
    main()
