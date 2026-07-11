"""
One JSON record per training run, for every model (letters, numbers, words).

Model iterations are only comparable when each run leaves a record: which
classes, how much data, how many capture sessions, what accuracy, which
hyperparameters and which code (git SHA). Lightweight on purpose: one file per
run under runs/, versioned in git.

Files are named runs/<model>_<timestamp>.json. Older word runs predate the
model prefix (plain <timestamp>.json) — they are left as they are.
"""

import json
import os
import subprocess
from datetime import datetime

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _git_sha():
    try:
        return subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"], cwd=HERE,
            capture_output=True, text=True, check=True).stdout.strip()
    except Exception:
        return "unknown"


def write_run_metadata(model_name, class_names, samples_per_class, n_sessions,
                       val_accuracy, config):
    """Write runs/<model>_<stamp>.json describing one training run.

    model_name        "letters" | "numbers" | "words"
    class_names       ordered class list (index == model output index)
    samples_per_class {class: count} over the full dataset
    n_sessions        distinct capture sessions (source files for the static
                      models, manifest session_ids for words)
    val_accuracy      final validation accuracy, or None when no val split
    config            dict of the hyperparameters that shaped this run
    """
    runs_dir = os.path.join(HERE, "runs")
    os.makedirs(runs_dir, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    record = {
        "model": model_name,
        "timestamp": stamp,
        "git_sha": _git_sha(),
        "vocab": list(class_names),
        "n_classes": len(class_names),
        "n_sessions": int(n_sessions),
        "n_samples": int(sum(samples_per_class.values())),
        "samples_per_class": {k: int(v) for k, v in sorted(samples_per_class.items())},
        "val_accuracy": round(float(val_accuracy), 4) if val_accuracy is not None else None,
        "config": config,
    }
    path = os.path.join(runs_dir, f"{model_name}_{stamp}.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(record, f, indent=2, ensure_ascii=False)
    print(f"Run    -> runs/{model_name}_{stamp}.json")
