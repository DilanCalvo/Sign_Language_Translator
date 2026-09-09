"""Performance graphs for the three models, built from runs/*.json.

Reads every training run record (see training/run_log.py) and saves three
PNGs to reports/: accuracy over time, latest accuracy per model, and the
class balance of the latest run per model.
"""

import glob
import json
import os
from datetime import datetime

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
RUNS_DIR = os.path.join(ROOT, "runs")
OUT_DIR = os.path.join(ROOT, "reports")

ORDER = ["letters", "numbers", "words"]
COLORS = {"letters": "#2a78d6", "numbers": "#eb6834", "words": "#1baf7a"}

SURFACE, GRID, AXIS, INK, MUTED = "#fcfcfb", "#e1e0d9", "#c3c2b7", "#0b0b0b", "#898781"


def load_runs():
    runs = []
    for path in sorted(glob.glob(os.path.join(RUNS_DIR, "*.json"))):
        with open(path, encoding="utf-8") as f:
            r = json.load(f)
        r["model"] = r.get("model", "words")  # pre-prefix runs (June 2026) are all words
        r["dt"] = datetime.strptime(r["timestamp"], "%Y%m%d_%H%M%S")
        runs.append(r)
    return runs


def latest_per_model(runs):
    latest = {}
    for r in runs:
        m = r["model"]
        if m not in latest or r["dt"] > latest[m]["dt"]:
            latest[m] = r
    return latest


def style(ax):
    ax.set_facecolor(SURFACE)
    ax.grid(axis="y", color=GRID, linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color(AXIS)
    ax.spines["bottom"].set_color(AXIS)
    ax.tick_params(colors=MUTED, labelsize=9)


def plot_accuracy_over_time(runs, out_path):
    fig, ax = plt.subplots(figsize=(8, 5))
    for model in ORDER:
        pts = sorted((r["dt"], r["val_accuracy"]) for r in runs
                     if r["model"] == model and r["val_accuracy"] is not None)
        if not pts:
            continue
        xs, ys = zip(*pts)
        ax.plot(xs, [y * 100 for y in ys], marker="o", markersize=6,
                linewidth=2, color=COLORS[model], label=model)
    style(ax)
    ax.set_ylabel("Validation accuracy (%)")
    ax.set_title("Validation accuracy across training runs")
    ax.legend(frameon=False)
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_latest_accuracy(latest, out_path):
    models = [m for m in ORDER if m in latest and latest[m]["val_accuracy"] is not None]
    values = [latest[m]["val_accuracy"] * 100 for m in models]

    fig, ax = plt.subplots(figsize=(6, 4.5))
    bars = ax.bar(models, values, color=[COLORS[m] for m in models], width=0.5, zorder=3)
    for b, v in zip(bars, values):
        ax.text(b.get_x() + b.get_width() / 2, v + 1.5, f"{v:.1f}%",
                ha="center", fontsize=10, color=INK)
    style(ax)
    ax.set_ylim(0, 108)
    ax.set_ylabel("Validation accuracy (%)")
    ax.set_title("Current accuracy (latest run per model)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_samples_per_class(latest, out_path):
    models = [m for m in ORDER if m in latest]
    max_classes = max(len(latest[m]["samples_per_class"]) for m in models)
    height = max(4, 0.35 * max_classes)

    fig, axes = plt.subplots(1, len(models), figsize=(5 * len(models), height))
    if len(models) == 1:
        axes = [axes]
    for ax, model in zip(axes, models):
        counts = latest[model]["samples_per_class"]
        classes, values = list(counts.keys()), list(counts.values())
        ax.barh(classes, values, color=COLORS[model], zorder=3)
        style(ax)
        ax.set_title(f"{model}  (run {latest[model]['timestamp']})")
        ax.set_xlabel("samples")
        ax.invert_yaxis()
    fig.suptitle("Samples per class — latest run per model")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main():
    runs = load_runs()
    if not runs:
        print("No runs found in runs/.")
        return
    os.makedirs(OUT_DIR, exist_ok=True)
    latest = latest_per_model(runs)

    plot_accuracy_over_time(runs, os.path.join(OUT_DIR, "accuracy_over_time.png"))
    plot_latest_accuracy(latest, os.path.join(OUT_DIR, "accuracy_latest.png"))
    plot_samples_per_class(latest, os.path.join(OUT_DIR, "samples_per_class.png"))
    print(f"Saved 3 graphs -> {os.path.relpath(OUT_DIR, ROOT)}/")


if __name__ == "__main__":
    main()
