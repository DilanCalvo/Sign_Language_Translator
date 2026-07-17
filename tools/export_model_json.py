"""
Export a model (letters, numbers, or words) to a browser-friendly JSON weight
file.

Why not the official tensorflowjs converter: it does not install on Windows
(its dependency chain pulls uvloop, which has no Windows support), and its
bundled Keras 2 could not read an h5 saved by this project's Keras 3 anyway.
Loading the model with the SAME Keras that trained it and dumping the weights
ourselves has zero format risk — and the web side verifies the result against
the mode's model_fixtures.json, so a broken export fails loudly.

Two model families share this one exporter:
  - letters/numbers: Dense + BatchNorm over a 63-value single-hand input.
  - words: a temporal TCN — Conv1D (causal, dilated) + BatchNorm +
    GlobalAveragePooling1D + Dense over a (WORD_SEQ_LEN, WORD_FEATURE_DIM)
    sequence. web/js/model.js has a matching plain-JS forward pass for each.

At inference Dropout and InputLayer vanish; the export walks the layers in
order and keeps only the ones that do math (dense, batchnorm, conv1d,
global_avg_pool1d). Weights are base64-encoded little-endian float32 (~25%
smaller than JSON numbers and decoded in three lines of JS).

Output: web/model/<mode>/model_weights.json  (read by web/js/model.js)

Usage (project venv, from the repo root):
    venv/Scripts/python tools/export_model_json.py [letters|numbers|words]

`mode` defaults to "letters". Re-run after every retrain of that model
(tools/update_web.py does this for you).
"""

import base64
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent

# Per-mode source model + web output. Keep in sync with tools/make_fixtures.py
# and tools/update_web.py.
MODES = {
    "letters": {
        "model": REPO / "model" / "model_one_hand.h5",
        "out":   REPO / "web" / "model" / "letters" / "model_weights.json",
    },
    "numbers": {
        "model": REPO / "model" / "model_numbers.h5",
        "out":   REPO / "web" / "model" / "numbers" / "model_weights.json",
    },
    "words": {
        "model": REPO / "model" / "model_words.h5",
        "out":   REPO / "web" / "model" / "words" / "model_weights.json",
    },
}


def _b64(arr):
    """Little-endian float32 bytes, base64-encoded."""
    return base64.b64encode(
        np.ascontiguousarray(arr, dtype="<f4").tobytes()
    ).decode("ascii")


def main():
    mode = sys.argv[1] if len(sys.argv) > 1 else "letters"
    if mode not in MODES:
        sys.exit(f"[ERROR] Unknown mode '{mode}'. Use one of: "
                 f"{', '.join(MODES)}")
    model_path = MODES[mode]["model"]
    out_path = MODES[mode]["out"]
    if not model_path.is_file():
        sys.exit(f"[ERROR] No model at {model_path}. Train it first.")

    import tensorflow as tf

    model = tf.keras.models.load_model(model_path, compile=False)

    layers_out = []
    for layer in model.layers:
        cls = type(layer).__name__
        if cls in ("InputLayer", "Dropout"):
            continue  # no effect at inference
        if cls == "Dense":
            kernel, bias = layer.get_weights()
            layers_out.append({
                "type": "dense",
                "activation": layer.activation.__name__,  # relu/softmax/linear
                "kernel_shape": list(kernel.shape),        # [in, out]
                "kernel": _b64(kernel),
                "bias": _b64(bias),
            })
        elif cls == "Conv1D":
            # Keras Conv1D kernel is (kernel_size, in_channels, filters).
            # Causal + dilation are needed to reproduce the temporal receptive
            # field exactly; the JS side left-pads and strides accordingly.
            kernel, bias = layer.get_weights()
            layers_out.append({
                "type": "conv1d",
                "activation": layer.activation.__name__,     # relu/linear
                "kernel_shape": list(kernel.shape),          # [k, in, out]
                "dilation": int(layer.dilation_rate[0]),
                "padding": layer.padding,                    # "causal"
                "kernel": _b64(kernel),
                "bias": _b64(bias),
            })
        elif cls == "BatchNormalization":
            gamma, beta, mean, variance = layer.get_weights()
            layers_out.append({
                "type": "batchnorm",
                "epsilon": float(layer.epsilon),
                "gamma": _b64(gamma),
                "beta": _b64(beta),
                "mean": _b64(mean),
                "variance": _b64(variance),
            })
        elif cls == "GlobalAveragePooling1D":
            # Mean over the time axis: (T, C) -> (C,). No weights.
            layers_out.append({"type": "global_avg_pool1d"})
        else:
            sys.exit(f"[ERROR] Unsupported layer type for export: {cls}. "
                     "Extend tools/export_model_json.py before retraining "
                     "with new layer types.")

    # Sequence models (the word TCN) have a 3D input (None, T, D); static models
    # (letters/numbers) have a 2D input (None, D). The web side reads whichever
    # shape fields are present to size and validate its forward pass. Key order
    # for the static case is kept as it always was, so re-exporting an unchanged
    # letters/numbers model stays byte-identical.
    input_shape = model.input_shape
    num_classes = int(model.output_shape[-1])
    if len(input_shape) == 3:
        payload = {
            "seq_len": int(input_shape[1]),
            "feature_dim": int(input_shape[2]),
            "num_classes": num_classes,
            "layers": layers_out,
        }
        shape_desc = f"({payload['seq_len']}x{payload['feature_dim']})"
    else:
        payload = {
            "input_dim": int(input_shape[-1]),
            "num_classes": num_classes,
            "layers": layers_out,
        }
        shape_desc = f"{payload['input_dim']}"

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f)

    kb = out_path.stat().st_size / 1024
    print(f"[{mode}] exported {len(layers_out)} inference layers "
          f"({shape_desc} -> {payload['num_classes']} classes) "
          f"-> {out_path}  ({kb:.0f} KB)")


if __name__ == "__main__":
    main()
