"""
Export the letters model to a browser-friendly JSON weight file.

Why not the official tensorflowjs converter: it does not install on Windows
(its dependency chain pulls uvloop, which has no Windows support), and its
bundled Keras 2 could not read an h5 saved by this project's Keras 3 anyway.
Loading the model with the SAME Keras that trained it and dumping the weights
ourselves has zero format risk — and the web side verifies the result against
web/fixtures/model_fixtures.json, so a broken export fails loudly.

At inference the model is only Dense (matmul + bias + activation) and
BatchNormalization (elementwise affine); Dropout and InputLayer vanish. The
export walks the layers in order and keeps just those two types. Weights are
base64-encoded little-endian float32 (~25% smaller than JSON numbers and
decoded in three lines of JS).

Output: web/model/letters/model_weights.json  (read by web/js/model.js)

Usage (project venv, from the repo root):
    venv/Scripts/python tools/export_model_json.py

Re-run after every retrain of model_one_hand.h5.
"""

import base64
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
MODEL_PATH = REPO / "model" / "model_one_hand.h5"
OUT_PATH = REPO / "web" / "model" / "letters" / "model_weights.json"


def _b64(arr):
    """Little-endian float32 bytes, base64-encoded."""
    return base64.b64encode(
        np.ascontiguousarray(arr, dtype="<f4").tobytes()
    ).decode("ascii")


def main():
    import tensorflow as tf

    model = tf.keras.models.load_model(MODEL_PATH, compile=False)

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
        else:
            sys.exit(f"[ERROR] Unsupported layer type for export: {cls}. "
                     "Extend tools/export_model_json.py before retraining "
                     "with new layer types.")

    payload = {
        "input_dim": int(model.input_shape[-1]),
        "num_classes": int(model.output_shape[-1]),
        "layers": layers_out,
    }
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_PATH, "w", encoding="utf-8") as f:
        json.dump(payload, f)

    kb = OUT_PATH.stat().st_size / 1024
    print(f"Exported {len(layers_out)} inference layers "
          f"({payload['input_dim']} -> {payload['num_classes']} classes) "
          f"-> {OUT_PATH}  ({kb:.0f} KB)")


if __name__ == "__main__":
    main()
