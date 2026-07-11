/**
 * Minimal inference engine for the letters model.
 *
 * Loads web/model/letters/model_weights.json (written by
 * tools/export_model_json.py from model_one_hand.h5) and runs the forward
 * pass in plain JS. No TF.js: the official converter does not install on
 * Windows and at inference this model is only Dense + BatchNorm — a hand
 * loop over ~200K weights runs in microseconds per frame.
 *
 * PARITY RULE: predictions must match the Python model. Verified against
 * web/fixtures/model_fixtures.json (real inputs -> expected softmax) by
 * utils.test.html — keep it green after any change here.
 */

function decodeF32(b64) {
  const bin = atob(b64);
  const bytes = new Uint8Array(bin.length);
  for (let i = 0; i < bin.length; i++) bytes[i] = bin.charCodeAt(i);
  return new Float32Array(bytes.buffer); // base64 payload is little-endian f32
}

function dense(x, layer) {
  const [nIn, nOut] = layer.kernel_shape;
  const out = new Float32Array(nOut);
  const k = layer._kernel, b = layer._bias;
  // Keras kernel layout is [in, out] row-major: k[i * nOut + j].
  for (let j = 0; j < nOut; j++) out[j] = b[j];
  for (let i = 0; i < nIn; i++) {
    const xi = x[i];
    if (xi === 0) continue;
    const row = i * nOut;
    for (let j = 0; j < nOut; j++) out[j] += xi * k[row + j];
  }
  if (layer.activation === "relu") {
    for (let j = 0; j < nOut; j++) if (out[j] < 0) out[j] = 0;
  } else if (layer.activation === "softmax") {
    let max = -Infinity;
    for (let j = 0; j < nOut; j++) if (out[j] > max) max = out[j];
    let sum = 0;
    for (let j = 0; j < nOut; j++) { out[j] = Math.exp(out[j] - max); sum += out[j]; }
    for (let j = 0; j < nOut; j++) out[j] /= sum;
  } // "linear": nothing
  return out;
}

function batchnorm(x, layer) {
  // Inference-mode BN: y = gamma * (x - mean) / sqrt(var + eps) + beta.
  const out = new Float32Array(x.length);
  const { _gamma: g, _beta: b, _mean: m, _variance: v, epsilon: eps } = layer;
  for (let i = 0; i < x.length; i++) {
    out[i] = g[i] * (x[i] - m[i]) / Math.sqrt(v[i] + eps) + b[i];
  }
  return out;
}

export class LettersModel {
  constructor(spec) {
    this.inputDim = spec.input_dim;
    this.numClasses = spec.num_classes;
    this._layers = spec.layers.map((l) => {
      const decoded = { ...l };
      if (l.type === "dense") {
        decoded._kernel = decodeF32(l.kernel);
        decoded._bias = decodeF32(l.bias);
      } else if (l.type === "batchnorm") {
        decoded._gamma = decodeF32(l.gamma);
        decoded._beta = decodeF32(l.beta);
        decoded._mean = decodeF32(l.mean);
        decoded._variance = decodeF32(l.variance);
      } else {
        throw new Error(`Unknown layer type in weight file: ${l.type}`);
      }
      return decoded;
    });
  }

  /** @param {Float32Array} x normalized 63-value vector @returns softmax probs */
  predict(x) {
    if (x.length !== this.inputDim) {
      throw new Error(`Expected ${this.inputDim} inputs, got ${x.length}`);
    }
    let h = x;
    for (const layer of this._layers) {
      h = layer.type === "dense" ? dense(h, layer) : batchnorm(h, layer);
    }
    return h;
  }
}

export async function loadLettersModel(url) {
  const res = await fetch(url);
  if (!res.ok) throw new Error(`Could not load model weights: ${url} (${res.status})`);
  return new LettersModel(await res.json());
}
