/**
 * Minimal inference engines for the browser models. No TF.js (its converter
 * does not install on Windows); each model is a small hand-rolled forward pass.
 *
 *   DenseModel  — letters/numbers: Dense + BatchNorm over a 63-value vector.
 *   TCNModel    — words: a temporal TCN over a (T, feature_dim) sequence —
 *                 Conv1D (causal, dilated) + BatchNorm + GlobalAveragePooling1D
 *                 + Dense. Reuses the same dense()/batchnorm() primitives.
 *
 * Both load web/model/<mode>/model_weights.json (written by
 * tools/export_model_json.py from the matching .h5).
 *
 * PARITY RULE: predictions must match the Python model. Verified against each
 * mode's model_fixtures.json (real inputs -> expected softmax) by
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

export class DenseModel {
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

export async function loadDenseModel(url) {
  const res = await fetch(url);
  if (!res.ok) throw new Error(`Could not load model weights: ${url} (${res.status})`);
  return new DenseModel(await res.json());
}

// ---- Word TCN (temporal) ------------------------------------------------

/**
 * Causal, dilated 1-D convolution over a sequence. `seq` is an array of T rows
 * (each a Float32Array of cIn); returns T rows of cOut.
 * Keras "causal" left-pads with (k-1)*d zeros, so output[t] depends only on
 * inputs at t and earlier: out[t] = Σ_j W[j] · x[t-(k-1)d+j·d] (+ bias), with
 * out-of-range (t-...<0) taps treated as zero. Kernel layout is [k, cIn, cOut]
 * row-major: K[j*cIn*cOut + ci*cOut + co].
 */
function conv1d(seq, layer) {
  const [k, cIn, cOut] = layer.kernel_shape;
  const d = layer.dilation;
  const K = layer._kernel, B = layer._bias;
  const relu = layer.activation === "relu";
  const T = seq.length;
  const out = new Array(T);
  for (let t = 0; t < T; t++) {
    const o = new Float32Array(cOut);
    for (let co = 0; co < cOut; co++) o[co] = B[co];
    for (let j = 0; j < k; j++) {
      const st = t - (k - 1) * d + j * d;
      if (st < 0) continue; // causal left-pad = zeros, contributes nothing
      const row = seq[st];
      const kBase = j * cIn * cOut;
      for (let ci = 0; ci < cIn; ci++) {
        const xi = row[ci];
        if (xi === 0) continue;
        const kk = kBase + ci * cOut;
        for (let co = 0; co < cOut; co++) o[co] += xi * K[kk + co];
      }
    }
    if (relu) for (let co = 0; co < cOut; co++) if (o[co] < 0) o[co] = 0;
    out[t] = o;
  }
  return out;
}

/** Mean over the time axis: T rows of C -> one Float32Array of C. */
function globalAvgPool1d(seq) {
  const T = seq.length, C = seq[0].length;
  const out = new Float32Array(C);
  for (let t = 0; t < T; t++) {
    const row = seq[t];
    for (let c = 0; c < C; c++) out[c] += row[c];
  }
  for (let c = 0; c < C; c++) out[c] /= T;
  return out;
}

export class TCNModel {
  constructor(spec) {
    this.seqLen = spec.seq_len;
    this.featureDim = spec.feature_dim;
    this.numClasses = spec.num_classes;
    this._layers = spec.layers.map((l) => {
      const decoded = { ...l };
      if (l.type === "conv1d" || l.type === "dense") {
        decoded._kernel = decodeF32(l.kernel);
        decoded._bias = decodeF32(l.bias);
      } else if (l.type === "batchnorm") {
        decoded._gamma = decodeF32(l.gamma);
        decoded._beta = decodeF32(l.beta);
        decoded._mean = decodeF32(l.mean);
        decoded._variance = decodeF32(l.variance);
      } else if (l.type === "global_avg_pool1d") {
        // no weights
      } else {
        throw new Error(`Unknown layer type in weight file: ${l.type}`);
      }
      return decoded;
    });
  }

  /**
   * @param {Array<Float32Array>} seq  featureDim-wide rows, length seqLen
   * @returns {Float32Array} softmax over the glosses
   */
  predict(seq) {
    if (seq.length !== this.seqLen || seq[0].length !== this.featureDim) {
      throw new Error(
        `Expected ${this.seqLen}x${this.featureDim} sequence, got ` +
        `${seq.length}x${seq[0] ? seq[0].length : "?"}`);
    }
    // `h` is a sequence (array of rows) until global pooling collapses it to a
    // single vector; the layers are ordered so that transition happens once.
    let h = seq;
    for (const layer of this._layers) {
      if (layer.type === "conv1d") {
        h = conv1d(h, layer);
      } else if (layer.type === "batchnorm") {
        h = Array.isArray(h) ? h.map((row) => batchnorm(row, layer))
                             : batchnorm(h, layer);
      } else if (layer.type === "global_avg_pool1d") {
        h = globalAvgPool1d(h);
      } else { // dense (post-pool: h is a vector)
        h = dense(h, layer);
      }
    }
    return h;
  }
}

export async function loadWordModel(url) {
  const res = await fetch(url);
  if (!res.ok) throw new Error(`Could not load word model: ${url} (${res.status})`);
  return new TCNModel(await res.json());
}
