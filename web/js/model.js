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

/**
 * Fold inference-mode BatchNorm into one multiply-add per channel, computed
 * once at load from the constant gamma/beta/mean/variance:
 *   y = gamma*(x-mean)/sqrt(var+eps) + beta = x*scale + shift
 * with scale = gamma/sqrt(var+eps) and shift = beta - mean*scale. This lifts
 * the per-element sqrt/div out of the per-frame path (they only depend on the
 * fixed weights). scale/shift are kept in float64 so the folded form stays
 * within ~1 ULP of the original Keras expression — the parity gate (1e-4)
 * covers it. Shared by both models so the two constructors stay identical.
 */
function decodeBatchNorm(l) {
  const gamma = decodeF32(l.gamma);
  const beta = decodeF32(l.beta);
  const mean = decodeF32(l.mean);
  const variance = decodeF32(l.variance);
  const eps = l.epsilon;
  const n = gamma.length;
  const scale = new Float64Array(n);
  const shift = new Float64Array(n);
  for (let i = 0; i < n; i++) {
    scale[i] = gamma[i] / Math.sqrt(variance[i] + eps);
    shift[i] = beta[i] - mean[i] * scale[i];
  }
  return { ...l, _scale: scale, _shift: shift };
}

// dense/batchnorm write into a caller-provided destination instead of
// allocating: predictions run dozens of times per second and the per-call
// Float32Arrays were the main source of GC pressure on phones. The arithmetic
// and loop order are IDENTICAL to the allocating originals (parity-gated).

function dense(x, layer, out) {
  const [nIn, nOut] = layer.kernel_shape;
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

function batchnorm(x, layer, out) {
  // Inference-mode BN as one multiply-add per channel; scale/shift were folded
  // from gamma/beta/mean/variance at load (see decodeBatchNorm). Safe when
  // out === x (element-wise) — the TCN uses it in place over conv rows.
  const s = layer._scale, sh = layer._shift;
  for (let i = 0; i < x.length; i++) out[i] = x[i] * s[i] + sh[i];
  return out;
}

export class DenseModel {
  constructor(spec) {
    this.inputDim = spec.input_dim;
    this.numClasses = spec.num_classes;
    let width = this.inputDim;
    this._layers = spec.layers.map((l) => {
      const decoded = l.type === "batchnorm" ? decodeBatchNorm(l) : { ...l };
      if (l.type === "dense") {
        decoded._kernel = decodeF32(l.kernel);
        decoded._bias = decodeF32(l.bias);
        width = l.kernel_shape[1];
      } else if (l.type !== "batchnorm") {
        throw new Error(`Unknown layer type in weight file: ${l.type}`);
      }
      // Constructor-owned output buffer, reused every predict (no per-frame
      // allocation). BN keeps the running width of the preceding dense layer.
      decoded._out = new Float32Array(width);
      return decoded;
    });
  }

  /**
   * @param {Float32Array} x normalized 63-value vector @returns softmax probs
   * The returned array is OWNED BY THE MODEL and only valid until the next
   * predict() call — copy it if you need it longer.
   */
  predict(x) {
    if (x.length !== this.inputDim) {
      throw new Error(`Expected ${this.inputDim} inputs, got ${x.length}`);
    }
    let h = x;
    for (const layer of this._layers) {
      h = layer.type === "dense"
        ? dense(h, layer, layer._out)
        : batchnorm(h, layer, layer._out);
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
  // Constructor-owned output rows (layer._outSeq): the bias init below fully
  // overwrites every element, so reuse is safe — and it removes the ~T
  // Float32Array allocations per conv layer that dominated GC churn.
  const out = layer._outSeq;
  for (let t = 0; t < T; t++) {
    const o = out[t];
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
  }
  return out;
}

/** Mean over the time axis: T rows of C -> one Float32Array of C (in `out`). */
function globalAvgPool1d(seq, out) {
  const T = seq.length, C = seq[0].length;
  out.fill(0);
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
    // Track the tensor shape through the stack so every layer can own its
    // output buffer(s): sequence stage -> seqLen rows of `width`; after the
    // global pool -> a single `width` vector.
    let width = this.featureDim;
    let pooled = false;
    this._layers = spec.layers.map((l) => {
      const decoded = l.type === "batchnorm" ? decodeBatchNorm(l) : { ...l };
      if (l.type === "conv1d" || l.type === "dense") {
        decoded._kernel = decodeF32(l.kernel);
        decoded._bias = decodeF32(l.bias);
        width = l.kernel_shape[l.kernel_shape.length - 1];
      } else if (l.type === "global_avg_pool1d") {
        pooled = true;
      } else if (l.type !== "batchnorm") {
        throw new Error(`Unknown layer type in weight file: ${l.type}`);
      }
      if (l.type === "conv1d") {
        decoded._outSeq = Array.from(
          { length: this.seqLen }, () => new Float32Array(width));
      } else if (l.type === "dense" || (l.type === "batchnorm" && pooled)) {
        decoded._out = new Float32Array(width);
      }
      // Sequence-stage batchnorm gets no buffer: it runs IN PLACE over the
      // previous conv layer's rows (element-wise, so self-write is safe).
      return decoded;
    });
  }

  /**
   * @param {Array<Float32Array>} seq  featureDim-wide rows, length seqLen
   * @returns {Float32Array} softmax over the glosses
   * The returned array is OWNED BY THE MODEL and only valid until the next
   * predict() call — copy it if you need it longer.
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
        if (Array.isArray(h)) {
          for (const row of h) batchnorm(row, layer, row); // in place
        } else {
          h = batchnorm(h, layer, layer._out);
        }
      } else if (layer.type === "global_avg_pool1d") {
        h = globalAvgPool1d(h, this._pooledOut
          || (this._pooledOut = new Float32Array(h[0].length)));
      } else { // dense (post-pool: h is a vector)
        h = dense(h, layer, layer._out);
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
