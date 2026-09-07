/**
 * SanketVani — client-side feature and decode parity reference.
 *
 * Every number the browser feeds the model must be built exactly the way the
 * training pipeline built it, and the decode loop must match the one Cell 2
 * verifies through onnxruntime. This file is that contract in executable form;
 * `vocab.json` carries the same spec as data so the client can assert it.
 *
 *   Holistic → frameVector → resampleTo → bodyNormalise
 *            → encoder.onnx (once) → decoder.onnx (per token) → sentence
 */

export const FEATURE_DIM = 225;
export const DEFAULT_FRAMES = 192;

// slot ranges, landmark-major, (x, y, z) per landmark
export const SLOTS = {
  pose:      { start: 0,   count: 33 },   //   0 – 98
  leftHand:  { start: 99,  count: 21 },   //  99 – 161
  rightHand: { start: 162, count: 21 },   // 162 – 224
};

// MediaPipe Pose indices used to place and scale the body
const L_SHOULDER = 11, R_SHOULDER = 12;

/**
 * One MediaPipe Holistic result -> one 225-slot frame.
 *
 * The buffer starts zeroed and stays zeroed wherever a landmark is missing —
 * an untracked hand, a short landmark list, an absent z. That is exactly what
 * training saw, so the tensor shape never changes and the distribution matches.
 *
 * @param {{poseLandmarks?, leftHandLandmarks?, rightHandLandmarks?}} results
 * @returns {Float32Array} length 225
 */
export function frameVector(results) {
  const v = new Float32Array(FEATURE_DIM); // zero-filled === "missing"
  const put = (landmarks, { start, count }) => {
    if (!landmarks) return;
    const n = Math.min(count, landmarks.length);
    for (let i = 0; i < n; i++) {
      const lm = landmarks[i];
      if (!lm) continue;
      const o = start + i * 3;
      v[o] = Number.isFinite(lm.x) ? lm.x : 0;
      v[o + 1] = Number.isFinite(lm.y) ? lm.y : 0;
      v[o + 2] = Number.isFinite(lm.z) ? lm.z : 0;
    }
  };
  put(results.poseLandmarks, SLOTS.pose);
  put(results.leftHandLandmarks, SLOTS.leftHand);
  put(results.rightHandLandmarks, SLOTS.rightHand);
  return v;
}

/**
 * Uniform temporal resampling, identical to Cell 1:
 *   idx = floor(linspace(0, n - 1, T) + 0.5)
 * Never truncates — a short clip repeats frames, a long one is subsampled, and
 * both endpoints are always kept.
 *
 * @param {Float32Array[]} frames per-frame vectors from frameVector()
 * @returns {Float32Array} flat (T * 225)
 */
export function resampleTo(frames, T = DEFAULT_FRAMES) {
  const out = new Float32Array(T * FEATURE_DIM);
  if (frames.length === 0) return out;      // all-zero input beats a crash
  const last = frames.length - 1;
  for (let t = 0; t < T; t++) {
    const src = T === 1 ? 0 : Math.floor((t * last) / (T - 1) + 0.5);
    out.set(frames[Math.min(src, last)], t * FEATURE_DIM);
  }
  return out;
}

/**
 * Centre one frame on the shoulders and scale by their span — the mirror of
 * body_normalise() in Cell 2. Makes the model indifferent to where the signer
 * stands and how far they are from the camera.
 *
 * Rules that must not drift: a landmark that is exactly (0,0,0) is *missing*
 * and stays zero; a frame whose shoulder span is zero becomes all-zero.
 *
 * @param {Float32Array} v one 225-slot frame
 * @returns {Float32Array} normalised copy
 */
export function bodyNormalise(v) {
  const out = new Float32Array(FEATURE_DIM);
  const l = L_SHOULDER * 3, r = R_SHOULDER * 3;
  const span = Math.hypot(v[l] - v[r], v[l + 1] - v[r + 1]);
  if (!(span > 1e-6)) return out;           // no shoulders -> drop the frame
  const cx = (v[l] + v[r]) / 2;
  const cy = (v[l + 1] + v[r + 1]) / 2;
  const cz = (v[l + 2] + v[r + 2]) / 2;
  for (let i = 0; i < FEATURE_DIM / 3; i++) {
    const o = i * 3;
    if (v[o] === 0 && v[o + 1] === 0 && v[o + 2] === 0) continue;  // missing
    out[o] = (v[o] - cx) / span;
    out[o + 1] = (v[o + 1] - cy) / span;
    out[o + 2] = (v[o + 2] - cz) / span;
  }
  return out;
}

/** Apply bodyNormalise across a flat (T * 225) buffer, in place. */
export function bodyNormaliseFlat(flat) {
  const T = flat.length / FEATURE_DIM;
  for (let t = 0; t < T; t++) {
    const at = t * FEATURE_DIM;
    flat.set(bodyNormalise(flat.subarray(at, at + FEATURE_DIM)), at);
  }
  return flat;
}

/** Token ids -> sentence. Mirrors decode_ids() in Cell 2. */
export function detokenize(ids, tokens, special) {
  const words = [];
  for (const raw of ids) {
    const id = Number(raw);
    if (id === special.pad || id === special.bos) continue;
    if (id === special.eos) break;
    words.push(tokens[id] ?? "<unk>");
  }
  return words.join(" ").replace(/\s+([.,?!])/g, "$1");
}

/**
 * Load the two graphs and return a translator.
 *
 *   const t = await loadTranslator("./sanketvani_encoder.onnx",
 *                                  "./sanketvani_decoder.onnx", "./vocab.json");
 *   const sentence = await t.translate(collectedFrames);   // one per Holistic frame
 *
 * The encoder runs once per clip; the decoder runs once per generated token
 * over a token buffer that is always max_tokens wide. The graph is causal, so
 * the padding after the real prefix cannot influence it — which is why no
 * dynamic shapes or KV cache are needed on the client.
 */
export async function loadTranslator(encoderUrl, decoderUrl, vocabUrl,
                                     ort = globalThis.ort) {
  const vocab = await (await fetch(vocabUrl)).json();

  if (vocab.input.feature_dim !== FEATURE_DIM) {
    throw new Error(`vocab.json expects ${vocab.input.feature_dim} features, ` +
                    `parity.js is built for ${FEATURE_DIM}`);
  }
  for (const layer of vocab.input.layout) {
    const key = layer.name.replace(/_(\w)/g, (_, c) => c.toUpperCase());
    if (SLOTS[key]?.start !== layer.slice[0]) {
      throw new Error(`layout drift on '${layer.name}': vocab.json says ` +
                      `${layer.slice}, parity.js says ${SLOTS[key]?.start}`);
    }
  }

  const options = { executionProviders: ["webgpu", "wasm"],
                    graphOptimizationLevel: "all" };
  const encoder = await ort.InferenceSession.create(encoderUrl, options);
  const decoder = await ort.InferenceSession.create(decoderUrl, options);

  const special = vocab.special_ids;
  const maxTokens = vocab.max_tokens;
  const V = vocab.vocab_size;

  return {
    vocab, encoder, decoder,
    async translate(frames, T = vocab.input.num_frames ?? DEFAULT_FRAMES) {
      const flat = resampleTo(frames, T);
      if (vocab.input.body_normalise) bodyNormaliseFlat(flat);

      const { memory } = await encoder.run({
        landmarks: new ort.Tensor("float32", flat, [1, T, FEATURE_DIM]),
      });

      const tokens = new BigInt64Array(maxTokens).fill(BigInt(special.pad));
      tokens[0] = BigInt(special.bos);
      for (let step = 1; step < maxTokens; step++) {
        const { logits } = await decoder.run({
          memory,
          tokens: new ort.Tensor("int64", tokens, [1, maxTokens]),
        });
        let best = 0, bestVal = -Infinity;
        const base = (step - 1) * V;
        for (let c = 0; c < V; c++) {
          const val = logits.data[base + c];
          if (val > bestVal) { bestVal = val; best = c; }
        }
        if (best === special.eos) break;
        tokens[step] = BigInt(best);
      }
      return detokenize(tokens, vocab.tokens, special);
    },
  };
}

/**
 * ── Word tagger (stage 1 of word-list + LLM translation) ────────────────────
 *
 * Loads sanketvani_word_tagger.onnx and returns ordered content words, never a
 * sentence. Hand the result to an LLM to write the English.
 *
 *   const tagger = await loadWordTagger("./sanketvani_word_tagger.onnx",
 *                                       "./word_vocab.json");
 *   const words = await tagger.detect(collectedFrames);
 *   // [{word:"i", confidence:0.82, at:14}, {word:"live", ...}, {word:"delhi", ...}]
 *   const prompt = tagger.prompt(words);
 */
export async function loadWordTagger(onnxUrl, vocabUrl, ort = globalThis.ort) {
  const vocab = await (await fetch(vocabUrl)).json();
  if (vocab.input.feature_dim !== FEATURE_DIM) {
    throw new Error(`word_vocab.json expects ${vocab.input.feature_dim} features, ` +
                    `parity.js is built for ${FEATURE_DIM}`);
  }
  const session = await ort.InferenceSession.create(onnxUrl, {
    executionProviders: ["webgpu", "wasm"], graphOptimizationLevel: "all",
  });

  return {
    vocab, session,
    /**
     * @returns {{word:string, confidence:number, at:number}[]} in signed order
     */
    async detect(frames, T = vocab.input.num_frames ?? DEFAULT_FRAMES,
                 threshold = vocab.threshold, topK = vocab.top_k) {
      const flat = resampleTo(frames, T);
      if (vocab.input.body_normalise) bodyNormaliseFlat(flat);
      const { frame_logits } = await session.run({
        landmarks: new ort.Tensor("float32", flat, [1, T, FEATURE_DIM]),
      });
      const V = vocab.num_words;
      const data = frame_logits.data;

      // MIL pooling: a word's score is its strongest frame, and that frame is
      // where it was signed. Max over logits, not probabilities — sigmoid is
      // monotonic, so the argmax is the same and this stays cheap.
      const best = new Float32Array(V).fill(-Infinity);
      const peak = new Int32Array(V);
      for (let t = 0; t < T; t++) {
        const row = t * V;
        for (let w = 0; w < V; w++) {
          const v = data[row + w];
          if (v > best[w]) { best[w] = v; peak[w] = t; }
        }
      }

      const picked = [];
      for (let w = 0; w < V; w++) {
        const confidence = 1 / (1 + Math.exp(-best[w]));
        if (confidence >= threshold) {
          picked.push({ word: vocab.words[w], confidence, at: peak[w] });
        }
      }
      picked.sort((a, b) => b.confidence - a.confidence);   // keep the strongest
      const top = picked.slice(0, topK);
      top.sort((a, b) => a.at - b.at);                      // then signed order
      return top;
    },

    /** Fill word_vocab.json's template with the detected words. */
    prompt(detected) {
      const listing = detected
        .map((d) => `${d.word} (${d.confidence.toFixed(2)})`)
        .join(", ");
      // Older exports double-escaped the template, so it can arrive carrying a
      // literal backslash-n where a line break belongs. Repair it here rather
      // than shipping a prompt with "\\n" printed inside it.
      return vocab.llm_prompt_template
        .replace(/\\n/g, "\n")
        .replace("{words}", listing || "(none)");
    },
  };
}
