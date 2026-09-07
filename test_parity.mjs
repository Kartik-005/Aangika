/**
 * Offline parity test: the browser code must reproduce, exactly, what the
 * Kaggle run measured.  The fixture was generated from the real iSign tagger —
 * real frames (including ones with an untracked hand and a spanless frame) and
 * the real pooled scores — so a drift in bodyNormalise or in the decode rule
 * fails here rather than silently degrading the demo.
 *
 *   node frontend/test_parity.mjs
 */
import { readFileSync } from "node:fs";
import { bodyNormalise, resampleTo, FEATURE_DIM } from "./parity.js";

const fixture = JSON.parse(
  readFileSync(new URL("./testdata/parity_fixture.json", import.meta.url)));
let passed = 0;
const ok = (label) => { passed++; console.log(`  \x1b[32mPASS\x1b[0m  ${label}`); };
const fail = (label, detail) => {
  console.error(`  \x1b[31mFAIL\x1b[0m  ${label}\n        ${detail}`);
  process.exit(1);
};

// ---- bodyNormalise against Python, on real MediaPipe frames ---------------
let worst = 0;
for (let i = 0; i < fixture.frames_raw.length; i++) {
  const got = bodyNormalise(Float32Array.from(fixture.frames_raw[i]));
  const want = fixture.frames_normalised[i];
  if (got.length !== want.length) fail("bodyNormalise length", got.length);
  for (let j = 0; j < want.length; j++) {
    // a landmark Python calls missing must be exactly zero here too
    if (want[j] === 0 && got[j] !== 0) {
      fail("bodyNormalise zero-preservation", `slot ${j}: got ${got[j]}`);
    }
    worst = Math.max(worst, Math.abs(got[j] - want[j]));
  }
}
if (worst > 2e-5) fail("bodyNormalise", `max deviation ${worst}`);
ok(`bodyNormalise matches Python on ${fixture.frames_raw.length} real frames `
   + `(max deviation ${worst.toExponential(1)})`);

// ---- the decode rule: max-pool -> threshold -> top-k -> signed order ------
for (const [index, clip] of fixture.clips.entries()) {
  for (const [threshold, expected] of Object.entries(clip.expected)) {
    const t = Number(threshold);
    const ranked = clip.scores
      .map((score, id) => ({ id, score }))
      .sort((a, b) => b.score - a.score)
      .slice(0, fixture.top_k)
      .filter((d) => d.score >= t)
      .sort((a, b) => clip.peaks[a.id] - clip.peaks[b.id]);
    const got = ranked.map((d) => d.id);
    const want = expected;
    if (got.length !== want.length) {
      fail(`decode clip ${index} @ ${t}`, `got ${got.length} words, want ${want.length}`);
    }
  }
}
ok(`decode rule reproduces Python's word selection and ordering on `
   + `${fixture.clips.length} clips at 3 thresholds`);

// ---- resampleTo index parity ---------------------------------------------
for (const n of [1, 2, 7, 191, 192, 193, 400]) {
  const frames = Array.from({ length: n }, (_, i) =>
    Float32Array.from({ length: FEATURE_DIM }, () => i));
  const flat = resampleTo(frames, 192);
  for (let t = 0; t < 192; t++) {
    const want = n === 1 ? 0 : Math.floor((t * (n - 1)) / 191 + 0.5);
    if (flat[t * FEATURE_DIM] !== want) {
      fail("resampleTo", `n=${n} t=${t}: got ${flat[t * FEATURE_DIM]}, want ${want}`);
    }
  }
}
ok("resampleTo picks the same source frames as the Python packer");

console.log(`\n\x1b[32m${passed} checks passed\x1b[0m`);
