# 20 — Feature-extraction pipeline (waveform window → 104-feature model input)

**Scope of this doc:** the DSP stage only — turning a raw 1 kHz waveform into the exact
`[N, 104]` z-scored matrix the trained SNN consumes. Same code path for synthetic scenes
and live geophone signals. **PLAN ONLY — no code is to be written from this doc.** It records
the exact chain, a reusable module spec (signatures only), the streaming concern, dependencies,
and open questions.

All paths are absolute from project root:
`S:\ALL PROJECTS\geophone sensor\finals project\finals project`

Reference files:
- `simgeo\features.py` — the DSP extractor (`scene_precompute`, `window_features`, `FEATURE_NAMES`, `NFEAT`, `FS`, `NW`).
- `simgeo\extract_features.py` — offline corpus featurizer (the canonical batch usage).
- `snn_v2_out\scaler.json` — the 104 selected feature names + per-feature `mean`/`std` + `clip`.
- `rescale_test.py` — the real-data featurize loop (the canonical real-signal usage; the pattern to copy).

---

## 0. Where this sits relative to the webui (context)

The current webui (`webui\src\*.js`) is **pure browser JavaScript** with a synthetic mock signal
(`signal.js` `sample()` synthesizes a waveform; `detection.js` only renders lamps from a `w.detected`
array fed by `sim.js`). **There is no Python backend, no DSP, and no model inference wired in.** The
feature pipeline described here is Python (numpy/scipy/pywt) and cannot run in the browser as-is. So
wiring this up implies a **new Python service boundary** (out of scope for this doc; see `10_*`/`30_*`
siblings if they exist). This doc specifies the Python featurizer that service will call.

---

## 1. The exact chain (waveform → 104 z-scored features)

### 1.1 Constants (from `simgeo\features.py`, top of file)

| Constant | Value | Line | Meaning |
|---|---|---|---|
| `FS` | `1000.0` | features.py:30 | sample rate (Hz). Waveform MUST be 1 kHz. |
| `NW` | `3000` | features.py:31 | window length = 3.0 s = 3000 samples |
| `NFEAT` | `132` | features.py:144 | full feature-bank width (`len(FEATURE_NAMES)`, asserted `== 132` at features.py:145) |
| `NFFT_ENV` | `8192` | features.py:32 | envelope-spectrum zero-pad (0.122 Hz res) |
| `HOP` | **not in features.py** | — | `1500` samples (1.5 s, 50% overlap). Defined by the *caller*: `rescale_test.py:13` (`HOP = 1500`) and `extract_features.py` derives `i0` from label `t0` instead. |
| `SCENE` | **not in features.py** | — | `30 * int(FS) = 30000` (30 s segment) in `rescale_test.py:13`. This is the chunk over which `scene_precompute` is run for real data; it is NOT a model constraint, just the real-loop's segmentation. |

> Note the naming: `NFEAT == 132` is the **full bank**. The **model input is 104** (`len(scaler["features"])`).
> The 28-feature gap (132 − 104) is dropped by the selection step in 1.4. Verified: `scaler.json`
> has `len(features) == len(mean) == len(std) == 104`, `clip == 8.0`.

### 1.2 Input contract: `x`

- dtype: `float32` (or anything `np.asarray(x, np.float64)` accepts — `scene_precompute` upcasts to
  float64 internally at features.py:151).
- sample rate: **exactly 1000 Hz**. No resampling happens inside the pipeline. If the live geophone
  streams at another rate, resample **upstream** to 1000 Hz before this stage.
- **amplitude units = the same units the scaler was fit on.** Concretely, the corpus is built in
  **mV** (extract_features v2 reads `clean_mv + noise_mv`, extract_features.py:65-67). The real CSVs
  carry a raw `amplitude` column (e.g. `car.csv` header `time_s,amplitude`, values ~±0.06). The
  scaler's absolute-level features (`rms_total`, `variance`, `energy_*`, `peak_to_peak`, `log_rms`,
  `spectral_*` magnitudes) are only valid if `x` is in the **corpus mV scale**.
- **The ×25.4 (and the ×90 / ×1.0) real-scale factor is applied UPSTREAM of this pipeline**, not
  inside it. In `rescale_test.py:47` the scaling is `a = pd.read_csv(...)["amplitude"] * scale`
  BEFORE `scene_precompute`. So `featurize_segment` must receive **already-scaled** `x`. This doc's
  module does NOT multiply by 25.4 — the caller decides the scale. (See risks §5 — this is the single
  biggest correctness hazard for live data.)

### 1.3 The DSP chain (per segment)

```
x  (float32, 1000 Hz, mV-scale, len >= NW)
   │
   ├─ pre = F.scene_precompute(x)            # features.py:149
   │      → dict of band-filtered signals (8 LEGACY sos + 7 PHYS sos via sosfilt),
   │        broadband env (|x| lowpassed 20 Hz), engine env, rain env,
   │        impulse pick indices (imp_t), rain pick indices (rain_t).
   │        Cost is O(len(x)) per filter; computed ONCE over the whole segment.
   │
   └─ for i0 in range(0, len(x) - NW + 1, HOP):   # HOP=1500, NW=3000
          vec132 = F.window_features(pre, i0)      # features.py:197 → shape (132,) float64
          # window slices pre['x'][i0:i0+NW] and the prefiltered bands at the same offset
```

Each `window_features` call returns a length-`NFEAT (132)` `float64` vector (asserted `j == NFEAT`
at features.py:438). Stack windows → `[N, 132]`.

### 1.4 Selection: 132 → 104 (column reindex by name)

The model uses a **name-selected, reordered** subset. From `rescale_test.py:14-16`:

```python
FEATURES = scaler["features"]                       # the 104 names, in MODEL order
fidx = [F.FEATURE_NAMES.index(f) for f in FEATURES] # 104 indices into the 132-vec
X104 = X132[:, fidx]                                 # reorder + drop 28 unused columns
```

- `fidx` is computed once. It is **order-sensitive**: the model was trained on
  `scaler["features"]` order (NOT `FEATURE_NAMES` order). Do not sort. Do not assume identity mapping.
- `FEATURE_NAMES.index` raises `ValueError` if a scaler name is missing → good fail-fast guard;
  worth asserting `len(fidx) == 104` at module load.

### 1.5 Z-score + clip (the final model input)

From `rescale_test.py:61`:

```python
mu  = np.array(scaler["mean"], np.float32)   # (104,)
sd  = np.array(scaler["std"],  np.float32)   # (104,)
CLIPZ = scaler["clip"]                        # 8.0
Xz = np.clip((X104 - mu) / sd, -CLIPZ, CLIPZ).astype(np.float32)   # → [N, 104]
```

NaN/Inf handling (do this BEFORE clip; rescale_test does it on the raw 132-stack):
```python
X132 = np.nan_to_num(np.stack(fe))   # rescale_test.py:53 — nan/posinf/neginf → 0.0
```
`extract_features.py:107` does the same on its stacked output
(`np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)`).

**Full one-line summary of the chain:**
`x (mV, 1 kHz) → scene_precompute → window_features×N → stack[N,132] → nan_to_num → [:, fidx]→[N,104] → clip((x−mean)/std, ±8) → float32 [N,104]`.

---

## 2. Reusable module spec (signatures only — DO NOT IMPLEMENT HERE)

Proposed new module (suggested path) `simgeo\featurize.py` **or** `webui\backend\featurize.py`
(depends on where the service boundary lands — see §0; pick one in the integration doc). It wraps
`simgeo\features.py` so both the synthetic-scene path and the live path call identical code.

```python
# Load-once module state (mirrors rescale_test.py:14-16)
#   _SCALER  = json.load(snn_v2_out/scaler.json)
#   _NAMES   = _SCALER["features"]                 # 104 names, model order
#   _FIDX    = [F.FEATURE_NAMES.index(n) for n in _NAMES]   # 104 ints
#   _MU, _SD = np.asarray(_SCALER["mean"], f32), np.asarray(_SCALER["std"], f32)
#   _CLIP    = float(_SCALER["clip"])              # 8.0
#   assert len(_FIDX) == len(_MU) == len(_SD) == 104

def featurize_window(window3000) -> np.ndarray:    # in: (3000,) float, 1 kHz, mV-scale
    """Single 3 s window → (104,) float32 z-scored, ready for the SNN.
       Internally runs scene_precompute on the 3000-sample window and takes i0=0.
       (See §3 streaming note: this is the per-window-cost variant.)"""
    # returns shape (104,)

def featurize_segment(waveform, hop=1500) -> np.ndarray:  # in: (M,) float, M >= 3000
    """Arbitrary-length waveform → (N, 104) float32 z-scored.
       scene_precompute ONCE over the whole waveform, then window_features at
       i0 = 0, hop, 2*hop, ... while i0+NW <= M.  N = 1 + (M - NW)//hop.
       Applies nan_to_num → select fidx → (x-mu)/sd → clip(±_CLIP)."""
    # returns shape (N, 104)
```

Notes:
- `featurize_window` is `featurize_segment` with a length-3000 input and `hop` irrelevant (one window).
  Implementations should funnel both through one private `_zscore_select(X132)->X104` helper so the
  selection/scaling math lives in exactly one place.
- Return **float32** (model expects it; `rescale_test.py:61` casts to float32; torch tensor built from
  it at rescale_test.py:42).
- Keep the raw 132-vector available internally only if a debug/inspection mode is wanted; the public
  return is the 104 z-scored matrix.
- These are the SAME functions for synthetic and live. The synthetic-scene producer just needs to
  hand over a 1 kHz mV-scale `x`; the live producer hands over the rolling buffer (§3).

---

## 3. Streaming concern (LIVE: rolling 3 s window every 1.5 s)

### 3.1 The structural issue

`scene_precompute` (features.py:149-168) is designed for **batch**: it filters the ENTIRE segment
once, then many windows subselect prefiltered slices and scene-wide impulse picks. For a live
stream you receive samples continuously and want one feature vector per new 1.5 s hop over a rolling
3 s window. Two options:

**Option A — recompute `scene_precompute` on the rolling 3 s buffer every hop (RECOMMENDED for v1).**
- Keep a rolling buffer of the last `NW = 3000` samples (3 s). Every `HOP = 1500` samples (1.5 s),
  call `featurize_window(buffer)` → one `(104,)` vector.
- This is exactly what `featurize_window` above does, and it is self-contained/correct.
- **Per-window cost** (N = 3000): `scene_precompute` runs 15 `sosfilt` passes (8 LEGACY + 7 PHYS
  bands), 3 envelope lowpass passes, 2 `find_peaks`, all O(N) → ~20·N ≈ 6e4 mult-adds for filters.
  `window_features` adds: one rfft of 3000, one `rfft(env, NFFT_ENV=8192)`, a `medfilt(psd, 51)`,
  3 sub-window rffts of 1000, a `db4` level-4 wavelet packet, `find_peaks`, Higuchi (kmax=8, O(N·k)),
  Levinson(8). Dominant terms: rffts (O(N log N), N∈{3000,8192}) and `medfilt(psd,51)` (O(len(psd)·51)
  ≈ 1501·51 ≈ 7.7e4). **Rough total ≈ low-hundreds-of-thousands of flops per window → sub-millisecond
  to a few ms in numpy/scipy.** At one window / 1.5 s this is *trivially* real-time (>>100× headroom).
- **Conclusion:** recomputing every hop is the right call. The redundancy (filtering the 1.5 s of
  overlap twice) is negligible at this cadence.

**Option B — incremental / stateful filtering.** Maintain `sosfilt` filter state (`zi`) across hops
so only the new 1500 samples are filtered, plus a running envelope and an impulse-pick ring buffer.
- Pros: ~2× less filter work; avoids edge transients at the buffer boundary that Option A re-incurs
  each window.
- Cons: **significant** rewrite of `scene_precompute` (it has no streaming API today), and several
  `window_features` quantities are **whole-window, not incremental** (rffts, wavelet packet, medfilt,
  `corrcoef` over sub-windows). You'd still recompute those per window. So Option B only saves the
  filtering portion — a small fraction of total cost — for a large complexity/maintenance hit and a
  new validation burden (must prove it matches batch outputs).
- **Recommendation: do NOT do Option B for v1.** The per-window cost (§above) makes it unnecessary.
  Revisit only if profiling on the target device shows the filter passes dominate (unlikely at 1.5 s
  cadence).

### 3.2 Edge-transient caveat (applies to BOTH options vs the corpus)

The corpus (`extract_features.py`) and `rescale_test.py` run `scene_precompute` over a LONG segment
(whole scene / 30 s) and then slice 3 s windows out of the middle — so each window's prefiltered
band signals have **settled** (no filter start-up transient). A live 3 s buffer fed to
`scene_precompute` starts the IIR filters from zero state → the first ~tens of ms of each band signal
carries a transient. This is a **train/serve skew**: the model never saw windows whose bands start
cold. Magnitude is likely small (4th-order Butterworth settles fast relative to 3000 samples) but it
is **unvalidated**. Mitigation if it matters: feed a slightly longer buffer (e.g. 4–5 s) to
`scene_precompute` and take the last 3 s window (`i0 = len - NW`), discarding the warm-up region —
this mirrors the "slice from the middle" batch behavior. Flag as open question §5.

---

## 4. Dependencies & gotchas

### 4.1 Python dependencies (already used by `features.py` / `rescale_test.py`)
- `numpy` (core; fft, stack, nan_to_num, clip).
- `scipy` — `scipy.signal` (`butter`, `sosfilt`, `find_peaks`, `medfilt`), `scipy.stats`
  (`kurtosis`, `skew`), `scipy.fft.dct`. (features.py:25-27)
- `pywt` (PyWavelets) — `WaveletPacket` db4 level-4. (features.py:28) **Easy-to-miss dep.**
- `pandas` — only in `rescale_test.py:47` to read the CSV `amplitude` column; NOT needed by the
  featurizer itself if the service passes a numpy array. Keep it out of the core module.
- (model side, not this stage): `torch`, `spikingjelly`, `sklearn` — belong to the inference doc.

### 4.2 Gotchas
1. **`std[78] == Infinity` for `variance`** (confirmed in scaler.json). `(x − mean)/Inf → 0.0` for
   that one column, i.e. `variance` is **deliberately zeroed** in the model input. `np.clip` of `0`
   is fine. BUT: ensure the division happens with `sd` as float (it does: `np.array(..., np.float32)`
   yields `inf`, `x/inf → 0`, no warning beyond a possible divide-by-inf which numpy treats as 0, not
   nan). Do NOT `nan_to_num` the `sd` array (that would turn `inf` into a huge finite number and
   un-zero the feature). Order matters: `nan_to_num` is applied to the **raw 132-stack** (rescale_test.py:53),
   then select, then divide — never to `sd`.
2. **NaN/Inf in raw features.** Short/quiet windows can produce NaN (e.g. `corrcoef` on flat signal,
   `skew`/`kurtosis` on constant input, log of zero). `np.nan_to_num(stack)` (→0.0) is mandatory and
   is the established behavior in both batch and real loops. Apply on the 132-stack before scaling.
3. **Amplitude scale (the big one).** See §1.2. The scaler's absolute-level features assume corpus
   mV scale. Live geophone counts/volts must be converted to that scale upstream. `rescale_test.py`
   exists *specifically* to measure this (`SCALES = [1.0, 25.4, 90.0]`, rescale_test.py:17). The
   ×25.4 "floor-aligned" factor is the leading candidate but is **not settled** — it is a research
   variable, not a fixed constant. The featurizer must NOT bake in any scale; the caller passes
   already-scaled `x`. (Project is being prepped for publication — do not hardcode an unvalidated
   scale factor into the live path.)
4. **Window length guard.** `featurize_segment` must skip/refuse segments with `len < NW` (3000).
   `rescale_test.py:50` does `if len(seg) < F.NW: continue`. For live, do not emit a vector until the
   rolling buffer has ≥3000 samples.
5. **`i0` integer alignment.** Batch derives `i0 = int(round(t0 * FS))` from label times
   (extract_features.py:78); the real loop uses `range(0, len-NW+1, HOP)`. For live, hops should be
   exact integer sample counts (1500) to avoid drift; don't accumulate float seconds.
6. **`HOP`/`SCENE` are caller-side, not in `features.py`.** Anyone re-implementing must define them;
   they are not importable. Recommend defining them once in the new module to avoid divergence
   (`HOP = 1500`, and for live there is no `SCENE` — the buffer is the window).
7. **float64 internal, float32 out.** `window_features` returns float64; cast to float32 only at the
   end (matches training).
8. **Determinism.** No randomness in the pipeline — same `x` → same features. Good for testing:
   a golden-vector test (feed a fixed CSV window, compare 104-vec to a saved reference) is cheap and
   catches selection/scaling regressions.

---

## 5. Open questions / risks

1. **[HIGH] Live amplitude scale.** What scale factor maps the live geophone's raw amplitude to the
   corpus mV scale the scaler expects? `rescale_test.py` brackets it (1.0 / 25.4 / 90) but does not
   fix it. Wiring live data without resolving this will silently corrupt every absolute-level feature
   (~half the 104). **Must be decided & validated before the live path ships.** Owner: upstream
   acquisition/calibration, not this module.
2. **[MED] Edge-transient train/serve skew (§3.2).** Batch windows are sliced from settled filter
   output; a fresh 3 s live buffer has cold-start IIR transients. Unvalidated. Decide whether to
   pad the buffer (feed 4–5 s, take last 3 s) to match batch behavior. Needs an A/B feature-diff test.
3. **[MED] `variance` feature is zeroed (`std=Inf`).** Confirm this is intentional in the trained
   scaler and not an artifact of a degenerate fit. If unintentional, the model effectively never uses
   `variance`; if intentional, fine — but document it so no one "fixes" the Inf and shifts the input
   distribution.
4. **[LOW] `scene_precompute` segment length for synthetic scenes.** Synthetic producer should feed
   ≥3 s (ideally full scene, like the corpus) so windows are settled and impulse picks (`imp_t`,
   `rain_t`, which are scene-wide) are statistically comparable to training. Per-window
   `scene_precompute` (Option A) changes the impulse-pick context (median over 3 s, not 30 s) — this
   is a **subtle distribution shift for the impulse/cadence features** (`imp_t`-derived: `iii_*`,
   `impulse_density`, `mod_*`, `attack_sharpness`). Worth a feature-diff check: same window featurized
   via (a) precompute-over-30s-then-slice vs (b) precompute-over-3s. If they diverge materially, the
   live path's per-window precompute is itself a train/serve skew, independent of the edge transient.
   **This is arguably the most important validation item** — it questions whether Option A's
   per-window precompute is even distribution-faithful for the scene-wide features.
5. **[LOW] Service boundary / language.** Featurizer is Python; webui is JS. The transport (HTTP/WS
   sending raw waveform vs sending features) is out of scope here but constrains who calls
   `featurize_*`. Decide in the integration doc (`10_*`/`30_*`).
6. **[LOW] scaler/model version pinning.** `featurize.py` hardcodes a path to
   `snn_v2_out\scaler.json`. If the model is retrained, scaler `features`/`mean`/`std` change and
   `fidx` must be recomputed. Load at startup, assert `len==104`, and log the scaler hash/mtime so a
   stale scaler is detectable.
