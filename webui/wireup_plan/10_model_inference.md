# Wire-up Plan 10 — Loading & Running the Fine-Tuned SNN for Per-Window Inference

Sub-area: **the inference engine.** Given a batch of 104-dim feature vectors (one per
analysis window), produce per-head presence probabilities and decode them into the UI's
4 classes (`human`, `car`/vehicle, `animal`, `nothing`). The UI today reads these from
`webui/src/sim.js` (`Sim.sampleWindow()`); this plan describes the model that must replace it.

> Scope guard: this document is **plan only**. It defines the artifact to produce, the
> exact inference recipe, the decode rule, and a module contract. It does **not** cover
> featurization (the 104-feature extraction from raw `amplitude`), serving/transport
> (HTTP vs WASM vs child process), or the front-end glue — those are sibling plans.

---

## 0. TL;DR — the artifact decision

**There is NO deployable fine-tuned checkpoint on disk today.** The headline real-data
result (`0.977` 5-fold acc, `snn_v2_out/FINETUNE_5FOLD.json`) is produced *transiently
inside a cross-validation loop* and **never saved**. The only `.pt` files that exist are
the **synthetic-pretrained** model and its variants:

| file | what it is | saved by | deployable as "fine-tuned"? |
|---|---|---|---|
| `snn_v2_out/model_ema.pt` | synthetic-pretrained EMA weights (the starting point for FT) | `geophone_snn_v2_train.py:232` | **No** — this is the *pre*-finetune model |
| `snn_v2_out/model_raw.pt` | synthetic-pretrained non-EMA weights | `geophone_snn_v2_train.py:233` | No |
| `snn_v2_out/model_focal_ema.pt` | focal-loss training variant (ablation) | (focal experiment) | No |
| `snn_v2_out/model_cdf_ema.pt` | CDF-scaler training variant (ablation) | (cdf experiment) | No |
| `snn_v2_out/FINETUNE_5FOLD.json` | **CV metrics only** — no weights | `finetune_5fold.py:108` | metrics, not a model |

**Decision (required before any wiring): we must PRODUCE a single deployable fine-tuned
checkpoint.** Recipe in §1. The user asked for the model "after fine-tuning", so shipping
`model_ema.pt` as-is is **wrong** — that is the zero-shot model (5-fold acc `0.859`, see
`FINETUNE_5FOLD.json:2`). It must be fine-tuned on the real CSVs first.

---

## 1. The fine-tuned-model artifact problem (and how to fix it)

### 1.1 Why no checkpoint exists

`finetune_5fold.py` is a **measurement script**, not a training/export script. Walking it:

- Lines 86-101: for each of `K=5` contiguous folds it fits a fresh per-fold
  `StandardScaler` on the fold's train split (line 88), loads `model_ema.pt` as the base
  (line 90), then trains a **5-member deep ensemble** of fine-tuned copies
  (`ENS=5`, lines 94-98), averages their presence probabilities, picks thresholds on the
  train split, and scores the held-out fold.
- The fine-tuned `model` objects (`finetune(copy.deepcopy(base), ...)`, line 96) are
  **local variables**. Nothing calls `torch.save`. Only `FINETUNE_5FOLD.json` (scalars)
  is written (line 108).
- So the reported `0.977` is the mean over **25 transient models** (5 folds × 5 ensemble
  members), each fitted to a *different* 4/5 subset of the real data with a *fold-local*
  scaler. **No single one of those 25 models is a deployable artifact**, and they used
  per-fold scalers that don't exist at deploy time.

**Two further blockers for direct reuse of the CV protocol:**

1. **Scaler mismatch.** CV uses a per-fold real `StandardScaler` (line 88). For
   deployment we need ONE fixed scaler. Options: (a) refit a real scaler on ALL real data
   and ship it; (b) keep the synthetic `scaler.json` (mean/std/clip from synthetic-train,
   `geophone_snn_v2_train.py:73-76`). The 5-fold result used (a)-style per-fold real
   scalers, so **to reproduce that accuracy we must ship a real-fitted scaler**, not the
   synthetic one. See open question Q3.
2. **Animal head is never fine-tuned.** `finetune_5fold.py` supervises only `yh`
   (human) and `yv` (vehicle) — see the loss at lines 78-79 and `probs()` at lines 60-62,
   which only read `human` and `vehicle`. There are **no real animal recordings**
   (`REALS` at line 37 = car/human/nothing only). So after fine-tuning, the **animal head
   carries only its synthetic-pretrained weights** and is unvalidated on real data. This
   is a real risk for the UI's `animal` lamp — see §3 and Q1.

### 1.2 Recipe to PRODUCE the deployable checkpoint

**Chosen approach: fine-tune the synthetic-pretrained EMA model on ALL real data once,
then save one `.pt` + one matching scaler.** This is the simplest deployable artifact and
matches the per-member training that produced the headline number; it drops the 5-fold CV
(CV is for *measuring* generalization, not for *producing* a shipping model). A deep
ensemble (persist all 5 members and average at inference) is the higher-accuracy
alternative — see §1.3 and Q2.

Concretely, write a new export script (e.g. `export_finetuned.py`) that **reuses
`finetune_5fold.py` verbatim except**: train on `tr = all real windows` instead of a fold,
fit the scaler on all real data, and `torch.save` at the end. Exact steps, referencing
`finetune_5fold.py`:

1. **Featurize real CSVs** exactly as lines 37-51: `SCALE=25.4` amplitude scale (line 12),
   `SCENE=30*FS` chunking, `HOP=1500`, `F.window_features`, select the 104 `fidx`
   (line 16). Result: `X [N,104] float32`, `y [N] str`, with
   `yh=(y=='human')`, `yv=(y=='vehicle')` (line 52). (N ≈ 2353, cf. `ERROR_ANALYSIS.json:9`.)
2. **Fit ONE deployment scaler on ALL real X:**
   `sc = StandardScaler().fit(X)` then `Xz = clip(sc.transform(X), -8, 8)` (mirrors
   line 88 but on the full set, `CLIPZ=8.0` from `scaler.json`). Persist it as
   `scaler_real.json` in the same schema as `scaler.json`
   (`{mean, std, clip, features}`) so inference is scaler-agnostic.
3. **Build base & load pretrained EMA** (mirror lines 89-90):
   ```
   base = FeatureSNN(104)                       # widths (512,256,128), dropout 0.1
   functional.set_step_mode(base, 'm')
   base.load_state_dict(torch.load('snn_v2_out/model_ema.pt', map_location=DEV))
   ```
4. **Fine-tune on the full real set** using the existing `finetune()` (lines 68-81) with
   `tr = np.ones(N, bool)` (all data), `seed=0`, `epochs=80`, `lr=4e-4`, cosine schedule,
   `Adam(wd=1e-4)`, batch 256, BCE-with-logits on human+vehicle heads. **No held-out set**
   — this is a final-fit, not a measurement.
5. **Pick deployment thresholds.** Either (a) reuse the per-head thresholds already in
   `ERROR_ANALYSIS.json` (`human 0.15, vehicle 0.05, animal 0.1`, lines 3-7) — these were
   calibrated for the **zero-shot** model so they may be miscalibrated for the fine-tuned
   one (Q4); or (b) recompute via `thr()` (lines 63-64, grid 0.05..0.95, maximize balanced
   accuracy) on the full real set and write them into the new artifact's sidecar. Prefer
   (b) and save alongside the checkpoint.
6. **Save the artifact:**
   ```
   torch.save(base.state_dict(), 'snn_v2_out/model_finetuned.pt')
   json.dump({...}, open('snn_v2_out/finetuned_meta.json','w'))   # thresholds, scaler ref, provenance
   ```
   The `state_dict` is architecture-identical to `model_ema.pt` (~900 KB), so all existing
   load code works unchanged.

**Compute cost (CPU, for the user's N≈2353, 80 epochs, batch 256 → ~10 steps/epoch ×
80 = ~800 forward+backward steps over T=4):** O(epochs × N × Σ widths) per head.
Σ widths ≈ 104·512 + 512·256 + 256·128 ≈ 215k MACs/window/step × 4 timesteps. This is
seconds-to-a-couple-minutes on CPU for a one-time export; GPU not required (cupy NOT
required for the *forward* pass on the readout LIF — see §2.3). Run it once, commit the
`.pt` + scaler + meta.

### 1.3 Alternative: persist the 5-member ensemble

If we want to reproduce the **exact** `0.977` deep-ensemble behavior, persist all 5
fine-tuned members (seeds 0-4) trained on ALL real data, and at inference **average the
sigmoid presence probabilities across members** before thresholding (mirrors lines 95-98).
Cost: 5× the inference latency and 5× checkpoint size (~4.5 MB total). For a UI doing one
window every ~2 s this is negligible. Decision deferred to Q2; the single-model path (§1.2)
is the default unless the accuracy delta matters for the publication.

---

## 2. Inference path (exact)

The canonical clean inference example is `rescale_test.py:35-43` — follow it verbatim. Note
it uses the **last-step membrane** decode (`v_seq[-1]`), which differs from training's
**mean-over-T** decode; see §2.2.

### 2.1 Model construction & load (one-time, at startup)

```
net = FeatureSNN(104)                          # rescale_test.py:35
functional.set_step_mode(net, 'm')             # multi-step mode for all layers
net.load_state_dict(torch.load(CKPT, map_location='cpu'))   # CKPT = model_finetuned.pt
net.eval()
```

`FeatureSNN.__init__` (`rescale_test.py:22-31`, identical to
`geophone_snn_v2_train.py:108-125`): a learned `gate` (104,), a body of 3 ×
`[Linear → SeqBN → ParametricLIFNode(init_tau=2.0, ATan(2.0), detach_reset) → Dropout(0.1)]`
with widths (512,256,128), then 3 heads `human=Linear(128,2)`, `animal=Linear(128,2)`,
`vehicle=Linear(128,1)`, each followed by a non-spiking readout
`LIFNode(v_threshold=inf, store_v_seq=True, backend='torch')`. Params ≈ 220,528
(`summary.json:2`). `T=4` constant-current encoding (input repeated over 4 timesteps,
`forward` line 33).

**Do NOT** call `functional.set_backend(net, 'cupy', ...)` (that line exists only in
`geophone_snn_v2_train.py:137` for *training* the PLIF layers). For inference the default
`torch` backend is correct and CPU-compatible.

### 2.2 Per-window forward (per batch of windows)

Follow `rescale_test.py:38-43` exactly:

```
@torch.no_grad()
def forward_probs(net, Xz):                     # Xz: [B,104] float32, ALREADY scaled+clipped
    out = {'human': [], 'animal': [], 'vehicle': []}
    for j in range(0, len(Xz), BATCH):          # BATCH e.g. 8192 (rescale_test uses 8192)
        functional.reset_net(net)               # MANDATORY before every forward — clears membrane state
        net(torch.as_tensor(Xz[j:j+BATCH]))     # runs body + all 3 heads, stores v_seq
        for nm in out:
            out[nm].append(torch.sigmoid(getattr(net, nm)[-1].v_seq[-1][:, 0]).cpu())
    return {k: np.array(torch.cat(v)) for k, v in out.items()}
```

Key details (load-bearing):

- **`functional.reset_net(net)` before EVERY forward call.** SpikingJelly neurons hold
  membrane state across calls; skipping reset corrupts results. (`rescale_test.py:41`,
  `finetune_5fold.py:60`, `geophone_snn_v2_train.py:181`.)
- **Head readout extraction:** `getattr(net, nm)` is the head `nn.Sequential`; `[-1]` is
  its `LIFNode`; `.v_seq` is `[T, B, out]`; `v_seq[-1]` is the **last timestep**
  membrane `[B, out]`; `[:, 0]` is the presence logit (column 0). `sigmoid` → presence
  probability. (`rescale_test.py:42`.)
- **Decode discrepancy to flag (Q5):** training and `geophone_snn_v2_train.py` use
  `v_seq.mean(0)` (mean over T, line 33/132), while `rescale_test.py` /
  `ERROR_ANALYSIS.json` use `v_seq[-1]` (last step). The `0.879` number in
  `ERROR_ANALYSIS.json` is "EMA + last-step", and the per-head thresholds there are
  calibrated for **last-step**. **Inference MUST use the same decode the thresholds were
  calibrated on.** If we recompute thresholds in §1.2 step 5, compute them with the same
  decode we ship. Recommend **last-step** (`v_seq[-1]`) to match `rescale_test.py` and
  `ERROR_ANALYSIS.json`. Do not mix.
- For a single window, `B=1`; the loop still works.

### 2.3 CPU-only & latency

- **CPU inference works and cupy is NOT required.** cupy is only used to accelerate the
  PLIF layers during *training* (`geophone_snn_v2_train.py:137-138`). The readout LIF is
  hard-coded `backend='torch'` (line 124 / `rescale_test.py:30`), and the PLIF default is
  `torch`. `rescale_test.py` and `finetune_5fold.py` both fall back to CPU when no CUDA
  (`DEV = 'cuda' if torch.cuda.is_available() else 'cpu'`).
- **Latency, single window (`B=1`, T=4), CPU:** O(T × Σ widths) ≈ 4 × 215k MACs ≈ <1 ms of
  matmul; SpikingJelly per-step Python overhead dominates. Realistically **a few ms per
  window** on CPU — far under the UI's ~2 s window cadence. The 5-member ensemble (§1.3)
  is ~5× that, still trivial. No GPU needed for serving.
- **One numpy caveat:** SpikingJelly's cupy import path references removed numpy aliases
  (`np.int`/`np.float`/`np.bool`); `geophone_snn_v2_train.py:14-15` shims them. On a
  pure-CPU torch-backend path this shim is typically not hit, but include it defensively
  if importing `spikingjelly.activation_based` raises on numpy ≥1.24 (Q6).

---

## 3. Head decode → UI's 4 classes

The model has **3 presence heads** (`human`, `animal`, `vehicle`). The UI
(`webui/src/sim.js`, `webui/src/detection.js`) speaks **4 classes**: `human`, `car`,
`animal`, `nothing`, where the UI's `car` == the model's `vehicle`.

### 3.1 Probabilities → presence

Per head: `p = sigmoid(v_seq[-1][:,0])` (column 0 = presence). The model also has a
column-1 logit on human/animal (ordinal "single vs multiple", `head(2)`); the UI only
needs presence, so **column 0 only**. (Vehicle is `head(1)`, presence-only.)

### 3.2 Per-head thresholds (the "nothing" rule)

From `snn_v2_out/ERROR_ANALYSIS.json:3-7` (or recomputed per §1.2 step 5):

```
TH = { human: 0.15, vehicle: 0.05, animal: 0.10 }
```

The decode rule used in the validated pipeline is `rescale_test.py:63-64` (and identical
in `finetune_5fold.py:65-66`), but note it is a **2-class argmax** (human vs vehicle) — it
does **not** include the animal head:

```
mh = p_human   - TH.human
mv = p_vehicle - TH.vehicle
pred = (mh < 0 && mv < 0) ? 'nothing'
     : (mh >= mv)         ? 'human'
     :                      'vehicle'
```

`nothing` ⇔ **all relevant heads below threshold**. This is why "nothing" is implicit, not
a 4th head.

### 3.3 Mapping to the UI and the animal problem

The UI wants 4 lamps (`human`/`car`/`animal`) plus an all-clear. The validated decode
(§3.2) only resolves **human vs vehicle vs nothing** — **animal is not part of any
real-data-validated decision** (no real animal data; animal head never fine-tuned, §1.1).
Two options:

- **Option A (faithful, recommended for the article):** drive only `human`, `car`, and
  `nothing` from the decode in §3.2; leave the `animal` lamp permanently off (or hidden)
  in `geo` mode, and surface a note that animal detection is synthetic-only/unvalidated.
  Honest; avoids shipping an unvalidated detector.
- **Option B (3-head independent thresholding):** treat each head independently:
  `human on ⇔ p_human≥TH.human`, `car on ⇔ p_vehicle≥TH.vehicle`,
  `animal on ⇔ p_animal≥TH.animal`; `nothing ⇔ none on`. This populates the animal lamp
  but with an **uncalibrated, real-unvalidated** animal head — expect false animal lamps
  given footstep human↔animal overlap. Only acceptable with a clear "experimental" label.

The UI's downstream contract (what `sim.js:sampleWindow()` returns) is
`{ win, probs:{human,car,animal,nothing}, detected:[...], snr, kind, r }`
(`webui/src/sim.js:122`); `detection.js` reads `w.detected` and `w.probs[primary]`. The
inference module's output must be adapted to that shape by the serving glue (sibling plan).
Map model `vehicle` → UI `car`. For Option A, set `probs.animal = 0` (or the raw
`p_animal` for display but never thresholded into `detected`).

---

## 4. Clean module spec (signatures only — NO implementation)

A small Python module, e.g. `inference.py`, with this contract:

```
load_model(path: str) -> net
    # path: path to model_finetuned.pt (state_dict, arch-identical to model_ema.pt)
    # builds FeatureSNN(104), set_step_mode('m'), load_state_dict, eval(); returns net.
    # NOTE: returns the torch module only. Scaler + thresholds are loaded separately
    #       (see load_scaler / thresholds) so this stays a pure model loader.

load_scaler(path: str) -> {mean: np.ndarray[104], std: np.ndarray[104], clip: float, features: list[str]}
    # reads scaler_real.json (or scaler.json). Used to z-score+clip raw 104-feature vectors
    # BEFORE predict(). Featurization itself is out of scope (sibling plan).

predict(net, features104_batch: np.ndarray[B,104]  # ALREADY scaled+clipped to [-clip,clip]
        ) -> {human: np.ndarray[B], animal: np.ndarray[B], vehicle: np.ndarray[B]}
    # per-head presence probabilities in [0,1]. Internally: batched loop with
    # functional.reset_net + forward + sigmoid(head[-1].v_seq[-1][:,0]). torch.no_grad.
    # (If ensemble per §1.3: averages sigmoid probs across the 5 members.)

decide(probs: {human,animal,vehicle}, thresholds: {human,vehicle,animal}
       ) -> {label: 'human'|'vehicle'|'nothing'(|'animal'), per_class_on: {...bool}}
    # applies §3.2 (Option A) or §3.3-B (Option B). Returns UI-facing decision.
    # The serving glue renames 'vehicle' -> 'car' for the UI.
```

Open contract question: whether `predict` should accept **raw** 104-features and scale
internally (carry the scaler inside `net`) or require pre-scaled input (as written). The
validated scripts scale *outside* the model, so the spec above keeps that boundary.

---

## 5. Dependencies, risks, open questions

### Dependencies
- `torch` (CPU build sufficient), `spikingjelly` (`activation_based`: neuron, surrogate,
  layer, functional), `numpy`, `pandas` (featurization), `scikit-learn`
  (`StandardScaler` for the deployment scaler; `balanced_accuracy_score` if recomputing
  thresholds). `simgeo/features.py` is needed for featurization (sibling plan), not for the
  pure inference module.
- Artifacts that MUST be produced before wiring: `model_finetuned.pt`, `scaler_real.json`
  (or a decision to reuse `scaler.json`), and a thresholds sidecar.
- numpy ≥1.24 alias shim may be needed at import (`geophone_snn_v2_train.py:14-15`).

### Risks
- **R1 — No fine-tuned artifact exists.** Headline `0.977` is unreproducible as a single
  shippable model without running the §1.2 export. Highest-priority blocker.
- **R2 — Animal head is unvalidated on real data** (never fine-tuned; no real animal
  recordings). Shipping an animal lamp risks false positives/negatives the paper can't
  defend. See Q1.
- **R3 — Scaler/domain mismatch.** The real recordings are a *different domain* than
  synthetic (`summary.json:27` `domain_classifier_AUC = 1.0` — synthetic vs real are
  perfectly separable). The model only works on real data if (a) inputs are featurized at
  `SCALE=25.4` and (b) scaled with a real-fitted scaler. Using the synthetic `scaler.json`
  on real inputs is a known degradation path. Q3.
- **R4 — Decode/threshold coupling.** Thresholds in `ERROR_ANALYSIS.json` are for the
  **zero-shot last-step** model. After fine-tuning, probability calibration shifts;
  reusing those thresholds may misfire. Recompute on the fine-tuned model. Q4/Q5.
- **R5 — Confound caveat (publication).** Real data is 1 session/class; the 5-fold number
  is session-confounded/optimistic (`real_baseline.py` docstring lines 4-8;
  `FINETUNE_5FOLD.json` `confound_clean_hgb = 0.937`). The UI must not present `0.977` as a
  field-accuracy claim. This is a reviewer hot-button per project memory.
- **R6 — Per-step Python overhead** in SpikingJelly torch-backend on CPU; fine for ~2 s
  cadence, but batch windows where possible.

### Open questions
- **Q1 (animal lamp):** Option A (drop animal, faithful) vs Option B (independent
  thresholding, experimental)? Affects §3.3 and the module's `decide`. **Recommend A**
  for the published article; expose B behind an "experimental" flag if desired.
- **Q2 (single model vs 5-ensemble):** ship one fine-tuned `.pt` (§1.2) or persist+average
  5 members (§1.3) to recover the exact `0.977`? Need the accuracy delta between
  single-member and ensemble — not currently measured (the JSON only reports the ensemble).
- **Q3 (which scaler):** fit + ship a real-data `StandardScaler` (matches the validated
  CV protocol, recommended) vs reuse synthetic `scaler.json` (zero new artifacts, but
  domain-mismatched and NOT what produced `0.977`)?
- **Q4 (which thresholds):** reuse `ERROR_ANALYSIS.json` (`0.15/0.05/0.10`, zero-shot) vs
  recompute on the fine-tuned model? Recommend recompute and ship in the sidecar.
- **Q5 (decode: mean-over-T vs last-step):** training uses `v_seq.mean(0)`;
  `rescale_test.py`/`ERROR_ANALYSIS.json` use `v_seq[-1]`. Pick ONE and calibrate
  thresholds to it. Recommend **last-step** to match the validated real-test path.
- **Q6 (numpy/cupy import):** confirm `import spikingjelly.activation_based` succeeds on
  the deploy box's numpy without the alias shim; include the shim if not.
- **Q7 (input domain for the UI's two modes):** the UI has `mode: 'sim' | 'geo'`
  (`sim.js:24`). The fine-tuned model is for the **real geophone** domain (`geo` mode). For
  `sim` mode, either keep `Sim` synthetic numbers OR run the *synthetic-pretrained*
  `model_ema.pt` on synthetic features. **Do not run the real-fine-tuned model on synthetic
  inputs** (domain mismatch) and vice-versa. Decide per-mode which checkpoint feeds the UI.

---

## Appendix — file:line index used

- `finetune_5fold.py` — FeatureSNN `21-34`; real featurize `37-51`; `yh/yv` `52`;
  `probs()` `56-62`; `thr()` `63-64`; `decide()` `65-66`; `finetune()` `68-81`;
  CV loop & per-fold scaler `86-101`; **only writes JSON** `103-108` (no `torch.save`).
- `geophone_snn_v2_train.py` — numpy shim `14-15`; `FeatureSNN` `108-133` (mean-over-T
  decode `132`); cupy training backend `137-138`; **saves `model_ema.pt`/`model_raw.pt`**
  `232-233`; real-test featurize/decode `353-398`.
- `rescale_test.py` — clean inference: build+load `35-36`; `hp()` last-step decode
  `37-43`; thresholds loaded from `ERROR_ANALYSIS.json` `16`; decode `63-64`.
- `snn_real_baseline.py` — same FT recipe, also un-saved (from-scratch vs FT comparison).
- `real_baseline.py` — confound caveats (docstring `4-8`); logreg/HGB refs.
- `snn_v2_out/ERROR_ANALYSIS.json` — thresholds `3-7`; last-step real acc `0.879` `8`.
- `snn_v2_out/FINETUNE_5FOLD.json` — zero-shot `0.859`, fine-tuned ensemble `0.977`,
  confound-clean `0.937`.
- `snn_v2_out/summary.json` — params `220528`, T=4, domain-classifier AUC `1.0`.
- `webui/src/sim.js` — output contract `94-123` (`probs{human,car,animal,nothing}`).
- `webui/src/detection.js` — consumes `w.detected` / `w.probs` `25-48`.
