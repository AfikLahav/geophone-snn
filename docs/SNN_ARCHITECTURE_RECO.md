# SNN Architecture + Hyperparameter Recommendation

**For:** single vertical 4.5 Hz geophone (1000 Hz), 3 s windows → human / vehicle / animal / nothing
with **per-class ordinal heads** (none / single / multiple), SpikingJelly (LIF/PLIF, surrogate-gradient BPTT).
**Status:** research-grounded recommendation. **Date:** 2026-06-13.
**Author note:** every quantitative SNN-on-seismic claim is cited; where the field is thin this document says so explicitly.

---

## 0. TL;DR (the headline recommendation)

1. **Train the feature-based MLP-SNN first** (Config A). The 102 features already give a gradient-boosted tree
   **0.963** 4-class balanced accuracy (`FEATURE_ANALYSIS.md` §1), so the discriminative signal is *known to be
   present and separable*. A feature-MLP-SNN converges in minutes, isolates "can an SNN read these features?"
   from "can an SNN do front-end DSP?", and gives you a working multi-head ordinal pipeline to validate the
   sim-to-real protocol on. It is the low-risk, high-information first move.
2. **Then train the raw-waveform 1D conv-SNN** (Config B) as the "no-hand-features" arm of the thesis. It is the
   scientifically interesting model (does sim-to-real transfer survive without hand-DSP?) and the one that maps
   cleanly to neuromorphic hardware, but it is higher-risk and slower, so it is second.
3. **Input encoding:** **direct/analog coding** for both — feed real values into the first spiking layer and
   repeat them over `T` steps. This is the dominant high-accuracy practice in modern audio SNNs and needs the
   fewest timesteps. Do **not** put the 3000 raw samples on the SNN time axis; use a strided conv stem to reduce
   the sequence and let `T` be small (4–8).
4. **Neuron:** **ParametricLIF (learnable τ)** as the default — you already use it, it is less sensitive to τ
   init, and "adaptation/learnable time-constant helps temporal/audio tasks" is the single most reproducible
   finding in the audio-SNN literature. Start `init_tau=2.0`, `v_threshold=1.0`, `v_reset=0.0`.
5. **Output:** **non-spiking membrane-potential readout** (integrator final layer, `v_threshold=∞`) — you already
   do this and it is the recommended readout for classification accuracy.
6. **Heads:** 3 independent **ordinal CORN heads** (2 logits each for none<single<multiple), co-occurrence handled
   because the heads are independent (binary-relevance style). **Demote the vehicle head to 2-level (none/present)**
   per `FEATURE_ANALYSIS.md` §5 — vehicle single-vs-multiple is at chance and not learnable as the data stands.
7. **Honest caveat:** there is **no published SNN that classifies human/vehicle/animal on a geophone**. The
   closest direct precedent is a 2007 dynamic-synapse net (Dibazar/Berger). The recommendation is built by
   transferring the mature **vibration-SNN** and **audio-SNN** recipes onto your task; expect a real-data gap and
   budget a small real fine-tune set.

---

## 1. Published SNN architectures for time-series / audio / seismic / UGS

### 1.1 SNN directly on seismic — the field is nascent (be honest)

There is **no published SNN doing human/vehicle/animal target ID on geophone data.** What exists:

| Work | What it does | Result | URL |
|---|---|---|---|
| Gaurav, Stewart, Yi 2023 (Frontiers Comp. Neurosci.) — Legendre reservoir / LSNN | Univariate TS classification incl. **UCR "Earthquakes"** (binary) | **80.43%** vs prior spiking SoTA 71.94%; ~31× energy on Loihi | https://pmc.ncbi.nlm.nih.gov/articles/PMC10285304/ |
| Kasabov NeuCube line; ICONIP'24 "SNN for NZ earthquake prediction" | 3-D reservoir of spiking neurons, spatio-temporal seismic | ~38%→70% as event nears (soft numbers) | https://ojs.aut.ac.nz/iconip24/2/article/view/52 ; framework: https://www.sciencedirect.com/science/article/abs/pii/S0893608014000070 |
| SNN for microseismic detection on DAS (Springer LNCS 2024) | Event detection on fiber-optic strain | paywalled, numbers not extractable | https://link.springer.com/chapter/10.1007/978-3-031-66965-1_31 |

The single clean, reproducible SNN-on-seismic accuracy number in the literature is **80.43% on UCR Earthquakes**
(a single-station binary "is a big quake imminent" task) — i.e. quantitative evidence for *your* problem is
effectively zero. Treat seismic-SNN as a research frontier you are extending, not a solved recipe.

### 1.2 The real evidence base: vibration / accelerometer SNNs (closest analog)

A geophone is a 1-axis vibration sensor, so vibration-SNNs are the most transferable prior art.

- **Vasilache et al. 2025, "SNNs for Low-Power Vibration-Based Predictive Maintenance"** — the single best
  architecture blueprint for you. Recurrent LIF, **input projection 12 → two recurrent LIF layers of 160 → 6
  outputs (3 regression + 3 classification)**, ~80k params. Step-Forward (delta) + Poisson encoding of 3-axis
  accelerometer. **97.95% classification, 0% false-negative on critical faults.** Adam lr≈1.2e-2, wd≈1.7e-6,
  dropout 0.07, fast-sigmoid surrogate (slope 5), BPTT, mean-over-time readout, multi-head regression+classification
  with **inverse-frequency-weighted CE**. Loihi 3.16e-3 J/inference vs ARM 1.18 J. → https://arxiv.org/html/2506.13416v1
- **Vathakkattil Joseph & Pakrashi 2022, "SNN for Structural Health Monitoring" (Sensors, on Loihi)** — spiking
  **gammatone filterbank (36 log filters 0–2000 Hz) → cepstral** front-end, two-layer LIF, 552 neurons, 1.5 s
  windows, ~0.1 W. Demonstrates a *spiking DSP front-end* on accelerometer vibration. → https://pmc.ncbi.nlm.nih.gov/articles/PMC9740015/
- **Dennler, Haessig, Cartiglia, Indiveri 2021, "Online Detection of Vibration Anomalies Using Balanced SNNs"
  (AICAS)** — cochlea front-end + balanced SNN, online/unsupervised, mixed-signal neuromorphic. → https://arxiv.org/abs/2106.00687

### 1.3 The mature analog: audio / keyword-spotting SNNs (the recipe to copy)

Audio SNNs are the closest *well-benchmarked* field. The consensus feedforward recipe and the SOTA points:

| Work | Architecture | Dataset / Result | URL |
|---|---|---|---|
| **Bittar & Garner 2022** (the reference baseline) | 2 hidden + readout; compares LIF / **adLIF** / RLIF / RadLIF; membrane-sum readout; 40 mel-frames as T | SHD **94.62%**; **non-recurrent adLIF 93.06% vs LIF 87.04%** (same size) → adaptation is worth ~6 pts | https://www.frontiersin.org/journals/neuroscience/articles/10.3389/fnins.2022.865897/full ; code https://github.com/idiap/sparch |
| **Yin, Corradi, Bohté 2021** (Nat. Mach. Intell.) | recurrent **adaptive-LIF (ALIF, adaptive threshold)** | SHD ~90.4%; established adaptation closes gap to LSTM cheaply | https://arxiv.org/abs/2103.12593 |
| **DCLS-Delays, Hammouamri et al. ICLR 2024** | **feedforward vanilla LIF + learnable synaptic delays** (dilated conv, learnable spacings), 2–3 hidden FC | SHD 95.07%, **SSC 80.69%**, GSC 95.35% — matches recurrent SNNs with no recurrence | https://arxiv.org/abs/2306.17670 |
| **GPN (gated parametric neuron) 2024** | FC gated-adaptive-LIF, **T=40 (SHD) / 60 (SSC)** | SHD **90.8% vs LIF 75.8% vs PLIF 71.7%** — adaptation/gating >> plain or parametric LIF | https://arxiv.org/html/2412.01087v1 |
| **Pellegrini et al. SLT 2021** (cleanest conv-SNN-KWS) | stacked **Conv→LIF→pool** (VGG-like) on log-mel frames, CuBa-LIF, BPTT | ~**94% on Google Speech Commands**, extreme sparsity, 7.5× energy saving | https://arxiv.org/abs/2011.06846 |
| **Yılmaz et al. Interspeech 2020** | deep convolutional SNN | matches CNN on Speech Commands, **beats it on wakeword** via event sparsity | https://www.isca-archive.org/interspeech_2020/ylmaz20_interspeech.html |
| SpikCommander 2025 (SOTA ceiling) | spike-driven transformer, LIF τ=2.0 Vth=1.0, T=100–200 | GSC **97.08%** (first SNN >97%), 2.13M params | https://arxiv.org/html/2511.07883 |

**The consensus feedforward audio-SNN recipe** (what actually works): framed log-mel/filterbank features → a small
**strided conv stem** over the feature axis → **2–3 adaptive-LIF (or PLIF) layers**, optionally **+ learnable
delays** → **membrane-sum readout**; **ATan surrogate α≈2, BPTT, Adam, cosine, BN-through-time**. Accuracy lands
low-to-mid 90s on GSC-difficulty tasks at a fraction of an ANN's compute. **adLIF > vanilla LIF is the single
highest-ROI architectural choice** (+5 to +15 pts on SHD across multiple papers).

### 1.4 Foundational non-spiking seismic UGS work (informs feature & baseline design)

These are not SNNs but they (a) justify the cadence/envelope feature family you built and (b) set the ANN baseline
your SNN must approach.

- **Park, Dibazar, Berger 2009 (ICASSP), "Cadence Analysis … Human vs Quadruped Footsteps"** — the canonical
  biped-vs-quadruped seismic work: temporal **gait-beat** structure (not just fundamental cadence) modeled with
  GMMs, **>95%**. This is exactly the `beat_depth` / `cad_harm2/harm4` / `footfall_rate` logic in your `features.py`
  (CAD12). → https://ieeexplore.ieee.org/document/4959942/
- **Dibazar & Berger 2007 (IJCNN), "Dynamic Synapse Neural Networks on Footstep and Vehicle Recognition"** — the
  closest *spiking-adjacent* precedent to your exact problem: footstep + vehicle recognition with animal rejection
  on a desert geophone, using biologically-realistic time-varying-dynamics synapses. → https://ieeexplore.ieee.org/document/4371238/
- **Modern ANN seismic-UGS baselines (the numbers to beat):** CNN on **SITEX02** (DARPA UGS) using LFCC features
  reaches **92.05%**; ACIDS (acoustic-seismic) is the other standard vehicle-type set. Contrastive/self-supervised
  seismic vehicle classification reports ~99.8% on a curated set. So a *tree at 0.96 and an ANN-CNN in the low-90s*
  is the regime; an SNN matching mid-80s–low-90s would be a strong result.
  → SITEX02/ACIDS: https://www.nature.com/articles/s41598-025-01684-x ; CNN footstep band: https://library.seg.org/doi/10.1190/tle39090654.1

### 1.5 1D conv-SNN design conventions for raw waveform (channels / kernels / strides)

From the audio-SNN and raw-waveform-CNN literature:

- **Do not stream raw samples as the SNN time axis.** Universal practice is to reduce the sequence first — either
  compute framed features (frames become `T`) or use a **strided Conv1D stem**, then keep `T` small. (LMUFormer
  explicitly shows the conv stem is what unlocks accuracy: naive LMU 76% → +conv 89%.)
- **First conv layer = large kernel, large stride** to downsample raw signal cheaply (M5/M-series raw-audio CNNs:
  kernel ~80, stride 4–16), then **small kernel-3 blocks doubling channels** (e.g. 64→128→256→512) with stride-2 or
  MaxPool between blocks. Ports directly to `Conv1d → BN → LIF → MaxPool1d` spiking blocks.
- **Spiking conv block (SpikingJelly idiom):** `Conv1d(bias=False) → BatchNorm1d → spiking_neuron → MaxPool1d`,
  repeated; classifier head with `Dropout` then `Linear → neuron → Linear → integrator`. Max-pool over avg-pool
  (cheaper, binary-friendly, no real accuracy loss — PLIF paper). For depth >~10 layers switch BN to **tdBN**.
- A concrete hybrid example: CNN-SNN auditory front-end uses a **Conv1D kernel=64 (≈0.25 s at 256 Hz), 40 channels**
  as the stem before spiking layers (https://arxiv.org/pdf/2307.08501). Your equivalent at 1 kHz is a kernel of
  ~64 samples with stride 8 (below).

### 1.6 Feature-vector MLP-SNNs (for Config A)

SNNs on tabular / feature-vector inputs are under-published but work: a simple **2-dense-layer LIF MLP** matches an
MLP on function regression (3× faster), and **direct coding** (analog input to the first spiking layer, rescaled to
[0,1], repeated each timestep) "dominates high-accuracy deep SNN training." Sigma-Delta neurons with direct input
hit **83.0% CIFAR-10 at T=2**. For a 102-feature vector the right design is exactly your existing notebook scaled up:
`Linear → PLIF → Linear → PLIF → integrator`, direct-coded, small T. → https://arxiv.org/html/2502.10423v1 ,
https://arxiv.org/pdf/2210.13107

---

## 2. Input encoding

### 2.1 The 102-feature vector → **direct (analog) coding**

**Recommendation: direct/analog coding** — standardize each feature (z-score, then clip to ±5σ as you already do),
feed the real-valued vector into the first `Linear`, and present the **same** vector at every one of the `T`
timesteps (the first PLIF layer is the analog-in/spike-out boundary). Reasons:

- It is the dominant high-accuracy practice and needs the **fewest timesteps** (rate/latency coding need many more
  steps and add timing fragility for no accuracy gain on a static feature vector).
- It preserves the full precision of features that are already information-dense (a tree gets 0.96 from them).
- It matches your existing notebook (which already direct-codes 32 features) — zero new machinery.

Do **not** rate-code or latency-code the feature vector: these throw away precision and inflate `T`. Empirical
encoder benchmarks on sensor time-series confirm rate-coding is the *most accurate* spike encoder but direct/analog
input is strictly better when you can keep an analog first layer (you can). If you ever target a chip that forbids
analog input, fall back to **multi-threshold delta / TAE** encoding (best fidelity-per-spike) — but that is a
deployment-time concern, not a training one. (Encoders: https://link.springer.com/article/10.1007/s11063-021-10562-2 ;
TAE/SF/MW benchmark https://arxiv.org/html/2503.11206 .)

**Normalization is load-bearing for sim-to-real.** Use **per-feature train-set z-score** (global μ/σ from the
synthetic train split, applied unchanged to val/real) for the feature vector. Critically, your `FEATURE_ANALYSIS.md`
§4/§9 already establishes the rule: **never feed absolute amplitude or narrow-band absolute energy** — they are
distance/terrain-confounded. The fractions/shape/envelope features are the transferable signal; the z-score keeps
them scale-stable across the sim-real gap.

### 2.2 The raw 3000-sample waveform → **conv stem, not the time axis**

**Recommendation:** feed the standardized 1×3000 waveform as a real-valued tensor into a **strided Conv1D stem**
(analog-in), let the stem downsample the sequence, and run the spiking layers with **small `T` (4–8)** using direct
coding (repeat the stem's output, or — better — run the conv stem once and inject its feature map into the spiking
stack each timestep). Do **not** map sample-index → SNN timestep: 3000 steps of BPTT is ruinously expensive
(memory and compute scale O(T)) and unnecessary.

**Per-window standardization** (per-instance zero-mean/unit-variance on the 3000 samples) is the recommended input
norm for the raw model — it makes the model **amplitude- and gain-invariant**, which is the single
highest-leverage, lowest-cost sim-to-real move (real sensor gain/coupling/distance scale amplitude arbitrarily;
RevIN-style instance norm removes that axis). Combine with a fixed analysis band: the useful band is ≤~200 Hz, so
**low-pass/anti-alias and downsample 1000 Hz → 250 Hz** before the stem. That cuts the sequence 4× (3000 → 750
samples) with no loss in-band, shrinking the conv stack and the spike budget for free. (Your features already treat
20–90 Hz as the footstep band and ≤200 Hz as the ceiling; 250 Hz Nyquist = 125 Hz, which clips the 90–180 "high"
band — if you want to keep the full 180 Hz, downsample to **400 Hz** instead, 3000 → 1200 samples. Recommended:
**400 Hz** to preserve the high band that carries footstep impulsiveness; 250 Hz only if you confirm >180 Hz is
unused.)

**Recommended downsampling: 1000 → 400 Hz (decimate-by-2.5 via polyphase/`scipy.signal.resample_poly`), giving a
1200-sample window**, then a strided conv stem reduces 1200 → ~75 timeable features.

---

## 3. Neuron choice

**Recommendation: ParametricLIF (PLIF, learnable τ per layer) as the default for both configs.** Rationale:

- You already use PLIF, and it is in SpikingJelly as `neuron.ParametricLIFNode(init_tau=...)`. Learnable τ makes the
  net **less sensitive to τ initialization** and **converges faster** (Fang et al. ICCV 2021), and different layers
  learn different τ (more expressiveness). → https://arxiv.org/abs/2007.05785
- "Adaptive / learnable time-constant helps temporal & audio tasks" is the **most reproducible** result in the
  audio-SNN literature (Bittar & Garner: adLIF +6 pts over LIF; GPN: +15 pts over LIF, +19 over plain PLIF). A
  geophone window is a temporal signal with cadence structure, so this applies.

**Starting values (both configs):** `init_tau = 2.0`, `v_threshold = 1.0`, `v_reset = 0.0` (hard reset),
`surrogate_function = surrogate.ATan(alpha=2.0)`, `detach_reset = True`, `step_mode = 'm'`.
Note SpikingJelly's *library* default surrogate is Sigmoid(α=4); **explicitly override to ATan(α=2)** — that is what
every modern high-accuracy SNN uses. Your old notebook used Sigmoid(α=25), which is far steeper than current
practice; α=2–4 is the recommended band (training is robust to surrogate *shape* but sensitive to *steepness*, and
steepness interacts with LR — Zenke & Vogels). Your old `v_threshold=0.5` is also fine, but `1.0` is the modern
default and what tdBN is tuned against; pick `1.0`.

**If you want to push accuracy on the temporal structure (a worthwhile experiment, especially for raw waveform):**
upgrade the hidden neurons to an **adaptive LIF** (adLIF / ALIF with an adaptation/recovery variable). SpikingJelly
does not ship adLIF as a one-liner the way snnTorch/sparch do, so this is a small custom-neuron effort; treat it as
**v2** once PLIF baselines are in. The expected upside (audio literature) is +5–15 pts *on hard temporal sets* —
your task is easier (features are near-separable), so the gain will be smaller, but it is the most promising single
lever if the raw-waveform model underperforms. Vanilla `LIFNode` is the simplest fallback and is fine for Config A.

**Do not** reach for fancier neurons (Izhikevich, full liquid-time-constant) — no evidence they help this task and
they complicate BPTT.

---

## 4. Timesteps T

The accuracy/compute trade-off: SNN accuracy rises steeply from T=1→4 then flattens; **T=4–8 is the modern sweet
spot** for low-latency classification, and energy/compute scale ~linearly in T. Static-image SNNs now run at T=4
(some at T=1 with <1% drop); neuromorphic/temporal data uses T=16; audio SNNs with distillation run as low as T=40
*only because they put mel-frames on the time axis* (their "T" is the sequence length, a different quantity from
ours). Because we use **direct coding with a conv stem (not frames-as-time)**, our `T` is the number of repeated
integration steps, and it should be small.

| Input type | Recommended T | Trade-off notes |
|---|---|---|
| **102-feature MLP-SNN (Config A)** | **T = 4** (sweep {2, 4, 8}) | Features are static per window; T is just integration depth. T=4 is plenty; T=2 may suffice (Sigma-Delta hits 83% CIFAR at T=2). Your old T=25 is overkill — drop it. |
| **Raw-waveform conv-SNN (Config B)** | **T = 8** (sweep {4, 8, 16}) | Slightly more temporal integration helps the conv features settle; still cheap. Go to 16 only if 8 underperforms. |

Energy/compute is O(T · synaptic-ops). For Config A at ~150–300k params, T=4 vs T=25 is a **~6× compute saving** with
equal-or-better accuracy. Train-time BPTT memory is also O(T), so small T speeds training and lets you use the full
7.4M-window corpus in reasonable wall-clock.

---

## 5. Full hyperparameter set (shared training recipe)

This is the consensus SpikingJelly recipe, reconciled with your existing notebook (which is already close — keep its
EMA, spike-reg, cosine, AdamW, gradient clipping; change surrogate α, T, readout-per-head, and loss).

| Knob | Recommendation | Why / source |
|---|---|---|
| **Framework mode** | `functional.set_step_mode(net,'m')`; GPU: `functional.set_backend(net,'cupy')`; call `functional.reset_net(net)` every batch | multi-step + cupy = ~11× speedup (SpikingJelly paper). Essential at 7.4M windows. |
| **Surrogate** | **ATan, α = 2.0** | modern default (SpikCommander/PLIF/LMUFormer); α 2–4 safe band; tune with LR. https://arxiv.org/abs/1901.09948 |
| **Optimizer** | **AdamW, lr = 1e-3** | default for neuromorphic/temporal data + PLIF paper; easiest to tune. (SGD+m 0.9 @0.1 is the static-image-CNN alternative; not needed here.) |
| **LR schedule** | **Cosine annealing** to `eta_min=1e-5`, over all epochs; optional 3–5 epoch linear warmup | near-universal in current SNN work; you already use it. |
| **Weight decay** | **1e-4** (AdamW, decoupled) | standard with Adam; your notebook's value. |
| **Batch size** | **256** (feature MLP) / **128** (raw conv) | large dataset → large batches fine; raw conv is memory-heavier. (Your old 16 is tiny — raise it; you have 7.4M windows.) |
| **Dropout** | **0.1–0.2** hidden (feature MLP); **0.2** in conv classifier head | your notebook uses 0.1; SpikingJelly model-zoo uses 0.5 in heads. With a huge dataset, lean lower (0.1–0.2). |
| **BatchNorm** | **`BatchNorm1d` after each Linear/Conv, before the neuron**; raw-conv depth is shallow so plain BN is fine. Use **tdBN** only if you deepen past ~10 layers. | BN balances firing rates / fixes gradient vanishing across the temporal unroll (Zheng tdBN, AAAI'21). https://arxiv.org/abs/2011.05280 |
| **Loss** | **TET-style: per-timestep loss averaged over T**, `L = (1/T) Σ_t loss(O_t, y)`, with the per-head ordinal/BCE losses of §6 | flatter minima, better low-T generalization (Deng et al. ICLR'22). https://arxiv.org/abs/2202.11946 . (Your notebook already sums CE over all T steps of the membrane readout — TET is the principled version.) |
| **Readout** | **non-spiking integrator final layer per head**: `LIFNode(v_threshold=float('inf'), store_v_seq=True)`, read membrane potential | recommended readout for classification accuracy; denser gradient, no spike-count ties (Eshraghian review). You already do this. |
| **Spike-rate reg** | **optional, λ ≈ 1e-3 toward target rate 0.1**, two-sided (penalize dead + over-firing) | light reg ≈ accuracy-neutral while improving sparsity/energy; your notebook's λ=1e-3/target=0.1 is reasonable. Keep small; turn off if it costs accuracy. |
| **Gradient clipping** | **global L2 norm = 1.0** | stabilizes BPTT unroll; you already do this. |
| **EMA of weights** | **keep, decay 0.999** (≈ your 0.9997 → adjust half-life to dataset size) | weight averaging → flatter, better-generalizing, **better-transferring** optima (SWA, Izmailov'18). Validate with EMA weights as you already do. |
| **Epochs** | **30–50** over the full 7.4M-window corpus (1 epoch is huge); use **early stopping** on real-val macro-F1 with patience 5 | with 7.4M windows you need far fewer epochs than the old 75 on a tiny set. Watch the learning curve. |
| **Class balancing** | effective-number reweighting (β≈0.999) and/or focal (γ=2) on the heads (§6) | abundant "nothing"/"none" windows otherwise dominate. https://arxiv.org/abs/1901.05555 , https://arxiv.org/abs/1708.02002 |

---

## 6. Multi-head / multi-task output design (the 3 ordinal heads)

### 6.1 Structure: 3 independent ordinal heads off a shared backbone

```
shared SNN backbone  ──►  head_human   : ordinal CORN, 2 logits (none<single<multiple)
                     ──►  head_vehicle : binary,      1 logit  (none / present)   [DEMOTED, see §6.5]
                     ──►  head_animal  : ordinal CORN, 2 logits (none<single<multiple)
```

The heads are **independent** (each a small `Linear → integrator` reading the shared backbone's last spiking
layer). Independence is exactly what gives you **co-occurrence**: human, vehicle and animal can each fire at once
because they are separate Bernoulli/ordinal outputs, not a softmax over mutually-exclusive classes. This is the
"binary relevance" multi-label design — **never put a softmax across co-occurring classes** (it forces them to
compete and sum to 1).

### 6.2 Ordinal head = CORN (recommended) over CORAL over softmax

For an ordinal scale with K=3 levels (none < single < multiple) you have **K−1 = 2 thresholds**, so each ordinal
head has **2 logits**:

- **CORN** (Shi, Cao, Raschka 2021): the 2 logits are **conditional probabilities** `P(y>none)` and
  `P(y>single | y>none)`, trained on *conditional subsets* (the second logit is trained only on samples with
  y>none). Unconditional rank probabilities reconstruct by the chain rule `P(y>single)=P(y>none)·P(y>single|y>none)`,
  which **guarantees rank-monotone outputs without weight-sharing** (so the backbone keeps full capacity). Predicted
  level = number of `P(y>r_k) > 0.5`. CORN ≥ CORAL ≥ ordinal-NN ≥ plain CE empirically (e.g. MORPH age MAE: CE 3.73 →
  CORAL 2.99 → CORN 2.98). → https://arxiv.org/abs/2111.08851
- **CORAL** (Cao et al. 2020): same K−1 binary outputs but enforces monotonicity by **sharing the weight vector and
  using K−1 separate biases**. Simpler, slightly less expressive. → https://arxiv.org/abs/1901.07884
- **Plain softmax (3-way)** ignores order — it penalizes none↔multiple the same as none↔single. **Honest caveat:**
  with only 2 thresholds the CORN/CORAL gain over a well-regularized softmax is *modest*. CORN is justified here
  mainly because (a) confusing **none↔multiple** is more costly than **none↔single** (you want rank-aware error),
  and (b) it gives **monotone, better-calibrated confidence** — useful for the hysteresis/policy layer downstream
  and for the marginal-SNR zone. If you want the absolute simplest first cut, a 3-way softmax per head will work and
  you can upgrade to CORN; but CORN is cheap (1 fewer logit than softmax) so prefer it from the start.

**SNN-specific note:** read each head's 2 (or 1) logits from a **non-spiking integrator** (membrane potential),
apply sigmoid for the CORN/BCE probabilities. This is your existing membrane-readout, just one small integrator
head per class instead of one shared 3-way readout.

### 6.3 Readout: membrane potential, not spike count

Use **membrane-potential readout** (integrator `v_threshold=∞`) per head — the recommended readout for
classification accuracy (denser gradient, no spike-count ties, robust to reset suppressing strong inputs). Average
the per-timestep membrane logits over T (or use the final step — your notebook uses `mem[-1]`; mean-over-T is the
TET-consistent choice and slightly more stable). Spike-count / population-voting readouts are an alternative but buy
nothing here.

### 6.4 Loss design

Per head:
- **Ordinal heads (human, animal):** **CORN loss** = sum of binary cross-entropies over the K−1 conditional tasks
  (second task computed only on the conditional subset). Implementable directly from the formula; or use
  `coral_pytorch`'s `corn_loss`. Wrap each per-timestep with the **TET average** (§5).
- **Vehicle (binary):** **BCE-with-logits**.
- **Class imbalance:** apply **effective-number class weights (β≈0.999)** and/or **focal (γ=2)** so the dominant
  "none"/"nothing" level does not swamp gradients. The corpus has abundant lead-in/lead-out `nothing` windows by
  construction (`CLASS_LIST_AND_VARIABLES.md` §6), so this matters.
- **Marginal-SNR zone (the [−20,0) dB faint band, `FEATURE_ANALYSIS.md` §6):** use **soft targets / label
  smoothing** (ε≈0.05–0.1) on examples in that band, and/or **feature-space mixup**, to represent genuine label
  uncertainty and improve calibration. Do **not** hard-label a −18 dB footstep as confidently "single." (Label
  smoothing: https://arxiv.org/abs/1906.02629 ; mixup: https://arxiv.org/abs/1710.09412 — prefer feature-space
  mixup; blind waveform mixup can create unphysical signals.)
- **Multi-head loss weighting:** start with **equal weights** (the heads are comparable BCE-scale losses). If one
  head dominates, switch to **Kendall uncertainty weighting** (learnable per-head σ, weight 1/(2σ²)+log σ) — the
  simplest robust auto-balancer. Escalate to GradNorm only if needed. (https://arxiv.org/abs/1705.07115)

Total loss:
```
L = Σ_heads  w_head · (1/T) Σ_t  HeadLoss_t            # TET-averaged, optionally uncertainty-weighted
  + λ_spike · spike_rate_reg                            # optional, small
```

### 6.5 Vehicle head: demote to 2-level (none / present)

`FEATURE_ANALYSIS.md` §5 is explicit and authoritative: **vehicle single-vs-multiple is at chance (bal-acc 0.515,
AUC 0.606); multi is called single 96.6% of the time; the level is not learnable from these features as the data
stands.** Recommendation: make the **vehicle head binary (none/present)** until convoy physics is strengthened
(per-vehicle RPM jitter, §5 option 2). Do not present vehicle single/multiple as working. Human (0.83) and animal
(0.92) single/multiple **do** work — keep those as 3-level CORN heads.

### 6.6 Multi-head SNNs — honest evidence note

True multi-*head*, multi-label SNNs are rare. MT-SNN time-shares one task at a time; MTSpark is RL; the **vibration
predictive-maintenance SNN (Vasilache 2025) is the one clear precedent doing simultaneous multi-label
classification + regression from a vibration sensor**, and it works (97.95%, inverse-frequency-weighted CE,
mean-over-time readout). So your design (CORN/BCE heads + membrane readout on a shared SNN backbone) is *porting
solid ANN practice onto an SNN backbone with one supporting vibration-SNN precedent* — reasonable, not directly
precedented for ordinal heads. Verify head balance empirically.

---

## 7. Sim-to-real for SNNs

The sim-to-real gap is your stated central risk (`INTRODUCTION.md`). Ranked by leverage:

1. **Per-instance / per-window amplitude-invariant normalization** — highest leverage, lowest cost. Raw model:
   per-window zero-mean/unit-variance. Feature model: train-set z-score + the existing rule **never use absolute
   amplitude or narrow-band absolute energy** (use fractions/shape/envelope — `FEATURE_ANALYSIS.md` §4/§9). Real
   gain/coupling/distance scale amplitude arbitrarily; instance norm removes that axis (RevIN, ICLR'22).
2. **Wide domain randomization in the simulator** — you already do this (K=300 soil profiles, per-scene Q jitter,
   randomized noise lines, SNR 0–40 dB, randomized gain). Tobin et al. 2017: *breadth* of randomization is the key
   transfer lever; randomize every uncertain factor (SNR, coupling f_c/Q_c, source distance/amplitude, sensor gain,
   noise type) even slightly past realistic ranges. → https://arxiv.org/abs/1703.06907
3. **Augmentation** — on top of the simulator's per-epoch fresh-noise + shift/scale (your `CLASS_LIST` §6c): add
   **SpecAugment** (time/freq masking) if you adopt a spectral front-end, plus classic **additive real-noise,
   random gain, time-shift, time-stretch**. Your existing **splice augmentation** (insert bursts into nothing
   windows) is excellent and directly creates marginal-SNR examples — keep it.
4. **Flat-minima regularizers transfer better:** keep **EMA/SWA** of weights, modest **weight decay**, **dropout**.
   Flatter optima generalize across distribution shift.
5. **Mild SNN-native robustness:** SNNs show *somewhat* higher inherent noise/adversarial robustness than equivalent
   ANNs (spike discretization + leaky LIF dynamics; smaller T and LIF-over-IF help). This may *slightly* help the
   sim-real gap but is **not** demonstrated domain transfer and is **not** a substitute for real validation.
   → https://arxiv.org/abs/2003.10399
6. **Plan for synthetic-pretrain → small real fine-tune.** The only existing SNN sim-to-real evidence is
   event-camera vision (zero-shot ~71% → ~92.5% after a small real fine-tune); **none on seismic.** Budget a small
   labeled real set (once the AD620 rig is up) to (a) measure the gap honestly — real is validation-only per the
   thesis — and (b) fine-tune if the gap is large (the documented fallback in `INTRODUCTION.md`).

---

## 8. CONFIG A — Feature-based MLP-SNN (~150–300k params) — **train this first**

**Input:** 100 features (102 minus the two dead features `skewness`, `ar_5` per `FEATURE_ANALYSIS.md` §7), z-scored
with train-set μ/σ, clipped to ±5. Direct/analog coding (same vector at every timestep).

**Backbone + heads (SpikingJelly, `step_mode='m'`):**

| # | Layer | Shape | Notes |
|---|---|---|---|
| in | feature vector | 100 | z-scored, clipped ±5; repeated over T |
| 1 | `Linear(100, 512, bias=False)` | →512 | |
| 2 | `BatchNorm1d(512)` | →512 | before neuron |
| 3 | `ParametricLIFNode(init_tau=2.0, v_threshold=1.0, surrogate=ATan(2.0), detach_reset=True)` | →512 | hidden 1 |
| 4 | `Dropout(0.15)` | →512 | |
| 5 | `Linear(512, 256, bias=False)` → `BatchNorm1d(256)` → `ParametricLIFNode(...)` → `Dropout(0.15)` | →256 | hidden 2 |
| 6 | `Linear(256, 128, bias=False)` → `BatchNorm1d(128)` → `ParametricLIFNode(...)` | →128 | shared feature (no dropout) |
| 7a | head_human: `Linear(128, 2)` → integrator `LIFNode(v_threshold=∞)` | →2 | CORN ordinal (none<single<multiple) |
| 7b | head_vehicle: `Linear(128, 1)` → integrator `LIFNode(v_threshold=∞)` | →1 | binary (none/present) — demoted |
| 7c | head_animal: `Linear(128, 2)` → integrator `LIFNode(v_threshold=∞)` | →2 | CORN ordinal |

**Param count:** 100·512 + 512·256 + 256·128 + 128·5 = 51,200 + 131,072 + 32,768 + 640 = **~216k** (+ BN/PLIF
scalars). Squarely within the 150–300k target. (Narrower 384→192→96 ≈ 131k — just under the band, fine if you want
a smaller net; wider 640→320→160 ≈ 321k — just over, trim hidden-1 to 600 for exactly ~300k.)

**Training:** T=4 (sweep {2,4,8}); AdamW lr=1e-3, cosine→1e-5, wd=1e-4; batch 256; dropout 0.15; TET loss with CORN
heads (human/animal) + BCE (vehicle), effective-number class weights β=0.999; grad-clip 1.0; EMA 0.999; spike-reg
λ=1e-3 target 0.1 (optional); 30–50 epochs over the full corpus, early-stop on real-val macro-F1 patience 5.

**Why first:** isolates the SNN-readout question from the front-end-DSP question; the 0.96 tree ceiling proves the
features are separable; converges in minutes; gives a working multi-head ordinal + sim-to-real harness immediately.
**Target:** approach the tree's 0.96 4-class; a strong result is ≥0.90 4-class on held-out *profiles*, with the
human/animal single-vs-multiple heads near their tree AUCs (0.93 / 0.98).

---

## 9. CONFIG B — Raw-waveform 1D conv-SNN (~1–2M params) — **train second**

**Input:** raw 1×3000 window → anti-alias low-pass → **decimate 1000→400 Hz (`resample_poly`, 1×1200)** →
**per-window standardize** (zero-mean/unit-variance). Direct/analog coding into the conv stem; small T.

**Backbone (spiking conv block = `Conv1d(bias=False) → BatchNorm1d → PLIF → MaxPool1d`):**

| # | Block | In→Out ch | kernel / stride / pool | Seq length | Notes |
|---|---|---|---|---|---|
| in | standardized waveform | 1 | — | 1200 | 400 Hz, 3 s |
| 1 | **stem** Conv1d k=**64**, s=**8** → BN → PLIF | 1→32 | k64/s8, no pool | 1200→~150 | large-kernel/large-stride stem downsamples raw signal (M-series convention) |
| 2 | conv block | 32→64 | k3/s1, MaxPool1d(4) | 150→~37 | double channels, pool 4 |
| 3 | conv block | 64→128 | k3/s1, MaxPool1d(2) | 37→~18 | |
| 4 | conv block | 128→256 | k3/s1, MaxPool1d(2) | 18→~9 | |
| 5 | conv block | 256→256 | k3/s1, MaxPool1d(2) | 9→~4 | |
| 6 | `AdaptiveAvgPool1d(1)` → flatten | 256 | — | →256 | global pool over remaining time |
| 7 | `Linear(256, 256, bias=False)` → BN → PLIF → `Dropout(0.2)` | →256 | | shared FC |
| 8a | head_human: `Linear(256, 2)` → integrator | →2 | | CORN ordinal |
| 8b | head_vehicle: `Linear(256, 1)` → integrator | →1 | | binary (none/present) |
| 8c | head_animal: `Linear(256, 2)` → integrator | →2 | | CORN ordinal |

**Param count (dominant terms):** stem 1·32·64≈2k; 32·64·3≈6k; 64·128·3≈25k; 128·256·3≈98k; 256·256·3≈197k;
FC 256·256≈66k; heads ≈1.3k → conv+FC ≈ **~0.4M**. To reach **1–2M** (the requested envelope, and useful for the
harder raw task) **widen channels** to 64→128→256→512→512 and FC 512: stem 4k; 64·128·3≈25k; 128·256·3≈98k;
256·512·3≈393k; 512·512·3≈786k; FC 512·512≈262k → **~1.6M**. **Recommended: the wide variant (~1.6M).**

**Training:** T=8 (sweep {4,8,16}); AdamW lr=1e-3, cosine→1e-5, wd=1e-4; batch 128; dropout 0.2; **plain BatchNorm1d**
(stack is ~6 layers — only switch to **tdBN** if you deepen); TET loss + CORN/BCE heads + class weights as Config A;
grad-clip 1.0; EMA 0.999; **per-window standardization + heavy augmentation** (additive real-noise, random gain ±,
time-shift, time-stretch, splice-augmentation); 40–60 epochs, early-stop on real-val macro-F1.
**Optional v2 upgrade:** swap hidden PLIF → adaptive-LIF (custom neuron) if accuracy lags — highest-ROI lever for
the temporal raw task.

**Why second:** it is the scientifically interesting "no hand-features" arm and the neuromorphic-deployment target,
but it is higher-risk (must learn the front-end DSP the features already encode) and slower. Let Config A de-risk the
pipeline first. **Target:** match or beat Config A on 4-class while using no hand-DSP; honest success is
low-90s–approaching-Config-A. If it underperforms Config A by a lot, that is itself a clean result (the hand-features
carry transferable signal the conv stem can't recover from limited real data).

---

## 10. Recommended order of work + decision gates

1. **Config A, T=4, 4-class only (collapse heads to a single 4-way softmax) — smoke test.** Confirm the SNN reads the
   features: target ≥0.90 4-class on held-out profiles within an hour of training. *Gate: if an SNN can't get near
   the 0.96 tree on these features, fix encoding/normalization before anything else.*
2. **Config A, full 3-ordinal-head design** (vehicle demoted). Validate the multi-head ordinal + class-weighting +
   marginal-SNR soft-target machinery. This is the deliverable model for the feature arm.
3. **Sim-to-real check on Config A** once the AD620 rig produces real captures — measure the gap (real = validation
   only). Decide if fine-tuning is needed.
4. **Config B (wide, ~1.6M), T=8.** The raw-waveform arm. Compare to Config A. Try the adaptive-LIF upgrade if it
   lags.
5. **(Deployment, separable):** if a neuromorphic target is chosen (Xylo-IMU is the strongest commercial fit for a
   vibration/IMU edge node; Loihi/Akida also viable), quantize + prune Config B (the vibration paper hit 175 KB /
   ~80k params at 18-bit with 25% pruning). Per `INTRODUCTION.md`, the SNN's efficiency is real only on neuromorphic
   HW; chip choice is a later decision.

---

## 11. Sources (all cited inline; grouped)

**SNN-on-seismic / vibration (closest prior art):**
- Gaurav, Stewart, Yi 2023, Legendre reservoir / LSNN, UCR Earthquakes 80.43% — https://pmc.ncbi.nlm.nih.gov/articles/PMC10285304/
- Kasabov NeuCube (Neural Networks 2014) — https://www.sciencedirect.com/science/article/abs/pii/S0893608014000070 ; NZ earthquake ICONIP'24 — https://ojs.aut.ac.nz/iconip24/2/article/view/52
- SNN microseismic on DAS (Springer 2024) — https://link.springer.com/chapter/10.1007/978-3-031-66965-1_31
- **Vasilache et al. 2025, vibration predictive-maintenance SNN (best blueprint)** — https://arxiv.org/html/2506.13416v1
- Vathakkattil Joseph & Pakrashi 2022, SNN SHM on Loihi — https://pmc.ncbi.nlm.nih.gov/articles/PMC9740015/
- Dennler et al. 2021, balanced SNN vibration anomaly — https://arxiv.org/abs/2106.00687

**Foundational seismic UGS (non-spiking; features + baselines):**
- Park, Dibazar, Berger 2009, cadence analysis biped-vs-quadruped >95% — https://ieeexplore.ieee.org/document/4959942/
- Dibazar & Berger 2007, dynamic-synapse footstep+vehicle on geophone — https://ieeexplore.ieee.org/document/4371238/
- SITEX02/ACIDS DL vehicle classification (CNN 92.05% on SITEX02) — https://www.nature.com/articles/s41598-025-01684-x
- CNN footstep detection band — https://library.seg.org/doi/10.1190/tle39090654.1

**Audio / KWS SNN recipe:**
- Bittar & Garner 2022 (adLIF baseline) — https://www.frontiersin.org/journals/neuroscience/articles/10.3389/fnins.2022.865897/full ; code https://github.com/idiap/sparch
- Yin, Corradi, Bohté 2021 (adaptive-LIF) — https://arxiv.org/abs/2103.12593
- DCLS-Delays, ICLR 2024 — https://arxiv.org/abs/2306.17670
- GPN gated parametric neuron 2024 — https://arxiv.org/html/2412.01087v1
- Pellegrini et al. SLT 2021 (conv-SNN KWS) — https://arxiv.org/abs/2011.06846
- Yılmaz et al. Interspeech 2020 — https://www.isca-archive.org/interspeech_2020/ylmaz20_interspeech.html
- SpikCommander 2025 (SOTA) — https://arxiv.org/html/2511.07883 ; LMUFormer ICLR'24 — https://arxiv.org/pdf/2402.04882
- CNN-SNN auditory conv stem (k=64) — https://arxiv.org/pdf/2307.08501

**Encoding:**
- Auge et al. 2021 encoding survey — https://link.springer.com/article/10.1007/s11063-021-10562-2
- Spike-encoding environmental-sound benchmark (TAE/SF/MW) 2025 — https://arxiv.org/html/2503.11206
- Speech2Spikes (log-mel + delta) — https://dl.acm.org/doi/fullHtml/10.1145/3584954.3584995
- Direct coding / Sigma-Delta tabular — https://arxiv.org/html/2502.10423v1 , https://arxiv.org/pdf/2210.13107

**SNN training methodology + SpikingJelly:**
- SpikingJelly (Science Advances 2023) — https://www.science.org/doi/10.1126/sciadv.adi1480 ; code https://github.com/fangwei123456/spikingjelly
- PLIF (ICCV 2021) — https://arxiv.org/abs/2007.05785
- tdBN (AAAI 2021) — https://arxiv.org/abs/2011.05280
- Surrogate-gradient (Neftci/Mostafa/Zenke 2019) — https://arxiv.org/abs/1901.09948 ; robustness (Zenke & Vogels 2021) — https://direct.mit.edu/neco/article/33/4/899/97482
- STBP (2018) — https://arxiv.org/abs/1706.02609 ; TET loss (ICLR 2022) — https://arxiv.org/abs/2202.11946
- Low-latency/T (ECCV 2022) — https://arxiv.org/abs/2110.05929 ; Eshraghian review (Proc. IEEE 2023) — https://arxiv.org/abs/2109.12894 ; direct-training review 2024 — https://arxiv.org/html/2405.04289v1

**Ordinal / multi-task / sim-to-real:**
- CORAL — https://arxiv.org/abs/1901.07884 ; CORN — https://arxiv.org/abs/2111.08851 ; code https://github.com/Raschka-research-group/coral-pytorch
- Uncertainty weighting (Kendall CVPR'18) — https://arxiv.org/abs/1705.07115 ; GradNorm — https://arxiv.org/abs/1711.02257
- Focal loss — https://arxiv.org/abs/1708.02002 ; class-balanced loss — https://arxiv.org/abs/1901.05555
- Label smoothing — https://arxiv.org/abs/1906.02629 ; mixup — https://arxiv.org/abs/1710.09412
- Domain randomization (Tobin 2017) — https://arxiv.org/abs/1703.06907 ; SpecAugment — https://arxiv.org/abs/1904.08779
- RevIN instance-norm (ICLR'22) — https://openreview.net/forum?id=cGDAkQo1C0p ; SWA — https://arxiv.org/abs/1803.05407
- SNN inherent robustness (ECCV'20) — https://arxiv.org/abs/2003.10399 ; SNN sim-to-real (event-camera) — https://arxiv.org/abs/2303.13077
- Multi-task SNN evidence: MT-SNN — https://arxiv.org/abs/2208.01522 ; MTSpark — https://arxiv.org/abs/2412.04847

**Honesty ledger (where evidence is thin):** (1) zero published SNNs do geophone human/vehicle/animal ID — the
design is transferred from vibration+audio SNNs; (2) multi-head *ordinal* SNNs are essentially unprecedented (one
multi-label vibration-SNN exists) — verify head balance empirically; (3) SNN sim-to-real is demonstrated only on
event-camera vision, none on seismic — budget a real fine-tune set; (4) the CORN-over-softmax gain at K=3 is real
but modest — justified by rank-aware error cost + calibration, not by a large accuracy jump; (5) several IEEE/paywalled
seismic figures (Dibazar 2007, Park 2009 exact numbers) are from abstracts/snippets, not full-text verification.
