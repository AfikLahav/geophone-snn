# v4.2 verification — diversity-over-calibration design (domain randomization + coverage)

*Two research-agent reports, 2026-07-07. Commissioned after the 3-for-3 empirical finding
(coupling bump, terrain-match-only, amplitude calibration: fidelity point-fixes never improved
sim→real transfer; composition changes always did) to verify the "diversity over calibration"
corpus design: which axes must be NEUTRAL, what distribution shapes, and how to GUARANTEE
coverage with normalized scene-type likelihoods.*

## Headline validations

1. **Our 3-for-3 result is the mainstream expected regime, not an anomaly.** DR literature:
   for classification tasks with appearance/nuisance-like sensor variation, randomization
   beats calibration (Tobin 2017 IROS; Peng 2018 ICRA arXiv:1710.06537; Chen 2022 ICLR
   arXiv:2110.03239 — theory incl. zero-real-sample transfer). Calibration only overtakes DR
   for precision/regression tasks (Sobanbabu 2025 CoRL arXiv:2505.14266). Frame the paper
   accordingly: the experiments *confirm the regime*, they aren't a surprise.
2. **Seismic sensor-chain DR has no precedent** — audio/ASC has mature device-response
   randomization (DIR augmentation, Freq-MixStyle: arXiv:2305.07499, 2206.12513), radar has
   physics+DR hybrids, seismic has none. Porting device-IR-style randomization to
   geophone coupling/electronics is a novel, citable contribution.

## A. Neutral axes (sensor-chain nuisances) — rules with evidence

| Axis | Shape | Anchoring |
|---|---|---|
| Gain (~50 dB span) | **uniform in dB** (log-uniform linear) | bracket the 3 measured rigs [floors 0.08 / 7.4 / 25.2 mV] + margin for unmeasured rig-4; optional curriculum narrow→wide (ADR lesson, arXiv:1910.07113) |
| ADC quantization | categorical mixture {continuous, 0.125, 0.2, 0.25 mV LSB} | rigs' LSBs + neighbors; keep an unquantized bucket |
| Clip rail | categorical/log-uniform headroom {±256, ±512 mV, never-clips} | measured rigs |
| Mains lines | Bernoulli presence × log-uniform level (0–25 dB-above-floor) × harmonics | real measurements |
| Spurious resonances | Poisson count × log-uniform freq × random Q | sparse, wide band |
| Coupling | **mixture over BOTH filter forms** (low-pass AND bump — structural DR for model-form uncertainty) × log-uniform fc ~30–150 Hz | per-surface conditioning is a REAL physical coupling — keep it, document it |

**The load-bearing rule — class-conditional independence:** every nuisance drawn
independently of the class label (and of each other unless physically coupled). Models
exploit ANY nuisance↔label correlation as a shortcut and die OOD (Geirhos 2020 Nat. Mach.
Intell.; Sagawa Group-DRO 2020 ICLR; our own ET-anti result). *Neutral = wide span AND
P(nuisance | class) flat — the flatness is auditable and is the reviewer-facing defense.*

**Pitfalls:** too-wide fixed ranges kill the learning signal (ADR; widen until training
degrades, back off); don't collapse to the 3-rig mean (that reproduces the measured
calibration failure); real system must lie inside the DR support with margin; wrong shape
(linear-uniform gain undersamples low gain — use dB-uniform).

**Two-stage deployment pattern validated:** train DR-invariant, then per-deployment
recalibrate operating point on a short unlabeled snippet — BN-statistics adaptation
(Nado 2020 arXiv:2006.10963; Schneider 2020 NeurIPS) + threshold/prior calibration. Distinct
from calibrating the *simulator* (which fails).

## B. Real-anchored axes (NOT neutral)

Propagation physics (340 banks), GRF biomechanics, speeds/masses, band structure — keep
realistic distributions. Uniformizing physics manufactures unphysical scenes; realism
belongs here, diversity belongs in the nuisance chain and the composition.

## C. Coverage guarantees (the "likelihood normalized by scene count" rule)

- **Semantic axes (class × subkind): hard floors + equal mass per type split across its
  subkinds** — mass(type)/n_subkinds(type), so a 12-subkind class doesn't outweigh a
  3-subkind class by combinatorics. This IS the normalization rule. Optional mild natural
  signal via square-root tempering (Kang 2020 ICLR arXiv:1910.09217); uniform-within-type +
  floor is the cleanest default. Train balanced; deployment priors injected at threshold
  time — provably consistent (logit adjustment, Menon 2021 ICLR arXiv:2007.07314; prior-shift
  EM, Saerens 2002 Neural Comp.). Record per-cell π_train so calibration can invert
  (Dal Pozzolo 2015).
- **Nuisance crossings (subkind × terrain × condition × SNR): do NOT floor the full grid**
  (intractable). Guarantee **2-way/3-way combinatorial coverage** (NIST covering arrays:
  2–4-way interactions trigger >90% of faults — Kuhn/NIST; Lanus 2021 SDCC) over
  (subkind × terrain-STRATUM × condition × SNR-bin), with the 340 terrains clustered into
  ~6–10 physics strata; LHS/orthogonal sampling within strata (McKay 1979).
- **Difficulty axis: ~uniform-in-dB** (or mild boundary emphasis via importance sampling
  with retained weights — O'Kelly 2018 NeurIPS rare-event AV testing), never natural
  frequency.
- **Floor sizes:** ~300–500 windows per evaluable group (subkind × SNR-bin) — CNN saturation
  evidence (Shahinfar 2020); ≥30–50 per finest audited cell; **CAP cheap cells at the same
  envelope** so easy-to-generate scenes can't dominate (Buda 2018).

## D. Audit battery (the verification the user asked for)

1. Feature→nuisance linear probe on frozen features (terrain/condition/rig): AUC ≈ chance
   (Alain & Bengio 2016).
2. Label-from-nuisance-only classifier: ≈ chance by construction (design confound check).
3. Corpus mutual information I(label; nuisance) ≈ 0 + bias-amplification ≈ 0 (Zhao 2017).
4. Worst-cell accuracy reported on a balanced eval split, not just averages (Sagawa 2020;
   Idrissi 2022 — simple balancing ≈ Group-DRO).
5. Leave-a-cell-out OOD stress (hold out a terrain stratum / condition; measure drop).
6. Per-cell count histograms ≥ floors; 2-way coverage = 100%; SDCC vs deployment ODD = 0
   (Lanus/NIST).
7. Datasheet recording generative priors/weights/floors (Gebru 2021) — enables calibration
   inversion and reviewer transparency.

## Sources

Tobin 2017 IROS · Peng 2018 arXiv:1710.06537 · OpenAI ADR arXiv:1910.07113 · Chen 2022
arXiv:2110.03239 · BayesSim arXiv:1906.01728 · BayRn arXiv:2003.02471 · SimOpt
arXiv:1810.05687 · Sobanbabu 2025 arXiv:2505.14266 · Vuong arXiv:1903.11774 · Mehta ADR
arXiv:1904.04762 · Geirhos 2020 shortcut learning · Sagawa Group-DRO arXiv:1911.08731 ·
Arjovsky IRM arXiv:1907.02893 · Prakash structured DR arXiv:1810.10093 · IR-GAN
arXiv:2010.13219 · DIR-augmentation arXiv:2305.07499 · Freq-MixStyle arXiv:2206.12513 ·
RadSimReal arXiv:2404.18150 · Nado arXiv:2006.10963 · Schneider arXiv:2006.16971 · Menon
logit adjustment arXiv:2007.07314 · Saerens 2002 · Buda arXiv:1710.05381 · Idrissi CLeaR
2022 · Dal Pozzolo 2015 · Kang arXiv:1910.09217 · Lanus/NIST 2021 SDCC · Kuhn/NIST
combinatorial coverage · O'Kelly arXiv:1811.00145 · OHEM arXiv:1604.03540 · de Gelder
arXiv:2409.01139 · McKay 1979 LHS · Shahinfar arXiv:2010.08186 · Alain & Bengio
arXiv:1610.01644 · Zhao EMNLP 2017 · REVISE arXiv:2004.07999 · Gebru Datasheets 2021
