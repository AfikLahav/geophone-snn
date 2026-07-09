# V4 verification — sub-threshold positives in detection ML (literature)

*Research agent report, 2026-07-06. Commissioned to verify the v4 supervision contract:
how mature detection fields handle present-but-undetectable positives, and what SNR
sampling distributions synthetic corpora should use.*

## Executive summary

1. **Sub-floor positives are never pooled as ordinary positives in evaluation.** Mature fields either evaluate detection as a function of SNR/distance, or impose an explicit lower SNR cutoff on the signal class. Pooling undetectable positives into one AUROC is a recognized bias that caps performance near chance — exactly the v3 symptom.
2. **The dominant convention is Pd-vs-SNR (efficiency) at fixed FAR**, not a scalar; scalars are integrals of that curve against a physical prior.
3. **Synthetic corpora:** uniform-in-dB is the speech default, but the state of the art **over-samples near/below the detection threshold** (triangular / half-normal skewed to low SNR) because uniform-in-SNR under-exposes the decision boundary.

## Domain evidence

### Gravitational-wave ML (closest analog: injections at known SNR)
- Nagarajan & Messenger (2025), arXiv:2501.13846 — lower bound on injected optimal SNR at **5**: "placing the lower bound for optimal SNR at 5 reduces the bias due to class overlap"; "a uniform distribution for SNR … biased the network against learning low SNR samples." Training density deliberately skewed toward low SNR (half-normal/beta/truncated-triangular variants).
- MLGWSC-1 (Schäfer et al., PRD 107, 023021, 2023; arXiv:2209.11146) — community metric: **sensitive distance vs FAR** on injection-free background; injections sampled uniform in *chirp distance* (importance-samples the detectable/threshold region, not uniform in volume).
- Koloniari et al. (2025), arXiv:2509.05283 — sensitive distance more stable than raw counts; FAR points 1/10/100 per month; background must be decontaminated of real signals before FAR estimation.
- NSBH classifier (PLB, arXiv:2210.15888) — ROC reported at SNR = 6, 8, 10 vs fixed FAP.

### Radar / sonar (origin of the conventions)
- Canonical object: **Pd-vs-SNR at fixed Pfa (CFAR)**; canonical scalar: minimum detectable SNR for Pd = 0.8/0.9. Examples: Lin-DBSCAN-CFAR (PMC12031068), U-Net staring radar Pd 0.97 @ Pfa 0.01 @ 10 dB (IET rsn2.12383), CNN vs CFAR ~0.5 dB at low SNR (EUSIPCO 2021).

### Seismology phase picking
- PhaseNet (Zhu & Beroza, GJI 2019), EQTransformer (Mousavi et al., Nat. Comm. 2020), STEAD — positives are **analyst-picked arrivals**, an implicit detectability gate baked into labeling; explicit noise class; per-event SNR metadata enables stratified eval. Synthetic pipelines that know physical presence below the pickable floor must add the gate that analyst labeling provides for free.

### Speech (VAD / KWS at low SNR)
- Multi-condition training: SNR **uniform in dB**, ranges 0–15/0–20/0–30 dB.
- Saeed et al. (Apple), ICASSP 2021, arXiv:2102.09666 — learned per-instance difficulty ("data parameters") → automatic curriculum easy→hard without SNR labels.
- Soft VAD (arXiv:1909.11886) — frame-wise soft presence posteriors instead of hard 0/1 on marginal frames.

### DCASE / PU learning
- DCASE 2023 Task 4B (soft labels): annotator agreement (soft-label value) degrades with SNR/polyphony. DCASE 2024 Task 4 (arXiv:2406.08056): "missing labels" protocol ≈ PU framing.
- Object detection as PU (Yang et al., BMVC 2020): missing label ≠ negative.
- Robust PU (Zhao et al., KDD 2023, arXiv:2308.00279): ad-hoc thresholding of weak positives into negatives destabilizes training; use easy-first + hardness measures.

## Recommendations adopted for v4

**(a) Evaluation — three-zone detectability-gated scheme:**
- Detectable (SNR ≥ τ_hi): headline AUROC/Pd. τ_hi from the idealized/realizable detector's Pd≈0.9 point at the operating FAR.
- Marginal (τ_lo ≤ SNR < τ_hi): scored separately as the Pd-vs-SNR curve; not folded into headline AUROC.
- Sub-floor (SNR < τ_lo): **excluded from eval positives** (not recast as negatives — they are unlabeled in the PU sense; recasting corrupts the FAR distribution).
- Always publish the full Pd-vs-SNR curve at fixed FAR + the gate definition — this neutralizes the "cherry-picked easy positives" reviewer objection.

**(b) Corpus SNR sampling:**
- Not uniform-in-distance (wastes mass on the undetectable tail — v3's failure), not uniform-in-SNR (biases against the boundary — GW evidence).
- **Triangular/half-normal density peaking just below τ_hi, floored at τ_lo** for hard positives; controlled sub-floor/zero tail retained as negatives/unlabeled; log per-window SNR metadata (STEAD practice).
- Optional: data-parameters difficulty weighting or loud→quiet curriculum.

## Key citations

arXiv:2501.13846 · arXiv:2209.11146 / PRD 107.023021 · arXiv:2509.05283 · arXiv:2210.15888 · arXiv:2102.09666 · arXiv:1909.11886 · DCASE 2023 Task 4B · arXiv:2406.08056 · Yang et al. BMVC 2020 · arXiv:2308.00279 · Zhu & Beroza GJI 2019 · Mousavi et al. Nat. Comm. 2020 / STEAD · IET rsn2.12383 · PMC12031068 · EUSIPCO 2021 (Yavuz)

*Caveat: triangular-distribution parameters (min 3 / mode 5 / max 27.5) and the exact speech dB ranges came via search summaries / HTML fetches; re-verify against primary PDFs before quoting them numerically in the paper.*
