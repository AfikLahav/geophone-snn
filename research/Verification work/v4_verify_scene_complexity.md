# V4 verification — scene complexity: hard confusions, simulatable sources, polyphonic labeling

*Three research-agent reports, 2026-07-06. Commissioned to make the v4 corpus as
non-trivial as physically defensible: what genuinely confuses fielded seismic
systems, which elaborate sources we can simulate with force-time models, and how
to label/score overlapping-event scenes.*

---

## PART A — Documented hard confusions (single vertical 4.5 Hz geophone)

Single-channel caveat throughout: **bearing/array cues — the most reliable
multi-target discriminator in the literature — are unavailable to one geophone.**
Our corpus therefore isolates exactly the regime the field says is hardest, which
is a publishable framing, not a weakness.

**Tier 1 — genuinely ambiguous at one vertical channel (target these hardest):**
1. **Slow quadruped (horse/cattle/dog) vs single human walk** — a slow quadruped
   produces the *same fundamental gait cadence* as a human; cadence-only recognizers
   collapse. The canonical documented failure. (Park, Dibazar & Berger, IEEE ICASSP
   2009 / WO2010118233A2)
2. **2–4 humans walking together vs a single quadruped** — overlapping bipedal
   footfalls create a ~4-beat/cycle structure that aliases onto a quadruped's
   4-footfall stride. Fielded systems train "multiple people" as its own class for
   exactly this reason. (Damarla/Rutgers, IEEE 2012, Xplore 6352571)
3. **Herd / group → quasi-continuous signal** — N overlapping impulse trains wash
   the discrete periodic spikes into a noise-like continuum that starts to resemble
   a *constant source* (vehicle/machinery). (Mazumdar multi-person, RG 336754624;
   PURE arXiv:2104.07177)
4. **Distant/large vehicle vs machinery vs herd continuum** — sustained
   broadband/harmonic drive, no impulsive structure; separating feature is
   low-freq-envelope-vs-full-band ratio. (US7710264; Groos & Ritter GJI 179:1213)

**Tier 2 — environmental false-positives (the "nothing"/confuser class must absorb these):**
5. **Wind-driven vegetation** — spurious spectral peak (~55 Hz observed) + broadband
   transients that trip footstep detectors; #1 quiet-site FAR driver.
   (Johnson et al. 2019 JGR-SE 124; arXiv:2509.02920)
6. **Livestock near a fence vs intruder** — same as #1, operationally the dominant
   perimeter-UGS false alarm. (DSIAC survey; Pakhomov RG 255661983)

**Tier 3 — deception / adversarial (documented):**
7. **Stealthy/creep gait** — detection range collapses ~52 m → few m; suppresses the
   heel-strike impulse classifiers rely on, mimicking a faint/distant animal.
   (Sabatier & Ekimov 2008)
8. **Human leading/riding an animal** — superposed biped+quadruped; ICA blind-
   separation papers exist *because* the mixture is not trivially separable.
   (Sabatier et al., RG 258716286)
9. **Human moving with a vehicle / crawling** — footstep energy masked under the
   vehicle continuum; prone locomotion removes the discrete heel-strike.
   *(feature-logic inference, no dedicated seismic study — flagged gap)*

**Feature axes and fragility (what separates the confusable pairs):**
- Cadence/fundamental gait freq: separates human↔quadruped nominally — **very fragile**
  (slow quadruped aliases; primary documented failure).
- Footfalls-per-stride (2 vs 4 beats): biped↔quadruped — **moderate**; breaks when
  N≥2 humans overlap or gait transitions (walk↔trot) change the beat count.
- Footfall-interval regularity: machine↔animal↔human — **moderate**; herd overlap and
  adversarial irregular gait destroy it.
- Impact impulsiveness (kurtosis, rise time): footstep↔vehicle — **fairly robust**;
  fragile for stealth (suppressed) and herds (overlap → continuous).
- Spectral centroid / HF content: **fragile at range** — ground low-pass + attenuation
  strip the class-defining HF impulsive content beyond ~50–80 m.
- GRF shape (double vs single hump): **weak at a buried geophone** — Green's-function
  convolution smears it; recoverable only very near-field.
- Low-freq-envelope-vs-full-band ratio: impulsive↔constant — **robust** for the
  vehicle/non-vehicle split.

**Counting limit:** at one geophone, "counting" people is expected to fail around
**N≈3–4** overlapping walkers as the impulse train fills in; beyond that a scene
should be labeled "group/continuum" and becomes vehicle-confusable. Single-geophone
N-person counting is a genuinely open, publishable question.

---

## PART B — Simulatable elaborate sources (force-time ⊛ Green's function)

Our pipeline builds each source as time-varying Fz+Fx applied at moving positions,
convolved with pyprop8 GFs. Groundedness triage for candidate additions:

| Candidate | GRF data? | Verdict | Parameterization |
|---|---|---|---|
| **Human groups / marching** | Yes (pedestrian DLF models, bridge lit) | **Well-grounded** | Sum N walker force trains with a `phase_coherence ∈ [0,1]` axis: random crowd (√N growth, broadened ~2 Hz peak) → locked march (N growth, sharp harmonics). Quick-time 120 spm = 2.0 Hz, double-time 180 spm = 3.0 Hz. DLF₁≈0.4 at 2 Hz, harmonics decay to ~6% by 10th. (Birrell; MDPI Vibration 5(4):52; Adv Civ Eng 2020:9093037; Broughton/Angers bridge collapses) |
| **Loaded human (ruck/carry)** | Yes (Birrell 2007; Liew 2016 meta) | **Well-grounded** | Single `payload_mass`: vGRF rises ~10 N per +1 kg; scale Fz ~linearly by (body+load). Rifle carriage ↑impact peak + ML impulse. |
| **Herd of quadrupeds** | Via single-animal GRF superposition | Method grounded; coherence assumed | Superpose K single-animal hoof trains, independent phase, small cadence spread (low coherence). |
| **Horse + rider/pack** | Base GRF yes; load effect by analogy | Grounded base; load-scaling **inferred** | Trot ~1.0–1.2 BW/limb, gallop lead >2.5 BW; fore 57–58% of weight. Scale hoof Fz by (body+rider)/body — label the assumption. |
| **Composite vehicle scenes (idle / door slam / occupants exiting)** | Partial (weight-classification patent) | Composition sound; magnitudes estimated | Steady idle at engine-order freqs + impulsive slam transient + mass-unloading steps + new footfall trains at offsets — all superpositions of existing primitives. |
| **Hand-knee crawl** | Yes (Xu 2023, N=6) | Grounded (hand-knee) | 4 low-amplitude contact trains: hand 0.37 BW, knee 0.69 BW, no aerial phase / no sharp impact. |
| **Low-crawl (prone, military)** | No | **Speculative** | Very-low-amplitude, broadband, slow (<0.5 Hz limb cycle); label synthetic-only. |
| **Digging / tool impacts** | No | **Speculative** | Impulse train, ~10–50 ms contact, cadence 0.3–1 Hz, broadband; amplitude = engineering estimate. Would want an in-house calibration recording before any published claim. |
| **Bicycle** | Qualitative only | Distinct class, quantitatively weak | Continuous low-amplitude rolling contact (no footfall impulses) + optional weak ~1 Hz pedaling modulation — lower kurtosis than footsteps (that low impulsivity IS the discriminator). |

**Highest-value, defensible-now additions:** groups/marching (phase-coherence axis),
loaded human (mass scaling), horse+rider (mass scaling), composite vehicle scenes.
**Defer / synthetic-only label:** digging, low-crawl, quantitative bicycle.

**Validation guardrail:** the Green's function must encode correct geometric +
anelastic attenuation, or HF impulsive cues survive unrealistically far and make the
task artificially easy. (This is the same attenuation-realism point as the propagation
report.)

---

## PART C — Confusers + polyphonic labeling/scoring

**Ranked confuser table** (1 = worst quiet-site FAR driver):

| # | Confuser | Band | Character | Fools | Synth vs sample |
|---|---|---|---|---|---|
| 1 | Wind on trees (root-coupled) | 0.5–5 Hz + >100 Hz gusts | continuous gusty | human / floor | **SAMPLE (must)** — site-specific, non-stationary |
| 2 | Distant traffic hum | <25 Hz | continuous | vehicle | synth approx + sample |
| 3 | Rain drop impacts | impulsive, to >100 Hz | intermittent impulses | human | **synth** (Poisson impulse train) |
| 4 | Machinery/generator (mains) | 50/60 Hz + harmonics | narrowband tonal | vehicle | **synth** (sinusoid stack + RPM drift) |
| 5 | Aircraft/heli/drone | 10–200 Hz rotor comb | continuous tonal | vehicle | **synth** (harmonic comb + envelope) |
| 6 | Livestock/quadruped | impulsive cadenced | cadenced | animal / human | sample preferred |
| 7 | Thunder | impulsive low-freq | rare | vehicle/human | sample |
| 8 | Ocean/surf microseism | 0.05–0.2 Hz | continuous | floor | sample / colored baseline |
| 9 | Construction/percussion | broadband + impulsive | mixed | vehicle/human | synth impacts + sample plant |
| 10 | Earthquake/teleseism | <1–4 Hz | transient | floor anomaly | sample (+ time-gate out) |
| 11 | Thermal / diurnal cultural | broadband / 1–20 Hz | impulsive / slow drift | floor | sample |

We already synthesize several of these (mains lines, impulsive micro-transients,
machinery tones in `r3_noise.py`). The must-sample ones (wind-on-vegetation, thunder,
microseism, real quadruped footfalls, diurnal drift) point back to the
`datasets/FETCH_PLAN.md` real-noise fetch.

**Polyphonic labeling — the DCASE-grounded recommendation:**
1. **Multi-hot labels per window across the 3 heads.** A human+vehicle+animal scene
   has all three presence heads positive. Do NOT collapse to a single dominant class —
   that discards the polyphony the elaborate scenes exist to teach. (DCASE Task 4;
   Mesaros et al. 2016)
2. **Keep ordinal presence per head, driven by per-class SNR.** 0 = absent,
   1 = present-but-masked (SNR below floor), 2 = clearly present (SNR ≥ high gate).
   This makes masking a first-class label. Aligns exactly with the v4 detectability
   gates.
3. **Report three metric tiers:**
   - Primary: **per-class PSDS** (Bilen et al. ICASSP 2020, arXiv:1910.08440) with the
     cross-trigger cost (CTTC) — a vehicle passage lighting up the human head is scored
     as a *cross-trigger*, the exact low-FAR-discrimination quantity we care about;
     threshold-independent (our 3 heads won't share an operating point).
   - Secondary: **segment-based (3 s window) multi-label macro-F1 / ER** — honest match
     to a window-presence detector, duration-robust. (Mesaros 2016)
   - Diagnostic: per-class TPR stratified by solo-vs-co-occurring and by SNR bin —
     exposes masking-driven misses that aggregate scores hide.
4. **Cross-trigger cost = the FAR knob:** set it high for operationally dangerous
   confusions (vehicle→animal miss, wind→human) and report PSDS at a couple of settings.

**Security masking framing:** for human+vehicle, label = {human present(masked),
vehicle present}; a system that reports only "vehicle" is a **missed detection**, not a
correct dominant call. Vehicle/track energy is >10× footstep energy at close-medium
range (Damarla IOA 2007), so footsteps sit tens of dB down — which is precisely why the
per-class-SNR ordinal is the right ground truth (report human recall vs vehicle-
interference SNR).

---

## Implications for GENERATION_PLAN_V4

- Add source subkinds: human groups (phase-coherence axis), marching, loaded human,
  herd, horse+rider, composite vehicle (idle/slam/exit). Defer/flag digging, low-crawl,
  quantitative bicycle.
- Deliberately generate the Tier-1 aliasing scenes (slow quadruped @ human cadence;
  2–4 humans @ quadruped-like beat; herd continuum) as the corpus's hard core.
- Move from single-label to **multi-hot per-window** labels; adopt per-class PSDS +
  segment F1 as the metric suite; keep per-class-SNR ordinal (already central to v4).
- Confusers: extend the synthesizable set (rotor comb, richer machinery, rain rate);
  the must-sample set is the concrete justification for executing the real-noise fetch.

## Sources (consolidated)

Park/Dibazar/Berger IEEE 2009 (4959942) · Damarla IEEE 2012 (6352571), IOA 2007 29(5) ·
Sabatier & Ekimov SPIE 6963 2008 · Sabatier ICA RG 258716286 · Wijesinghe arXiv:2509.02920 ·
elephant rumble arXiv:2312.02831 · SensIT/SITEX02 arXiv:1902.09981 · PURE arXiv:2104.07177 ·
multi-person RG 336754624 · counting IEEE 7528073 · Griffin JEB 207:3545 · Heglund & Taylor
Science 186:1112 · Usman eLife 29495 · Birrell 2007 Gait&Posture (PubMed 17337189) · Liew 2016
(PubMed 27705050) · Xu 2023 crawl (PMC10067345) · MDPI Vibration 5(4):52 · Adv Civ Eng
2020:9093037 · Broughton/Angers bridge (Wikipedia) · Mad Barn equine GRF · Roland 2005 J Biomech
38:2102 · Clayton piaffe PMC7915051 · Bilen PSDS arXiv:1910.08440 · Ferroni 2021 arXiv:2010.13648 ·
Mesaros 2016 Applied Sciences 6(6):162 · DCASE Task 4 (Turpault & Serizel) · Johnson 2019 JGR-SE
124 (10.1029/2018JB017151) · Riahi & Gerstoft 2015 GRL · Wolin & McNamara 2020 BSSA 110(1):270 ·
McNamara & Buland 2004 BSSA · Surv. Geophys. 2022 (10712-022-09713-4) · Groos & Ritter GJI 179:1213
