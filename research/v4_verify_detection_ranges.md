# V4 verification — empirical seismic detection ranges & SNR (literature)

*Research agent report, 2026-07-06. Commissioned to verify the working numbers behind
the v3/v4 corpus design (R_DET per class, detection floors, SNR bands).*

## 1. Detection range table (single vertical geophone, ~4.5–10 Hz)

| Target | Reported range | Conditions / band | Source |
|---|---|---|---|
| Human, normal walk | ~30 m reliable | 4.5 Hz geophone, Yuma Proving Grounds | field-test refs via Tremor Tech / Pakhomov lineage |
| Human, walkers/joggers | 10–50 m test circles | Fort Devens; 8-person march Fort Irwin | Succi et al., SPIE 2001 (10.1117/12.441277) |
| Human, stealthy walk | ~51.5 m (quiet area, computed); undetectable at few m in urban | noise-floor limited; transfer fn peak **20–90 Hz** | Sabatier & Ekimov, SPIE 6963, 2008 (10.1117/12.785235) |
| Human (WSN system) | 10–20 m (100% @ <10 m; ~95% @ <15 m) | 250 Hz sample, adaptive threshold, wet soil | Koç & Yegin, *J. Sensors* 2013 (10.1155/2013/120386) |
| Human, generic UGS | 3–50 m | REMBASS/IREMBASS SA sensor | FAS/GlobalSecurity REMBASS spec |
| Human, MSSH | 10–30 m | soil-dependent | DSIAC UGS survey |
| Light van | 40–50 m | RMS-above-background | "Problems in seismic detection and tracking" (ResearchGate 255661983) |
| Medium truck | 60–80 m | same | same |
| Wheeled vehicle | 15–250 m | REMBASS/IREMBASS | FAS REMBASS spec |
| Tank / tracked | "several 100 m"; spec 25–350 m | heavy tracked, band ~5–25 Hz | Problems-in-seismic / REMBASS |
| Vehicle, generic | 50–100 m | MSSH, soil-dependent | DSIAC UGS survey |
| Elephant | 155.6 m (controlled); 66.9–140 m (wild); ~32 m (human-noise areas) | 10 Hz geophone 85.8 V/m/s, band ~20 Hz | Rathnayake et al., arXiv 2406.05140 / 2509.02920 |
| Horse/cattle/dog | **no direct field study found** | — | must interpolate |

## 2. Frequency bands (well-supported)

- **Human footstep:** broadband, exploited band <100 Hz; site transfer-function peak **20–90 Hz** (Sabatier & Ekimov 2008). Step-impulse cadence ~2 Hz.
- **Vehicle:** ~**5–20/25 Hz** (track-impact periodic + broadband).
- **Large animal (elephant):** mean footfall ~20–24 Hz; 20–90 Hz plausible for smaller quadrupeds (sharper, lighter impacts skew higher).

## 3. SNR / Pd / FAR — what detectors actually use

- Published seismic detectors operate on **ROC curves spanning 0–20 dB SNR**; no field paper found reporting operation at negative in-band SNR of −8 to −12 dB.
- Koç & Yegin (2013): 100% detection <10 m / <5% FAR; ~95% hit / <10% false <15 m.
- Elephant SVM (arXiv 2509.02920): 99% controlled → 70–73% in noisy natural settings.
- Quiet-vs-urban noise floor: background vibration in 5–50 Hz band ~**37 dB lower** in quiet areas than urban (Sabatier & Ekimov 2008) — dominant driver of range variance.

## 4. Surface-wave attenuation over 100–500 m

A(r) = A₀ · r^(−n) · exp(−α r), α = πf / (Q·V).
- Geometric spreading: Rayleigh n≈0.5 far-field (r^−1.1 near-field bilinear form) — PMC8115131.
- Material damping dominates: soils Q ~10–50. Vehicle at f=15 Hz, V≈200 m/s, Q≈20 → α ≈ 0.012 /m → 1/33 amplitude over 300 m from damping alone.
- Light van/car RMS-above-background: **40–80 m** on typical soil — not 300–400 m. 400 m is a tank/heavy-tracked figure.

## 5. Verdicts on our working numbers

| Working number | Verdict |
|---|---|
| Human ~50–60 m max | Optimistic but defensible as quiet-site ceiling; typical operational 20–35 m |
| Vehicle 220–400 m | **Only heavy tracked/trucks. Cars: 40–80 m. 400 m for a car is not supported.** |
| Animal ~40–50 m | Plausible interpolation; no primary source — flag as extrapolated |
| Floors −8/−12/−9 dB in-band | **Not corroborated**; field detectors run 0–20 dB. Likely reviewer challenge — justify from our own Pd curves or soften |
| Bands human/animal 20–90 Hz, vehicle 5–25 Hz | Well-supported |

## Sources

- Succi et al. (2001), SPIE — 10.1117/12.441277
- Sabatier & Ekimov (2008), SPIE 6963 69630V — 10.1117/12.785235
- Ekimov & Sabatier, ultrasonic footstep components — researchgate.net/publication/6413849
- Koç & Yegin (2013), J. Sensors — 10.1155/2013/120386
- Pakhomov et al. (2004/2005), SPIE 5403/5778 — 10.1117/12.545293
- REMBASS/IREMBASS spec — man.fas.org/dod-101/sys/land/rembass.htm
- DSIAC UGS survey — dsiac.dtic.mil/technical-inquiries/notable/unattended-ground-sensor-survey/
- Problems in seismic detection and tracking — researchgate.net/publication/255661983
- Rathnayake et al., arXiv 2406.05140, 2509.02920
- Rayleigh attenuation — pmc.ncbi.nlm.nih.gov/articles/PMC8115131/
- Arora et al. (2004), A Line in the Sand — people.eecs.berkeley.edu/~prabal/pubs/papers/arora04lites.pdf
