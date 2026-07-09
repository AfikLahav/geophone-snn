# V4 verification — propagation model validity & range envelope (literature)

*Research agent report, 2026-07-06. Commissioned to verify whether the 320 m cutoff
should be raised (v3 set vehicle R_DET = 400 m against a 320 m physics cutoff) and
what extending Green's-function banks to 500 m would cost.*

## Q1. Validity of a 1D layered elastic model at 300–500 m, 5–90 Hz

- **Measured near-surface Q:** Meng, Ben-Zion & Johnson (2021, SRL 92(4)) derive **Q = 3–40 up to 150 Hz from car-traffic signals on geophone lines** — the exact source type we simulate (osti.gov/pages/biblio/1814784). Parolai et al. (2022, J. Seismol., 10.1007/s10950-021-10066-5): unconsolidated-soil Qs ~5–50 with large scatter; crosshole studies show frequency dependence (J. Appl. Geophys. 2015).
- **What 1D misses:** a large share of measured surface-wave attenuation is **scattering from lateral heterogeneity** (GJI 1989 98:183); fractures/lithology cause localized attenuation anomalies (EPS 2016, 10.1186/s40623-016-0487-0); topography scatters high-f Rayleigh waves (J. Appl. Geophys. 2018).
- Scale check: at 30 Hz, Vs 150–250 m/s → λ_R ≈ 5–8 m, so 300–500 m = **50–100 wavelengths**: phase-coherent 1D fidelity is gone; what survives is **amplitude/spectral envelope statistics**.
- Field evidence: Altmann (2004, JSV 273:713–740) — heavy-vehicle seismic amplitude decays faster than 1/r with strong positional variation; signal shape varies strongly with position. Traffic-vibration decay exponents "strongly dependent on site and frequency" (Soils & Foundations 2020).

**Bottom line:** layered half-space + causal Azimi t\* is defensible to 300–500 m only as an *envelope/spectrum statistics generator*, and only with per-realization heterogeneity proxies (randomized Q/t\*, site-response and decay-exponent jitter). Otherwise systematically over-coherent/optimistic at long range — a reviewer target.

## Q2. Measured vehicle detection ceiling

- Ekimov & Sabatier (2008): human footsteps ~52 m quiet-site ceiling (calibration anchor).
- **DARPA SensIT / SITEX02** (AAV tracked + Dragon Wagon heavy wheeled): geophone nodes at **~20–100 m** from the road — the canonical vehicle-seismic ML dataset never tested beyond ~100 m.
- ACIDS (ARL): CPA distances not confirmable from open sources — treat as unverified.
- Altmann (2004): heavy vehicles >100 m; **tracked seismic >10× stronger than wheeled** (track-slap harmonics) — the only class plausibly detectable at several hundred meters.
- Industry UGS practice: vehicles 50–100 m, soil-dependent.
- Riahi & Gerstoft (2015, GRL 42:2674): 5200-geophone array — cars studied <30 m; trains at 15/155 m; ordinary vehicles not trackable at long range.
- **Noise baseline for 10–100 Hz:** use Wolin & McNamara (2020, BSSA 110(1):270–278) high-frequency baselines, not Peterson NLNM (stops at 10 Hz).
- Sanity: Q=20, v_R=300 m/s, f=30 Hz → intrinsic attenuation ≈ **−43 dB per 100 m** on top of spreading.

**Bottom line:** no published measurement supports light-wheeled seismic detection beyond ~100–200 m in soil; heavy wheeled ~100–300 m; heavy tracked on competent ground is the only 300–500 m case. **The 320 m cutoff already sits at/above the supported envelope for most classes.**

## Q3. Cost of extending banks 320 m → 500 m

From pyprop8 `_core.py` and our `simgeo_v3/gfbank_build.py`:
- Fixed trapezium wavenumber stencil; Bessel term O(nk·nr) per frequency; ghost-avoidance constraint `nk ≥ 0.239·kmax·Vp_max·T` with `T = r_max/vr + coda`; measured cost linear in nk; kmax set by slowest Rayleigh at 250 Hz, independent of r_max.
- Total cost ≈ (T_new/T_old)²: soft profile (v_R 150 m/s) ≈ **1.6×**; stiff (1.5 km/s) ≈ **1.15×**. Radii count grows log(r_max): cap 48 → ~52.
- The existing Q-horizon gate `r_max = clip(6.9·q·vr_kms/(π·25), 0.06, 0.32)` means unclipping to 0.50 only rebuilds profiles with q·vr_kms ≥ ~5.7 (Q=30 needs v_R ≥ 190 m/s) — soft profiles stay attenuation-capped and rebuild nothing.

**Bottom line:** raising the clip to 0.50 km is cheap (≤1.6× on the affected stiff/high-Q subset) and physically defensible precisely where 1D modeling is most valid. Not needed for cars — only if heavy tracked at long range is in scope.

## Recommendations

1. Do **not** chase light/medium wheeled detections beyond the measured ~100–200 m envelope; keep the 320 m cutoff as the honest default.
2. If heavy tracked on stiff ground is in scope: raise Q-horizon clip 0.32 → 0.50 km (delta rebuild of the high-Q·v_R subset only).
3. Beyond ~200 m in soft soils: add heterogeneity-proxy augmentation and frame long-range synthetic data as envelope statistics (cite Meng 2021, Parolai 2022, Altmann 2004).
4. Use Wolin & McNamara (2020) baselines for the 5–90 Hz noise/SNR budget in the paper.

## Sources

Meng, Ben-Zion & Johnson 2021 SRL (OSTI 1814784) · Parolai et al. 2022 J. Seismol. · GJI 1989 98:183 · EPS 2016 · J. Appl. Geophys. 2018 (topography), 2015 (crosshole Q) · Altmann 2004 JSV 273:713 · Ekimov & Sabatier 2008 SPIE 6963 · Riahi & Gerstoft 2015 GRL · SensIT/SITEX02 (Butler program overview; arXiv:1902.09981) · Wolin & McNamara 2020 BSSA · Soils & Foundations 2020 · O'Toole & Woodhouse 2011 GJI 187:1516
