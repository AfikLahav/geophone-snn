# localization_1d/ — the pyprop8 localization testbed (Track 2)

The **cheap, exact, CPU** track: develop and validate the localization *methods* on flat
layered soil (pyprop8) before committing to the expensive 3-D house sim. Reuses the existing
generator pipeline (`simgeo/.../gfbank_build.py`, `scenes.py emit()`). ~10⁶× cheaper than the
house sim; the method *rankings* transfer even though absolute accuracy won't.

- `01_methods_and_protocol.md` — the method bake-off (MFP, SO-TDOA, RSS, polarization bearing,
  time-reversal, likelihood-surface fusion) + the experimental protocol + the blind-velocity rigor.
- `02_testbed_architecture.md` — the concrete build: 3-component GF banks, the 2-D N-sensor
  scene harness, the localizer modules, the GDOP / sensor-count Monte-Carlo, CRLB validation.
- `03_pyprop8_validity.md` — verification that 1-D pyprop8 is scientifically sound for this
  (real dispersion at 3–20 m; the key limitation = soil-Rayleigh, not slab-Lamb).

Execution plan: [`../ENGINEERING_ROADMAP.md`](../ENGINEERING_ROADMAP.md) §3 (E2.1–E2.4).
