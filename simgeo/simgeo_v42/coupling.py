"""D5 — ground-coupling resonance re-anchor (shared by the v4 generator AND snr_maps.py).

C0 measured the real rig's coupling bump (dataset_validation/coupling_verify.json): the
primary cluster (active sessions car 41.5 / human 55 Hz + paired nothing 25/47.5 Hz)
sits at ~40-55 Hz; weighted geo-mean 53 Hz. The v3 bank library centers coupling_fc at
a median of 127.3 Hz (per-scene draw profile_fc x 10^U(-0.15,0.15) in sensor.render_hp) —
i.e. ~2.4x too high, a known sim->real domain-gap contributor. v4 re-centers the
POPULATION on the measured value while preserving the firmer-ground-couples-higher
ordering (compressed) and a physically-plausible burial/soil spread.

anchor_fc replaces the CENTER of the coupling draw; render_hp's own +-0.15-decade jitter
(sensor.py:145) still rides on top as fine per-scene spread. MUST be applied identically
in the corpus render and the SNR(r) maps so the maps match the corpus.
"""
import numpy as np

FC_ANCHOR_HZ = 53.0          # C0 weighted geo-mean of real coupling bumps (primary cluster 40-55)
MEDIAN_PROFILE_FC = 127.3    # median coupling_fc over the 340 v3 banks (library_v3.json)
BETA = 0.5                   # compress profile-to-profile spread: fc ~ (profile/median)^BETA
SIGMA = 0.35                 # per-scene lognormal jitter (burial/soil), on top of render_hp's own
FC_LO, FC_HI = 30.0, 150.0   # physical span cap (real showed a 25 Hz soft-coupling bump)


def anchor_fc(profile_fc, rng):
    """Re-anchored per-scene coupling center (Hz). Population median lands on FC_ANCHOR_HZ
    (lognormal jitter has median 1); firmer-ground profiles still couple higher, compressed
    by BETA; clipped to [FC_LO, FC_HI]. Pass the RESULT to sensor.render_hp as profile_fc."""
    rel = (float(profile_fc) / MEDIAN_PROFILE_FC) ** BETA
    fc = FC_ANCHOR_HZ * rel * float(rng.lognormal(0.0, SIGMA))
    return float(np.clip(fc, FC_LO, FC_HI))
