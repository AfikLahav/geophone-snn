"""Derive detectability gates (tau_lo / tau_hi per class) from the THEORETICAL matched-filter
threshold: the in-band SNR where a matched filter reaches d' = 2.33 (Pd~50% @ Pfa 1%) and
d' = 3.6 (Pd~90% @ Pfa 1%) for a 3 s window and the class diagnostic bandwidth.

Model-independent by construction (no circularity with any trained detector) and needs no
data. Same closed form as detectability_curve.py::theo_thr. Consumed by the trainer
(GEO_GATED_EVAL / GEO_ORDINAL_MASK / GEO_BALANCED_SAMPLER), gated_rescore.py and
eval_v3_compare.py. See GENERATION_PLAN_V4.md §1.
"""
import os, sys, json
import numpy as np

BANDW = {"human": 70.0, "vehicle": 20.0, "animal": 70.0}  # diagnostic band width (Hz), detectability_curve.py
DPRIME = {"tau_lo_db": 2.33, "tau_hi_db": 3.6}            # Pd50 / Pd90 at Pfa 1%

# T from CLI: `--T 10` (else 3 s). Gates shift by -10*log10(T/3) vs the 3 s reference
# (matched-filter integration gain: longer window -> lower detection floor).
T = 3.0
if "--T" in sys.argv:
    T = float(sys.argv[sys.argv.index("--T") + 1])


def theo_thr(B, dprime, t=None):
    """SNR_dB where matched-filter d' = dprime (detectability_curve.py:27), for window t s."""
    return 10 * np.log10(dprime ** 2 / (2 * (t or T) * B))


gates = {c: {k: round(float(theo_thr(B, d)), 2) for k, d in DPRIME.items()} | {"band_hz": B}
         for c, B in BANDW.items()}
HERE = os.path.dirname(os.path.abspath(__file__))
# canonical 3 s file stays gates.json (trainer + generator read it); every T also writes gates_<T>s.json
tag = f"{T:g}s"
targets = [os.path.join(HERE, f"gates_{tag}.json")]
if abs(T - 3.0) < 1e-9:
    targets.append(os.path.join(HERE, "gates.json"))
for out in targets:
    json.dump({"note": f"theoretical matched-filter gates; d'=2.33->Pd50, 3.6->Pd90 @ Pfa 1%; T={T:g} s",
               "T_s": T, "classes": gates}, open(out, "w"), indent=1)
for c, v in gates.items():
    print(f"{c:8s} tau_lo {v['tau_lo_db']:6.2f} dB | tau_hi {v['tau_hi_db']:6.2f} dB | band {v['band_hz']:.0f} Hz")
print("T =", T, "-> wrote", ", ".join(os.path.basename(t) for t in targets))
