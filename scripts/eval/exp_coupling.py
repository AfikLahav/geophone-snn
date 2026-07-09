"""Experiment: does the coupling MODEL (not just fc) cause the upper-human-band deficit?

The domain analysis found v4 synthetic humans carry 21.6% of their 20-90 Hz energy above
55 Hz vs 46.6% for real. Hypothesis: coupling_response is a 2nd-order LOW-PASS, so anchoring
fc to the measured ~53 Hz rolls off the 55-90 Hz footstep band. A real geophone is flat above
its 4.5 Hz corner and the ground-coupling resonance is a PEAK perturbation, not a low-pass.

Test 3 coupling variants on rendered human scenes (isolating coupling by reapplying H):
  LP53  : current 2nd-order low-pass, anchored fc ~53 Hz (v4 baseline)
  BUMP53: resonance PEAK on flat unity (spurious_bump form), fc ~53 Hz  -> preserves upper band
  LP_HI : low-pass but fc kept high (~110 Hz, ~v2-like)                  -> control
Compare each variant's upper-band fraction (55-90 / 20-90 Hz) of the human clean signal to the
real target 0.466. Fast (no full render).
"""
import os, sys, json
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, os.path.join(ROOT, "simgeo_v4"))
import numpy as np
import sources_v2 as src, sensor, r3_noise, label
from scenes import Bank
from coupling import anchor_fc
import snr_maps as SM

REAL_UPPER = 0.466
FS = 1000.0
PROFILES = ["soft_soil_0", "loess_2", "gravel_0", "clay_1", "dirt_road_0", "sand_4",
            "wet_soil_binv_3", "kurkar_0", "rock_0", "terra_rossa_1"]


def coupling_lowpass(n, fc, qc, fs=FS):
    return sensor.coupling_response(n, fc, qc, fs)


def coupling_bump(n, fc, qc, fs=FS, gain_db=None):
    """Resonance PEAK on flat unity response (spurious_bump form): =1 away from fc, bump at fc.
    Preserves energy above fc. gain_db ~ the coupling amplification at resonance."""
    A = 10 ** ((gain_db if gain_db is not None else 6.0) / 20.0)
    f = np.fft.rfftfreq(n, 1 / fs); fr = np.maximum(f, 1e-9)
    detune = qc * (fr / fc - fc / fr)
    return 1.0 + (A - 1.0) / np.sqrt(1.0 + detune ** 2)


def band_frac(x, lo, hi, tot=(20, 90)):
    X = np.fft.rfft(x); f = np.fft.rfftfreq(len(x), 1 / FS); p = np.abs(X) ** 2
    return float(p[(f >= lo) & (f < hi)].sum() / (p[(f >= tot[0]) & (f < tot[1])].sum() + 1e-30))


def human_ground(pid, rng):
    bank = Bank.get(pid)
    path, tg, dur = SM._stationary_path(4.0, dur=12.0)
    em, _ = src.human(path, tg, rng, bank.n, gait="walk")
    return SM._assemble320(em, bank, dur), bank


rows = {"LP53": [], "BUMP53": [], "LP_HI": []}
for pid in PROFILES:
    try:
        rng = np.random.default_rng(hash(pid) % 2**32)
        v_sig, bank = human_ground(pid, rng)
        n = len(v_sig)
        fc = anchor_fc(bank.meta["coupling_fc"], rng); qc = bank.meta["coupling_q"]
        geo = sensor.geophone_response(n, 4.5, 0.6, 28.8)
        spur = sensor.spurious_bump(n, 150.0, 20.0, 12.0)
        for name, Hc in (("LP53", coupling_lowpass(n, fc, qc)),
                         ("BUMP53", coupling_bump(n, fc, qc)),
                         ("LP_HI", coupling_lowpass(n, max(fc, 110.0), qc))):
            clean = np.fft.irfft(np.fft.rfft(v_sig) * Hc * geo * spur, n=n)
            rows[name].append(band_frac(clean, 55, 90))
    except Exception as e:
        print(f"{pid}: {type(e).__name__}: {e}")

print(f"REAL upper-band (55-90/20-90 Hz) target: {REAL_UPPER:.3f}\n")
res = {"real_target": REAL_UPPER, "variants": {}}
for name, vals in rows.items():
    m = float(np.median(vals))
    res["variants"][name] = {"upper_band_frac": round(m, 3), "n": len(vals),
                             "gap_to_real": round(m - REAL_UPPER, 3)}
    print(f"  {name:7s} upper-band frac {m:.3f}  (gap to real {m-REAL_UPPER:+.3f})")
best = min(res["variants"], key=lambda k: abs(res["variants"][k]["gap_to_real"]))
res["closest_to_real"] = best
print(f"\nclosest to real: {best}")
json.dump(res, open(os.path.join(HERE, "exp_coupling.json"), "w"), indent=1)
print("wrote exp_coupling.json")
