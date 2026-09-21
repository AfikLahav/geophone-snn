"""Which of the 340 synthetic terrains best matches the REAL data — separately for the human
session and the car session (different rigs/sites). Resolves the '1 real terrain vs 340
synthetic' confound in the discrepancy report: instead of pooling all terrains, find the
nearest terrain(s) and check whether real falls INSIDE the synthetic terrain envelope.

Match axis = SIGNAL spectral SHAPE (terrain propagation shapes it), amplitude-independent:
per-class post-sensor band-fraction vector over [5-25, 20-55, 55-90, 90-180] Hz.
  REAL: median SIGNAL-EXCESS PSD of active windows (active minus paired-nothing floor).
  SYNTH: median PSD of the pure per-class clean_mv blob (no floor confound), per profile.
Reports top-k profiles/class + whether each real band-fraction is inside the [p5,p95]
across-profile synthetic envelope.
"""
import os, sys, glob, sqlite3, json
import numpy as np, pandas as pd
from scipy.signal import welch

_GEO_ROOT = __import__("os").environ.get("GEO_SYNTH_ROOT", __import__("os").path.join(__import__("os").path.dirname(__import__("os").path.abspath(__file__)), "..", "..", "geophone_synth"))
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
FS = 1000.0
BANDS = [(5, 25), (20, 55), (55, 90), (90, 180)]
BLAB = ["5-25", "20-55", "55-90", "90-180"]
REAL_DIR = os.path.join(ROOT, "Goephone-Project", "geophone_data")
NW, HOP = 3000, 1500
PER_PROFILE = 8


def psd(x):
    f, p = welch(np.asarray(x, float) - np.mean(x), fs=FS, nperseg=min(2048, len(x)))
    return f, p


def bandfrac(f, p):
    v = np.array([p[(f >= lo) & (f < hi)].sum() for lo, hi in BANDS])
    return v / (v.sum() + 1e-30)


def real_excess_bands(active_csv, floor_csv):
    ra = pd.read_csv(os.path.join(REAL_DIR, active_csv))["amplitude"].to_numpy(float) * 1000
    rf = pd.read_csv(os.path.join(REAL_DIR, floor_csv))["amplitude"].to_numpy(float) * 1000
    # median PSD over windows
    def medpsd(a):
        ps = []
        for i in range(0, len(a) - NW + 1, HOP):
            f, p = psd(a[i:i + NW]); ps.append(p)
        return f, np.median(np.stack(ps), 0)
    f, pa = medpsd(ra); _, pf = medpsd(rf)
    exc = np.clip(pa - pf, 0, None)                          # signal excess over the rig+terrain floor
    return bandfrac(f, exc), bandfrac(f, pa)                 # (excess, raw-active)


# ---- real per session ----
real = {"human": real_excess_bands("human.csv", "human_nothing.csv"),
        "car": real_excess_bands("car.csv", "car_nothing.csv")}
print("REAL band-fractions [5-25,20-55,55-90,90-180]:")
for k, (exc, raw) in real.items():
    print(f"  {k:6s} excess {np.round(exc,3)} | raw-active {np.round(raw,3)}")

# ---- synthetic per profile ----
shards = sorted(glob.glob(os.path.join(_GEO_ROOT, "dataset_v431/shard_*.sqlite")))
prof = {}                                                    # profile -> {class: [bandfrac,...]}
meta = {}                                                    # profile -> (family, vs)
need = {"human": "human", "car": "vehicle"}
for sh in shards:
    db = sqlite3.connect(sh)
    for realk, coarse in need.items():
        rows = db.execute(
            "SELECT s.profile_id,s.family,s.terrain_vs,w.clean_mv,w.clean_cls FROM scenes s "
            "JOIN waveforms w USING(scene_id) WHERE s.coarse=? AND s.target_snr_db>6 AND w.clean_mv IS NOT NULL",
            (coarse,)).fetchall()
        for pid, fam, vs, cm, cls in rows:
            d = prof.setdefault(pid, {}); lst = d.setdefault(realk, [])
            if len(lst) >= PER_PROFILE: continue
            x = np.frombuffer(cm, np.float32)
            if len(x) < NW: continue
            i0 = (len(x) - NW) // 2
            f, p = psd(x[i0:i0 + NW]); lst.append(bandfrac(f, p)); meta[pid] = (fam, float(vs))
    db.close()
    if all(len(prof.get(p, {}).get(k, [])) >= PER_PROFILE for p in meta for k in need) and len(meta) > 300:
        break

# per-profile median bandfrac
pmed = {k: {} for k in need}
for pid, d in prof.items():
    for k in need:
        if d.get(k): pmed[k][pid] = np.median(np.stack(d[k]), 0)

# ---- match + envelope ----
out = {"real": {k: {"excess": real[k][0].round(4).tolist(), "raw": real[k][1].round(4).tolist()} for k in real},
       "bands": BLAB, "matches": {}, "envelope": {}}
for k in need:
    rvec = real[k][0]                                        # use signal-excess
    dists = {pid: float(np.sqrt(((v - rvec) ** 2).sum())) for pid, v in pmed[k].items()}
    top = sorted(dists, key=dists.get)[:8]
    allv = np.stack(list(pmed[k].values()))
    lo, hi = np.percentile(allv, 5, 0), np.percentile(allv, 95, 0)
    inside = [(lo[i] <= rvec[i] <= hi[i]) for i in range(len(BANDS))]
    out["matches"][k] = [{"profile": p, "family": meta[p][0], "vs_ms": round(meta[p][1]),
                          "dist": round(dists[p], 4), "bands": pmed[k][p].round(3).tolist()} for p in top]
    out["envelope"][k] = {"real": rvec.round(3).tolist(),
                          "synth_p5": lo.round(3).tolist(), "synth_p95": hi.round(3).tolist(),
                          "inside": [bool(x) for x in inside], "n_profiles": len(pmed[k])}
    print(f"\n=== {k.upper()} : best-matching terrains (by signal band-shape) ===")
    for m in out["matches"][k][:6]:
        print(f"  {m['profile']:20s} {m['family']:14s} vs={m['vs_ms']:4} dist={m['dist']:.3f} bands={m['bands']}")
    print(f"  real bands {rvec.round(3)} | synth env p5 {lo.round(3)} p95 {hi.round(3)}")
    print(f"  real INSIDE synth envelope per band {BLAB}: {inside}  "
          f"({'ALL inside' if all(inside) else 'OUTSIDE on '+','.join(BLAB[i] for i in range(4) if not inside[i])})")

json.dump(out, open(os.path.join(HERE, "TERRAIN_MATCH.json"), "w"), indent=1)
print("\nwrote TERRAIN_MATCH.json")
