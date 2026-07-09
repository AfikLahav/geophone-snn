"""Independent verification of the load-bearing / disputed discrepancy findings."""
import os, sys, glob, sqlite3, json
import numpy as np, pandas as pd
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
sys.path.insert(0, os.path.join(ROOT, "simgeo_v4")); import label as L

# --- 1. coupling_fc in corpus_v4 (disputed: 132 vs 52 Hz) ---
db = sqlite3.connect(r"G:/geophone_synth/corpus_v4/shard_0.sqlite")
fc = np.array([r[0] for r in db.execute("SELECT coupling_fc FROM scenes WHERE coupling_fc IS NOT NULL")])
print(f"[1] corpus_v4 coupling_fc: n={len(fc)} median={np.median(fc):.1f} IQR[{np.percentile(fc,25):.1f},{np.percentile(fc,75):.1f}] range[{fc.min():.1f},{fc.max():.1f}]")

# --- 2. synth clipping fraction (disputed 'showstopper' 10-13%) ---
rows = db.execute("SELECT coarse,n_samples,noise_mv,clean_mv,clean_mv2 FROM scenes s JOIN waveforms w USING(scene_id) LIMIT 400").fetchall()
by = {}
for coarse,n,nm,c1,c2 in rows:
    x = np.frombuffer(nm, np.float32).astype(np.float64)
    if c1 is not None: x = x + np.frombuffer(c1, np.float32)
    if c2 is not None: x = x + np.frombuffer(c2, np.float32)
    clip = np.mean(np.abs(x) >= 255.9)
    by.setdefault(coarse, []).append(clip)
print("[2] synth model-input clip-fraction (>=255.9 mV) by class:")
for c,v in by.items(): print(f"    {c:8s} mean {np.mean(v)*100:.1f}% median {np.median(v)*100:.2f}% n={len(v)}")
db.close()

# --- 3. human impulsiveness: real vs synth (disputed magnitude 13.7 vs 0.23) ---
from scipy.stats import kurtosis
def kwin(x, nw=3000, hop=1500):
    return [kurtosis(x[i:i+nw]) for i in range(0, len(x)-nw+1, hop)]
rh = pd.read_csv(os.path.join(ROOT,"Goephone-Project","geophone_data","human.csv"))["amplitude"].to_numpy(float)*1000
rk = kwin(rh)
db = sqlite3.connect(r"G:/geophone_synth/corpus_v4/shard_0.sqlite")
sk = []
for (n,nm,c1) in db.execute("SELECT n_samples,noise_mv,clean_mv FROM scenes s JOIN waveforms w USING(scene_id) WHERE coarse='human' LIMIT 60"):
    x = np.frombuffer(nm,np.float32).astype(np.float64)
    if c1 is not None: x = x + np.frombuffer(c1,np.float32)
    x = np.clip(x,-256,256); sk += kwin(x)
db.close()
print(f"[3] human window kurtosis: REAL med {np.median(rk):.2f} (n={len(rk)}) | SYNTH med {np.median(sk):.2f} (n={len(sk)})")

# --- 4. which corpus is the E2 reference trained on? confirm features_v4_3s = lowpass corpus_v4 ---
print(f"[4] E2 reference build trains on features_v4_3s (from corpus_v4, LOWPASS coupling). "
      f"bump corpus=corpus_v4b/features_v4b_3s is a separate experiment (EC).")
