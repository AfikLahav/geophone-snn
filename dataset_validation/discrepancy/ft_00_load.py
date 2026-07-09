"""
ft_00_load.py — shared data loading for discrepancy analysis.

Loads:
  synth_df  : DataFrame (N_synth x 132+meta) from features_v4_3s shards
  real_X    : ndarray (N_real x 132), real window features
  real_labels : list of str labels per real window ('human','vehicle','nothing')
  FEAT_NAMES : list[str] len=132
  FAMILIES   : dict family_name -> list[int] indices into FEAT_NAMES
  MODEL_FEATS: list[str] len=104, the discriminative+is_rep features from sqlite
  SCALER     : dict with 'mean', 'std', 'features' (the 104 scaler features)
"""
import sys, os, glob, json, sqlite3
import numpy as np
import pandas as pd

# ---- paths -----------------------------------------------------------------
ROOT = r"S:\ALL PROJECTS\geophone sensor\finals project\finals project"
SYNTH_GLOB = r"G:\geophone_synth\features_v4_3s\features_shard_*.parquet"
DATA_DIR   = os.path.join(ROOT, "Goephone-Project", "geophone_data")
SCALER_PATH= os.path.join(ROOT, "snn_v4_out", "E2_uniform_sampler", "scaler.json")
SQLITE_PATH= os.path.join(ROOT, "feature_analysis_v2.sqlite")

sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "simgeo_v4"))
import features as F

# ---- feature families (index ranges in FEATURE_NAMES) ----------------------
FEATURE_NAMES = F.FEATURE_NAMES
assert len(FEATURE_NAMES) == 132

FAMILIES = {
    "LEGACY32": list(range(0,  32)),
    "PHYS7":    list(range(32, 39)),
    "CAD12":    list(range(39, 51)),
    "ENG8":     list(range(51, 59)),
    "WX2":      list(range(59, 61)),
    "CEP16":    list(range(61, 77)),
    "WPE16":    list(range(77, 93)),
    "AR8":      list(range(93, 101)),
    "NL1":      list(range(101,102)),
    # v2-NEW sub-families
    "MOD7":     list(range(102,109)),
    "III5":     list(range(109,114)),
    "TONAL7":   list(range(114,121)),
    "STAT5":    list(range(121,126)),
    "HOS3":     list(range(126,129)),
    "COOC3":    list(range(129,132)),
}
# aggregate v2-NEW
FAMILIES["v2NEW30"] = list(range(102, 132))

# ---- load scaler -----------------------------------------------------------
def load_scaler():
    with open(SCALER_PATH) as fh:
        s = json.load(fh)
    return s  # keys: mean, std, clip, features (list of 104 feat names)

# ---- load 104-model-feature list ------------------------------------------
def load_model_features():
    conn = sqlite3.connect(SQLITE_PATH)
    rows = conn.execute(
        "SELECT feature FROM feature_scorecard WHERE discriminative=1 AND is_rep=1"
    ).fetchall()
    conn.close()
    return [r[0] for r in rows]

# ---- load synthetic features -----------------------------------------------
SYNTH_CLASSES = ["human", "vehicle", "nothing"]  # classes matched to real
SYNTH_SAMPLE_PER_CLASS = 8000  # stratified sample per relevant class

def load_synth(sample_per_class=SYNTH_SAMPLE_PER_CLASS, rng_seed=42):
    """
    Stratified-samples sample_per_class windows per coarse class
    (human/vehicle/nothing) from all shards.  Reads one shard at a time to
    avoid loading the full ~3.5 GB dataset.
    """
    shards = sorted(glob.glob(SYNTH_GLOB))
    if not shards:
        raise FileNotFoundError(f"No shards matched: {SYNTH_GLOB}")
    rng = np.random.default_rng(rng_seed)

    # first pass: count per class per shard so we know target per shard
    print(f"[load_synth] {len(shards)} shards - counting class distribution...")
    shard_counts = {}  # shard_idx -> {class -> count}
    for si, s in enumerate(shards):
        df_tiny = pd.read_parquet(s, columns=["coarse"])
        shard_counts[si] = {c: int((df_tiny.coarse == c).sum()) for c in SYNTH_CLASSES}

    total_per_class = {c: sum(shard_counts[si][c] for si in range(len(shards)))
                       for c in SYNTH_CLASSES}
    print(f"[load_synth] total per class: {total_per_class}")

    # compute fractional target per shard
    chunks = []
    feat_cols = FEATURE_NAMES + ["coarse"]
    for si, s in enumerate(shards):
        df_s = pd.read_parquet(s, columns=feat_cols)
        shard_chunks = []
        for c in SYNTH_CLASSES:
            sub = df_s[df_s.coarse == c]
            n_shard = shard_counts[si][c]
            if n_shard == 0:
                continue
            # fractional target for this shard
            frac = n_shard / max(total_per_class[c], 1)
            n_take = max(1, int(round(sample_per_class * frac)))
            n_take = min(n_take, len(sub))
            idx = rng.choice(len(sub), size=n_take, replace=False)
            shard_chunks.append(sub.iloc[idx])
        if shard_chunks:
            chunks.append(pd.concat(shard_chunks, ignore_index=True))

    df = pd.concat(chunks, ignore_index=True)
    # report actual sample
    actual = {c: int((df.coarse == c).sum()) for c in SYNTH_CLASSES}
    print(f"[load_synth] sampled {len(df)} rows: {actual}")
    return df

# ---- compute real features -------------------------------------------------
CSV_LABEL_MAP = {
    "human.csv":       "human",
    "car.csv":         "vehicle",
    "human_nothing.csv": "nothing",
    "car_nothing.csv":   "nothing",
}

def load_real():
    """Returns (X_real, labels_real) arrays."""
    SCENE = 30_000
    STRIDE = 1_500
    all_feat = []
    all_labels = []
    for fname, label in CSV_LABEL_MAP.items():
        fpath = os.path.join(DATA_DIR, fname)
        if not os.path.exists(fpath):
            print(f"[WARN] not found: {fpath}")
            continue
        df_raw = pd.read_csv(fpath, low_memory=False)
        # CSV has columns: time_s, amplitude  (or just one numeric column)
        if 'amplitude' in df_raw.columns:
            raw = df_raw['amplitude'].values.astype(np.float64)
        else:
            # fallback: last numeric column
            raw = df_raw.iloc[:, -1].values.astype(np.float64)
        raw = raw * 25.4  # pipeline amplitude convention
        n = len(raw)
        print(f"[load_real] {fname}: {n} samples ({n/F.FS:.1f}s) label={label}")
        # chunk into SCENE-length scenes
        for sc_start in range(0, n - SCENE + 1, SCENE):
            seg = raw[sc_start: sc_start + SCENE]
            pre = F.scene_precompute(seg)
            for i0 in range(0, len(seg) - F.NW + 1, STRIDE):
                try:
                    fv = F.window_features(pre, i0)
                    all_feat.append(fv)
                    all_labels.append(label)
                except Exception as e:
                    print(f"  [WARN] window_features failed i0={i0}: {e}")
        # handle remainder scene if >= 2*NW
        remainder_start = (n // SCENE) * SCENE
        if n - remainder_start >= 2 * F.NW:
            seg = raw[remainder_start:]
            pre = F.scene_precompute(seg)
            for i0 in range(0, len(seg) - F.NW + 1, STRIDE):
                try:
                    fv = F.window_features(pre, i0)
                    all_feat.append(fv)
                    all_labels.append(label)
                except Exception as e:
                    pass

    X = np.array(all_feat, dtype=np.float64)
    labels = np.array(all_labels)
    print(f"[load_real] total real windows: {len(X)}, label dist: "
          f"{dict(zip(*np.unique(labels, return_counts=True)))}")
    return X, labels


if __name__ == "__main__":
    sc = load_scaler()
    print("scaler features:", len(sc["features"]))
    mf = load_model_features()
    print("model features:", len(mf))
    df = load_synth()
    print("synth shape:", df.shape)
    X, L = load_real()
    print("real shape:", X.shape)
