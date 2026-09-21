"""Load synthetic feature tables and prepare model-specific training tensors."""
import glob
import json
import os
import sqlite3
import sys
import time

import numpy as np
import torch

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sys.path.insert(0, os.path.join(ROOT, "simgeo", "simgeo_v42"))
import features as F                                            # noqa: E402  (132-name bank)

SYNTH_ROOT = os.environ.get("GEO_SYNTH_ROOT", os.environ.get("GEO_DB_ROOT", os.path.join(ROOT, "..", "geophone_synth")))
HEADS = ("human", "animal", "vehicle")
LVL = {"none": 0, "single": 1, "multiple": 2}

# Group boundaries in FEATURE_NAMES order, checked against the actual bank. Used by the
# cumulative ablation -- "did each family of features earn its place".
GROUPS = [
    ("legacy32", 0, 32), ("phys7", 32, 39), ("cad12", 39, 51), ("eng8", 51, 59),
    ("wx2", 59, 61), ("cep16", 61, 77), ("wpe16", 77, 93), ("ar8", 93, 101),
    ("nl1", 101, 102), ("v2new30", 102, 132),
]

# The seven ratio features whose scaling collapses. They divide a band's energy by the window's
# total loudness, which is exactly zero in about half a percent of windows, so the value reaches
# 1e9 and mean-and-spread scaling flattens every ordinary window to a constant. The information
# is real -- rank-based screening rates all seven discriminative, because ranks ignore the
# outlier magnitude -- so the fix is the scaling, not deletion.
RATIO_FEATURES = ["frac_wind_1_5", "frac_veh_5_25", "frac_foot_20_90", "frac_high_90_180",
                  "ratio_veh_foot", "hop_band_frac", "veh_foot_simultaneity"]

META = ["split", "profile_id", "family", "subkind", "coarse", "terrain_vs", "common_snr",
        "noise_condition", "scene_id", "t0",
        "human_level", "human_soft", "human_snr", "human_occ",
        "vehicle_level", "vehicle_soft", "vehicle_snr", "vehicle_occ",
        "animal_level", "animal_soft", "animal_snr", "animal_occ"]

WINDOW_DIRS = {
    "3s": "features_v431_3s", "1s": "features_v431_1s", "2s": "features_v431_2s",
    "5s": "features_v431_5s", "8s": "features_v431_8s", "10s": "features_v431_10s",
    "3s_bl200": "features_v431_bl200_3s",   # matched-band arm: input decimated to 200 Hz and back
}


def context_index(keys, order, K, step=1.0, tol=1e-3):
    """(N, K) row indices: for row i, the K rows ending at i that belong to the same stream
    (`keys` equal) and follow each other in `order` by `step`. Missing history repeats the
    earliest available row, so a stream's first window sees itself K times."""
    keys = np.asarray(keys); order = np.asarray(order, dtype=np.float64)
    n = len(keys)
    prev = np.arange(n)
    same = (keys[1:] == keys[:-1]) & (np.abs(order[1:] - order[:-1] - step) < tol)
    prev[1:][same] = np.arange(n - 1)[same]
    S = np.empty((n, K), dtype=np.int64)
    S[:, K - 1] = np.arange(n)
    for k in range(K - 2, -1, -1):
        S[:, k] = prev[S[:, k + 1]]
    return S


def _restrict(S_all, mask):
    """Re-index a full-table context index to the rows selected by `mask`."""
    full_to_sub = np.full(len(mask), -1, dtype=np.int64)
    full_to_sub[np.flatnonzero(mask)] = np.arange(int(mask.sum()))
    S = full_to_sub[S_all[mask]]
    if (S < 0).any():
        raise RuntimeError("context window crosses the split boundary")
    return S


class SeqView:
    """An (N, K, F) view over an (N, F) table through an (N, K) index; gathered on access.

    Provides the tensor operations used by the trainer and scorer: shape, len, device, indexing
    by slice, tensor, or numpy array, and .cpu().numpy() (which returns a CPU view)."""

    def __init__(self, X, S):
        self.X, self.S = X, S

    @property
    def shape(self):
        return (self.S.shape[0], self.S.shape[1], self.X.shape[1])

    @property
    def device(self):
        return self.X.device

    def __len__(self):
        return self.S.shape[0]

    def __getitem__(self, idx):
        if isinstance(idx, np.ndarray):
            idx = torch.as_tensor(idx, device=self.S.device)
        return self.X[self.S[idx]]

    def cpu(self):
        return SeqView(self.X.cpu(), self.S.cpu())

    def numpy(self):
        return self


def select_features(spec):
    """Turn a feature specification into a list of names, in bank order."""
    names = list(F.FEATURE_NAMES)
    if isinstance(spec, (list, tuple)):          # an explicit list, in bank order
        return [n for n in names if n in set(spec)]
    if spec == "all132":
        return names
    if spec == "sel104":
        # the historical selection: screened on v2 data, kept here as a comparison cell rather
        # than as the default, because nothing has re-checked its drop list on v4.3.1
        db = os.path.join(ROOT, "feature_analysis_v2.sqlite")
        con = sqlite3.connect(db)
        sel = [r[0] for r in con.execute(
            "SELECT feature FROM feature_scorecard WHERE discriminative=1 AND is_rep=1")]
        con.close()
        return [n for n in names if n in set(sel)]
    if spec.startswith("cumulative:"):
        upto = spec.split(":", 1)[1]
        end = None
        for nm, a, b in GROUPS:
            if nm == upto:
                end = b
        if end is None:
            raise ValueError(f"unknown feature group {upto!r}")
        return names[:end]
    if isinstance(spec, (list, tuple)):
        return list(spec)
    raise ValueError(f"unknown feature spec {spec!r}")


class RankScaler:
    """Maps each feature onto its own rank, fitted on training rows only.

    Standard mean-and-spread scaling is destroyed by a heavy tail: one window in two hundred with
    zero loudness sends a ratio feature to 1e9, the spread becomes 5e7, and every ordinary window
    lands on the same value. A rank transform is immune to that, and rank is also exactly what the
    feature screening measured when it rated those features useful."""

    def __init__(self, n_q=1024):
        self.n_q = n_q
        self.edges = None

    def fit(self, X):
        qs = np.linspace(0.0, 1.0, self.n_q)
        self.edges = np.stack([np.quantile(X[:, j], qs) for j in range(X.shape[1])], 1)
        return self

    def transform(self, X):
        out = np.empty_like(X, dtype=np.float32)
        for j in range(X.shape[1]):
            out[:, j] = np.searchsorted(self.edges[:, j], X[:, j]).astype(np.float32)
        out /= (self.n_q - 1)
        return (out * 2.0 - 1.0).astype(np.float32)          # centre on zero, span [-1, 1]

    def to_json(self):
        return {"kind": "rank", "n_q": self.n_q, "edges": self.edges.tolist()}


class StandardScaler:
    """Mean and spread, fitted on training rows only, clipped -- what every existing model used."""

    def __init__(self, clip=8.0):
        self.clip, self.mu, self.sd = clip, None, None

    def fit(self, X):
        self.mu = X.mean(0)
        self.sd = X.std(0) + 1e-8
        return self

    def transform(self, X):
        return np.clip((X - self.mu) / self.sd, -self.clip, self.clip).astype(np.float32)

    def to_json(self):
        return {"kind": "standard", "clip": self.clip,
                "mean": self.mu.tolist(), "std": self.sd.tolist()}


class SyntheticData:
    """The resident synthetic tables, loaded once and shared by every run.

    Feature selection and scaling are per-run and cheap, so the expensive part -- reading 1.9 GB
    of columnar files -- happens exactly one time.
    """

    def __init__(self, window="3s", device="cuda", verbose=True, data_dir=None):
        import pandas as pd
        d = data_dir or os.path.join(SYNTH_ROOT, WINDOW_DIRS[window])
        shards = sorted(glob.glob(os.path.join(d, "features_shard_*.parquet")))
        if not shards:
            raise FileNotFoundError(f"no feature shards under {d}")
        t0 = time.time()
        names = list(F.FEATURE_NAMES)
        parts = [pd.read_parquet(s, columns=names + META) for s in shards]
        df = pd.concat(parts, ignore_index=True)
        del parts
        self.window = window
        self.device = device
        self.names = names
        self.X_raw = df[names].to_numpy(np.float32)
        self.is_train = (df["split"] == "train").to_numpy()
        self.is_val = (df["split"] == "val").to_numpy()

        self.coarse = df["coarse"].to_numpy()
        self.family = df["family"].to_numpy()
        self.subkind = df["subkind"].to_numpy()
        self.profile = df["profile_id"].to_numpy()
        self.common_snr = df["common_snr"].to_numpy(np.float32)
        self.scene_id = df["scene_id"].to_numpy()
        self.t0 = df["t0"].to_numpy(np.float32)

        self.lvl, self.soft, self.snr, self.occ = {}, {}, {}, {}
        for h in HEADS:
            self.lvl[h] = df[f"{h}_level"].map(LVL).fillna(df[f"{h}_level"]).to_numpy(np.int64)
            self.soft[h] = df[f"{h}_soft"].to_numpy(np.float32)
            self.snr[h] = df[f"{h}_snr"].to_numpy(np.float32)
            self.occ[h] = df[f"{h}_occ"].to_numpy(np.float32)
        del df
        if verbose:
            print(f"[data] {window}: {len(self.X_raw):,} windows "
                  f"(train {self.is_train.sum():,} / val {self.is_val.sum():,}) "
                  f"in {time.time() - t0:.0f}s, {self.X_raw.nbytes / 2**30:.2f} GiB")

    # -- per-run views ---------------------------------------------------

    def prepare(self, cfg):
        """Feature subset, scaler and label tensors for one configuration, moved to the card.

        Animal handling is a data decision, not a model one:
          2head_neg       -- animal scenes stay, but count as negatives for both heads
          2head_noanimal  -- animal scenes are dropped from training entirely
        """
        names = select_features(cfg.get("features", "all132"))
        idx = [self.names.index(n) for n in names]
        X = self.X_raw[:, idx]

        scaler = (RankScaler() if cfg.get("scaling") == "rank" else StandardScaler())
        scaler.fit(X[self.is_train])
        Xz = scaler.transform(X)

        tr = self.is_train.copy()
        if cfg.get("heads") in ("2head_noanimal", "separate2", "1head_human", "1head_human_noveh"):
            tr &= ~(self.coarse == "animal")
        if cfg.get("heads") == "1head_human_noveh":
            # the model never meets a vehicle at all -- tests whether vehicle knowledge is what
            # keeps the person head from firing on low-frequency ground motion
            tr &= ~(self.coarse == "vehicle")

        dev = self.device
        K = int(cfg.get("context", 1) or 1)
        Xtr_t = torch.as_tensor(Xz[tr]).to(dev)
        Xva_t = torch.as_tensor(Xz[self.is_val]).to(dev)
        if K > 1:
            # sequence models see the K consecutive windows of the same scene ending at the
            # current one; the table is gathered on access, so nothing is stored K times
            S_all = context_index(self.scene_id, self.t0, K, step=1.5)
            Xtr_t = SeqView(Xtr_t, torch.as_tensor(_restrict(S_all, tr)).to(dev))
            Xva_t = SeqView(Xva_t, torch.as_tensor(_restrict(S_all, self.is_val)).to(dev))
        out = {
            "features": names,
            "scaler": scaler,
            "context": K,
            "Xtr": Xtr_t,
            "Xva": Xva_t,
            "labels": {},
            "meta_val": {
                "coarse": self.coarse[self.is_val],
                "family": self.family[self.is_val],
                "subkind": self.subkind[self.is_val],
                "common_snr": self.common_snr[self.is_val],
                "scene_id": self.scene_id[self.is_val],
                "t0": self.t0[self.is_val],
            },
        }
        heads = ["human", "animal", "vehicle"]
        if cfg.get("heads") in ("2head_neg", "2head_noanimal", "separate2"):
            heads = ["human", "vehicle"]
        if cfg.get("heads") in ("1head_human", "1head_human_noveh"):
            heads = ["human"]
        if cfg.get("heads") == "4way":
            heads = []              # one mutually-exclusive output instead of separate heads
        for h in heads:
            out["labels"][h] = {
                "tr_lvl": torch.as_tensor(self.lvl[h][tr]).to(dev),
                "tr_soft": torch.as_tensor(self.soft[h][tr]).to(dev),
                "tr_snr": torch.as_tensor(self.snr[h][tr]).to(dev),
                "va_lvl": torch.as_tensor(self.lvl[h][self.is_val]).to(dev),
                "va_soft": torch.as_tensor(self.soft[h][self.is_val]).to(dev),
                "va_snr": torch.as_tensor(self.snr[h][self.is_val]).to(dev),
            }
        if cfg.get("heads") == "plus_any":
            any_tr = np.maximum.reduce([self.lvl[h][tr] for h in HEADS])
            any_va = np.maximum.reduce([self.lvl[h][self.is_val] for h in HEADS])
            soft_tr = np.maximum.reduce([self.soft[h][tr] for h in HEADS])
            soft_va = np.maximum.reduce([self.soft[h][self.is_val] for h in HEADS])
            out["labels"]["any"] = {
                "tr_lvl": torch.as_tensor((any_tr > 0).astype(np.int64)).to(dev),
                "tr_soft": torch.as_tensor(soft_tr).to(dev),
                "tr_snr": torch.as_tensor(np.zeros_like(soft_tr)).to(dev),
                "va_lvl": torch.as_tensor((any_va > 0).astype(np.int64)).to(dev),
                "va_soft": torch.as_tensor(soft_va).to(dev),
                "va_snr": torch.as_tensor(np.zeros_like(soft_va)).to(dev),
            }
        if cfg.get("heads") == "4way":
            m = {"nothing": 0, "human": 1, "vehicle": 2, "animal": 3, "mixed": 1}
            ytr = np.array([m.get(c, 0) for c in self.coarse[tr]], dtype=np.int64)
            yva = np.array([m.get(c, 0) for c in self.coarse[self.is_val]], dtype=np.int64)
            out["labels"]["fourway"] = {
                "tr_lvl": torch.as_tensor(ytr).to(dev),
                "va_lvl": torch.as_tensor(yva).to(dev),
            }
        out["heads"] = list(out["labels"].keys())
        return out

    def background_mask_val(self):
        """Validation windows with no source of any kind -- where thresholds are calibrated.

        Real data never touches calibration, which is the whole point of the protocol."""
        m = np.ones(self.is_val.sum(), dtype=bool)
        for h in HEADS:
            m &= (self.lvl[h][self.is_val] == 0)
        return m
