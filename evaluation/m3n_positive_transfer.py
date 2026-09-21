"""Positive-only M3N-VC adaptation with the synthetic 200 Hz training set retained.

M3N-VC has vehicle passes and GPS distance, but no verified quiet class.  Real windows are used as
vehicle positives only when they are continuous three-second recordings and their independently
measured 5-25 Hz level is at least 6 dB above the recording reference.  Synthetic windows remain
half of every batch and preserve both positive and negative targets for the human and vehicle
heads.  A whole M3N scene is held out in each fold.

All thresholds are recalibrated after fine-tuning from synthetic background windows at a 0.1%
false-alarm operating point.  No M3N window is used to choose a threshold.

Compute is O(E * B * T * P) per fit and stored scores are O(N * R), where E is the number of
steps, B the batch size, T=16 spiking steps, P the model parameter count, N=2,311 valid real
windows, and R is the number of model/fold conditions.
"""

from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
import math
import os
import sqlite3
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
from spikingjelly.activation_based import functional, neuron


HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "simgeo" / "simgeo_v42"))

import features as FEATURE_MODULE  # noqa: E402
from training import data, model as MODEL, evaluate as score, train as train_one
from evaluation import configs, prepare_real


AUDIT_DB = Path(os.environ.get("M3N_AUDIT_DB", ROOT / "m3n_gps_audit.sqlite"))
CAMPAIGN_DB = Path(os.environ.get("GEO_CAMPAIGN_DB", ROOT / "campaign_out" / "campaign.sqlite"))
VEHICLE_X = Path(os.environ.get("GEO_REAL_CACHE", str(ROOT / "real_cache"))) / "vehicles_X.npy"
VEHICLE_ROWS = Path(os.environ.get("GEO_REAL_CACHE", str(ROOT / "real_cache"))) / "vehicles_rows.json"
DEFAULT_OUT = Path(os.environ.get("M3N_OUT", ROOT / "m3n_out")) / "m3n_positive_transfer.sqlite"
DEFAULT_MODEL_DIR = Path(os.environ.get("M3N_OUT", ROOT / "m3n_out")) / "models"

FAMILIES = {
    "SNN-S": "Z1_snn_S",
    "SNN-M": "Z2_snn_M",
    "SNN-L": "Z3_snn_L",
}
SCENES = ("a06", "h08", "h24", "i29", "s31")
DISTANCE_BINS = (
    ("0-10 m", 0.0, 10.0),
    ("10-20 m", 10.0, 20.0),
    ("20-40 m", 20.0, 40.0),
    ("40-80 m", 40.0, 80.0),
    ("80+ m", 80.0, math.inf),
)


def log(message):
    print(f"[{time.strftime('%H:%M:%S')}] {message}", flush=True)


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as fh:
        for block in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def auc(y, values):
    from scipy.stats import rankdata

    y = np.asarray(y, bool)
    values = np.asarray(values, float)
    n1 = int(y.sum())
    n0 = int((~y).sum())
    if not n1 or not n0:
        return None
    ranks = rankdata(values)
    return float((ranks[y].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def build_net(cfg, n_features, device, weights):
    net = MODEL.build(cfg, n_features).to(device)
    functional.set_step_mode(net, "m")
    if cfg.get("kind", "snn") == "snn" and device == "cuda":
        try:
            functional.set_backend(net, "cupy", instance=neuron.ParametricLIFNode)
        except Exception as exc:
            log(f"CuPy backend unavailable, using torch backend: {exc}")
    net.load_state_dict(torch.load(weights, map_location=device))
    return net


def model_scores(net, X, heads, device, batch=8192):
    net.eval()
    out = {head: [] for head in heads}
    with torch.no_grad():
        for start in range(0, X.shape[0], batch):
            functional.reset_net(net)
            logits = net(X[start : start + batch])
            for head in heads:
                out[head].append(torch.sigmoid(logits[head][:, 0]).float().cpu().numpy())
    return {head: np.concatenate(parts) for head, parts in out.items()}


def calibrate_thresholds(net, prepared, background_indices, heads, device):
    idx = torch.as_tensor(background_indices, dtype=torch.long, device=device)
    values = model_scores(net, prepared["Xva"][idx], heads, device)
    return {head: float(np.quantile(values[head], 0.999)) for head in heads}


def decide(scores, thresholds, heads):
    n = len(next(iter(scores.values())))
    result = np.full(n, "nothing", dtype=object)
    best = np.full(n, -np.inf)
    for head in heads:
        active = scores[head] > thresholds[head]
        take = active & (scores[head] > best)
        result[take] = head
        best[take] = scores[head][take]
    return result


def fine_tune(
    cfg, prepared, X_real, heads, source_weights, device, steps, batch, lr,
    synthetic_share, seed,
):
    torch.manual_seed(seed)
    np.random.seed(seed)
    net = build_net(cfg, prepared["Xtr"].shape[-1], device, source_weights)
    net.train()
    ema = copy.deepcopy(net)
    for parameter in ema.parameters():
        parameter.requires_grad_(False)
    optimizer = torch.optim.AdamW(
        net.parameters(), lr=lr, weight_decay=float(cfg.get("weight_decay", 1e-4))
    )
    generator = torch.Generator(device=device).manual_seed(seed)
    n_syn_batch = int(round(batch * synthetic_share))
    n_syn_batch = min(max(n_syn_batch, 1), batch - 1)
    n_real_batch = batch - n_syn_batch
    X_syn = prepared["Xtr"]
    labels = prepared["labels"]
    n_syn = X_syn.shape[0]
    n_real = X_real.shape[0]
    history = []
    t0 = time.time()

    for step in range(steps):
        syn_idx = torch.randint(0, n_syn, (n_syn_batch,), generator=generator, device=device)
        real_idx = torch.randint(0, n_real, (n_real_batch,), generator=generator, device=device)
        xb = torch.cat([X_syn[syn_idx], X_real[real_idx]], dim=0)
        losses = []
        functional.reset_net(net)
        outputs = net(xb)
        for head in heads:
            real_value = 1.0 if head == "vehicle" else 0.0
            real_soft = torch.full((n_real_batch,), real_value, device=device)
            real_lvl = real_soft.to(torch.int64)
            lvl = torch.cat([labels[head]["tr_lvl"][syn_idx], real_lvl])
            soft = torch.cat([labels[head]["tr_soft"][syn_idx], real_soft])
            losses.append(
                train_one.head_loss(
                    outputs[head], lvl, soft, cfg.get("ordinal_loss", "conditional")
                )
            )
        loss = sum(losses) + net.gate_penalty()
        if not torch.isfinite(loss):
            raise RuntimeError(f"loss became non-finite at step {step + 1}")
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
        optimizer.step()
        with torch.no_grad():
            for ema_parameter, parameter in zip(ema.parameters(), net.parameters()):
                ema_parameter.mul_(0.99).add_(parameter.detach(), alpha=0.01)
            for ema_buffer, buffer in zip(ema.buffers(), net.buffers()):
                ema_buffer.copy_(buffer)
        if step == 0 or (step + 1) % 50 == 0 or step + 1 == steps:
            history.append((step + 1, float(loss.detach().cpu()), time.time() - t0))
    ema.eval()
    return ema, history, time.time() - t0


def load_real_data():
    X = np.load(VEHICLE_X)
    rows = json.loads(VEHICLE_ROWS.read_text(encoding="utf-8"))
    con = sqlite3.connect(AUDIT_DB)
    cur = con.execute(
        "SELECT window_id,recording,scene,terrain,selected_label,gps_distance_m,"
        "band_excess_db,analysis_included FROM windows ORDER BY window_id"
    )
    columns = [d[0] for d in cur.description]
    audit = [dict(zip(columns, row)) for row in cur]
    con.close()
    if not (len(X) == len(rows) == len(audit)):
        raise RuntimeError("M3N feature, row, and audit counts differ")
    for i, (row, check) in enumerate(zip(rows, audit)):
        if row["recording"] != check["recording"] or check["window_id"] != i:
            raise RuntimeError(f"M3N row order mismatch at {i}")
    eligible = np.array(
        [bool(row["analysis_included"]) and row["band_excess_db"] >= 6.0 for row in audit]
    )
    valid = np.array([bool(row["analysis_included"]) for row in audit])
    scene = np.array([row["scene"] for row in audit])
    terrain = np.array([row["terrain"] for row in audit])
    distance = np.array([row["gps_distance_m"] for row in audit], np.float64)
    excess = np.array([row["band_excess_db"] for row in audit], np.float64)
    selected = np.array([row["selected_label"] for row in audit])
    return X.astype(np.float32, copy=False), audit, eligible, valid, scene, terrain, distance, excess, selected


SCHEMA = """
CREATE TABLE IF NOT EXISTS metadata (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS experiments (
    experiment_id TEXT PRIMARY KEY,
    run_id TEXT NOT NULL,
    family TEXT NOT NULL,
    seed INTEGER NOT NULL,
    heldout_scene TEXT NOT NULL,
    status TEXT NOT NULL,
    n_train_real INTEGER NOT NULL,
    n_test_signal INTEGER NOT NULL,
    steps INTEGER NOT NULL,
    batch_size INTEGER NOT NULL,
    learning_rate REAL NOT NULL,
    synthetic_share REAL NOT NULL,
    parameter_count INTEGER,
    wall_seconds REAL,
    source_weights TEXT NOT NULL,
    adapted_weights TEXT,
    started_utc TEXT NOT NULL,
    finished_utc TEXT,
    error TEXT
);
CREATE TABLE IF NOT EXISTS training_membership (
    experiment_id TEXT NOT NULL,
    window_id INTEGER NOT NULL,
    role TEXT NOT NULL,
    PRIMARY KEY(experiment_id, window_id)
);
CREATE TABLE IF NOT EXISTS loss_history (
    experiment_id TEXT NOT NULL,
    step INTEGER NOT NULL,
    loss REAL NOT NULL,
    elapsed_seconds REAL NOT NULL,
    PRIMARY KEY(experiment_id, step)
);
CREATE TABLE IF NOT EXISTS thresholds (
    experiment_id TEXT NOT NULL,
    condition TEXT NOT NULL,
    head TEXT NOT NULL,
    threshold REAL NOT NULL,
    PRIMARY KEY(experiment_id, condition, head)
);
CREATE TABLE IF NOT EXISTS window_scores (
    experiment_id TEXT NOT NULL,
    condition TEXT NOT NULL,
    window_id INTEGER NOT NULL,
    human_score REAL NOT NULL,
    vehicle_score REAL NOT NULL,
    human_on INTEGER NOT NULL,
    vehicle_on INTEGER NOT NULL,
    decision TEXT NOT NULL,
    PRIMARY KEY(experiment_id, condition, window_id)
);
CREATE INDEX IF NOT EXISTS ix_m3n_scores_window ON window_scores(window_id);
CREATE TABLE IF NOT EXISTS metrics (
    experiment_id TEXT NOT NULL,
    condition TEXT NOT NULL,
    subset TEXT NOT NULL,
    terrain TEXT NOT NULL,
    n_windows INTEGER NOT NULL,
    vehicle_recall REAL,
    human_head_rate REAL,
    human_without_vehicle_rate REAL,
    human_vehicle_both_rate REAL,
    human_decision_rate REAL,
    mean_vehicle_score REAL,
    mean_human_score REAL,
    PRIMARY KEY(experiment_id, condition, subset, terrain)
);
CREATE TABLE IF NOT EXISTS distance_metrics (
    experiment_id TEXT NOT NULL,
    condition TEXT NOT NULL,
    distance_bin TEXT NOT NULL,
    n_windows INTEGER NOT NULL,
    vehicle_recall REAL,
    human_head_rate REAL,
    PRIMARY KEY(experiment_id, condition, distance_bin)
);
CREATE TABLE IF NOT EXISTS retention_metrics (
    experiment_id TEXT NOT NULL,
    condition TEXT NOT NULL,
    dataset TEXT NOT NULL,
    metric TEXT NOT NULL,
    value REAL,
    n INTEGER,
    PRIMARY KEY(experiment_id, condition, dataset, metric)
);
"""


def prepare_output(path, args):
    path.parent.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(path, timeout=60)
    con.execute("PRAGMA journal_mode=WAL")
    con.execute("PRAGMA synchronous=NORMAL")
    con.executescript(SCHEMA)
    meta = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "protocol": (
            "positive-only M3N adaptation with "
            f"{100 * args.synthetic_share:g}% synthetic replay"
        ),
        "real_positive_rule": "continuous 3 s window and 5-25 Hz excess >= 6 dB",
        "fold_rule": "leave one complete M3N scene out",
        "threshold_rule": "99.9th percentile of synthetic validation background",
        "window": "3s_bl200",
        "sampling_rate_hz": "200",
        "steps": str(args.steps),
        "batch_size": str(args.batch),
        "learning_rate": str(args.lr),
        "synthetic_share": str(args.synthetic_share),
        "audit_database": str(AUDIT_DB),
        "audit_database_sha256": sha256(AUDIT_DB),
        "vehicle_features": str(VEHICLE_X),
        "vehicle_features_sha256": sha256(VEHICLE_X),
    }
    con.executemany(
        "INSERT OR REPLACE INTO metadata(key,value) VALUES (?,?)", meta.items()
    )
    con.commit()
    return con


def record_scores_and_metrics(
    con,
    experiment_id,
    condition,
    heldout,
    scores,
    thresholds,
    audit,
    valid,
    eligible,
    scene,
    terrain,
    distance,
    excess,
    selected,
):
    held_valid = valid & (scene == heldout)
    indices = np.flatnonzero(held_valid)
    local_scores = {head: values[indices] for head, values in scores.items()}
    decisions = decide(local_scores, thresholds, ("human", "vehicle"))
    human_on = local_scores["human"] > thresholds["human"]
    vehicle_on = local_scores["vehicle"] > thresholds["vehicle"]
    con.executemany(
        "INSERT OR REPLACE INTO window_scores VALUES (?,?,?,?,?,?,?,?)",
        [
            (
                experiment_id,
                condition,
                int(window_id),
                float(local_scores["human"][j]),
                float(local_scores["vehicle"][j]),
                int(human_on[j]),
                int(vehicle_on[j]),
                str(decisions[j]),
            )
            for j, window_id in enumerate(indices)
        ],
    )
    for head, value in thresholds.items():
        con.execute(
            "INSERT OR REPLACE INTO thresholds VALUES (?,?,?,?)",
            (experiment_id, condition, head, value),
        )

    subsets = {
        "all_valid": held_valid,
        "signal_ge6": held_valid & eligible,
        "signal_ge12": held_valid & (excess >= 12.0),
        "selected_near_signal_ge6": held_valid & eligible & (selected == "vehicle"),
    }
    for subset_name, subset_mask in subsets.items():
        for terrain_name in ["ALL"] + sorted(set(terrain[subset_mask].tolist())):
            mask = subset_mask & ((terrain == terrain_name) if terrain_name != "ALL" else True)
            ids = np.flatnonzero(mask)
            if not len(ids):
                continue
            hon = scores["human"][ids] > thresholds["human"]
            von = scores["vehicle"][ids] > thresholds["vehicle"]
            decision_values = decide(
                {head: values[ids] for head, values in scores.items()},
                thresholds,
                ("human", "vehicle"),
            )
            con.execute(
                "INSERT OR REPLACE INTO metrics VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
                (
                    experiment_id,
                    condition,
                    subset_name,
                    terrain_name,
                    len(ids),
                    float(von.mean()),
                    float(hon.mean()),
                    float((hon & ~von).mean()),
                    float((hon & von).mean()),
                    float((decision_values == "human").mean()),
                    float(scores["vehicle"][ids].mean()),
                    float(scores["human"][ids].mean()),
                ),
            )

    signal_mask = held_valid & eligible
    for label, low, high in DISTANCE_BINS:
        mask = signal_mask & (distance >= low) & (distance < high)
        ids = np.flatnonzero(mask)
        if not len(ids):
            continue
        hon = scores["human"][ids] > thresholds["human"]
        von = scores["vehicle"][ids] > thresholds["vehicle"]
        con.execute(
            "INSERT OR REPLACE INTO distance_metrics VALUES (?,?,?,?,?,?)",
            (
                experiment_id,
                condition,
                label,
                len(ids),
                float(von.mean()),
                float(hon.mean()),
            ),
        )


def record_retention(con, experiment_id, condition, net, prepared, thresholds, elbit, val_indices, heads, device):
    X_elbit, labels = elbit
    elbit_scores = model_scores(net, X_elbit, heads, device)
    prediction = decide(elbit_scores, thresholds, heads)
    quiet = labels == "nothing"
    values = {
        "accuracy_raw": (float((prediction == labels).mean()), len(labels)),
        "human_auc_vs_quiet": (
            auc((labels[(labels == "human") | quiet] == "human"), elbit_scores["human"][(labels == "human") | quiet]),
            int(((labels == "human") | quiet).sum()),
        ),
        "vehicle_auc_vs_quiet": (
            auc((labels[(labels == "vehicle") | quiet] == "vehicle"), elbit_scores["vehicle"][(labels == "vehicle") | quiet]),
            int(((labels == "vehicle") | quiet).sum()),
        ),
    }
    for name, (value, n) in values.items():
        con.execute(
            "INSERT OR REPLACE INTO retention_metrics VALUES (?,?,?,?,?,?)",
            (experiment_id, condition, "elbit_bl200", name, value, n),
        )

    val_tensor = prepared["Xva"][torch.as_tensor(val_indices, dtype=torch.long, device=device)]
    val_scores = model_scores(net, val_tensor, heads, device)
    for head in heads:
        truth = prepared["labels"][head]["va_lvl"].detach().cpu().numpy()[val_indices] > 0
        value = auc(truth, val_scores[head])
        con.execute(
            "INSERT OR REPLACE INTO retention_metrics VALUES (?,?,?,?,?,?)",
            (
                experiment_id,
                condition,
                "synthetic_val_sample",
                f"{head}_auc",
                value,
                len(val_indices),
            ),
        )


def export_summaries(con, output_dir):
    output_dir.mkdir(parents=True, exist_ok=True)
    queries = {
        "fold_metrics.csv": "SELECT * FROM metrics ORDER BY experiment_id,condition,subset,terrain",
        "distance_metrics.csv": "SELECT * FROM distance_metrics ORDER BY experiment_id,condition,distance_bin",
        "retention_metrics.csv": "SELECT * FROM retention_metrics ORDER BY experiment_id,condition,dataset,metric",
        "experiments.csv": "SELECT * FROM experiments ORDER BY family,seed,heldout_scene",
    }
    for filename, query in queries.items():
        cur = con.execute(query)
        fields = [d[0] for d in cur.description]
        with (output_dir / filename).open("w", newline="", encoding="utf-8") as fh:
            writer = csv.writer(fh)
            writer.writerow(fields)
            writer.writerows(cur.fetchall())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--families", nargs="+", choices=tuple(FAMILIES), default=list(FAMILIES))
    parser.add_argument("--seeds", nargs="+", type=int, choices=(1, 2, 3), default=[1, 2, 3])
    parser.add_argument("--folds", nargs="+", choices=SCENES, default=list(SCENES))
    parser.add_argument("--steps", type=int, default=600)
    parser.add_argument("--batch", type=int, default=2048)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--synthetic-share", type=float, default=0.5)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL_DIR)
    parser.add_argument("--no-save-models", action="store_true")
    args = parser.parse_args()
    args.output = args.output.resolve()
    args.model_dir = args.model_dir.resolve()
    args.model_dir.mkdir(parents=True, exist_ok=True)
    if not 0.0 < args.synthetic_share < 1.0:
        parser.error("--synthetic-share must be between 0 and 1")

    log(f"device={args.device}; families={args.families}; seeds={args.seeds}; folds={args.folds}")
    X_raw, audit, eligible, valid, scenes, terrains, distances, excess, selected = load_real_data()
    log(
        f"M3N windows: {len(X_raw):,}; continuous={int(valid.sum()):,}; "
        f"training-positive={int(eligible.sum()):,}"
    )
    elbit_raw, elbit_rows, _ = prepare_real.cached("elbit_bl200")
    elbit_labels = np.array([row["label"] for row in elbit_rows])
    con = prepare_output(args.output, args)

    log("loading the 200 Hz matched synthetic feature table")
    synthetic = data.SyntheticData(window="3s_bl200", device=args.device)
    background_indices = np.flatnonzero(synthetic.background_mask_val())
    val_rng = np.random.default_rng(20260920)
    val_indices = np.sort(
        val_rng.choice(int(synthetic.is_val.sum()), size=min(50000, int(synthetic.is_val.sum())), replace=False)
    )

    campaign = sqlite3.connect(CAMPAIGN_DB.as_uri() + "?mode=ro", uri=True)
    for family in args.families:
        prefix = FAMILIES[family]
        for seed in args.seeds:
            run_id = f"{prefix}_rep{seed}"
            cfg = next(item for item in configs.RUNS if item["run_id"] == run_id)
            run_row = campaign.execute("SELECT params FROM runs WHERE run_id=?", (run_id,)).fetchone()
            source_weights = Path(os.environ.get("GEO_CAMPAIGN_OUT", str(ROOT / "campaign_out"))) / "models" / run_id / "model_ema.pt"
            if not source_weights.exists():
                raise FileNotFoundError(source_weights)
            log(f"preparing {run_id}")
            prepared = synthetic.prepare(cfg)
            heads = prepared["heads"]
            if heads != ["human", "vehicle"]:
                raise RuntimeError(f"{run_id}: expected human and vehicle heads, got {heads}")
            feature_indices = [FEATURE_MODULE.FEATURE_NAMES.index(name) for name in prepared["features"]]
            X_scaled = prepared["scaler"].transform(X_raw[:, feature_indices])
            X_real_all = torch.as_tensor(np.asarray(X_scaled, np.float32), device=args.device)
            elbit_scaled = prepared["scaler"].transform(elbit_raw[:, feature_indices])
            X_elbit = torch.as_tensor(np.asarray(elbit_scaled, np.float32), device=args.device)

            baseline = build_net(cfg, prepared["Xtr"].shape[-1], args.device, source_weights)
            baseline_thresholds = calibrate_thresholds(
                baseline, prepared, background_indices, heads, args.device
            )
            baseline_scores = model_scores(baseline, X_real_all, heads, args.device)

            for heldout in args.folds:
                experiment_id = (
                    f"{run_id}|holdout={heldout}|steps={args.steps}|batch={args.batch}|"
                    f"lr={args.lr:g}|synthetic_share={args.synthetic_share:g}"
                )
                done = con.execute(
                    "SELECT status FROM experiments WHERE experiment_id=?", (experiment_id,)
                ).fetchone()
                if done and done[0] == "complete":
                    log(f"skip complete {experiment_id}")
                    continue
                train_mask = eligible & (scenes != heldout)
                test_mask = eligible & (scenes == heldout)
                started = datetime.now(timezone.utc).isoformat()
                con.execute(
                    "INSERT OR REPLACE INTO experiments VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                    (
                        experiment_id,
                        run_id,
                        family,
                        seed,
                        heldout,
                        "running",
                        int(train_mask.sum()),
                        int(test_mask.sum()),
                        args.steps,
                        args.batch,
                        args.lr,
                        args.synthetic_share,
                        run_row[0] if run_row else None,
                        None,
                        str(source_weights),
                        None,
                        started,
                        None,
                        None,
                    ),
                )
                con.execute("DELETE FROM training_membership WHERE experiment_id=?", (experiment_id,))
                con.executemany(
                    "INSERT INTO training_membership VALUES (?,?,?)",
                    [
                        (experiment_id, int(i), "train_real_positive")
                        for i in np.flatnonzero(train_mask)
                    ]
                    + [
                        (experiment_id, int(i), "test_real_positive")
                        for i in np.flatnonzero(test_mask)
                    ],
                )
                con.commit()
                try:
                    record_scores_and_metrics(
                        con,
                        experiment_id,
                        "synthetic_only",
                        heldout,
                        baseline_scores,
                        baseline_thresholds,
                        audit,
                        valid,
                        eligible,
                        scenes,
                        terrains,
                        distances,
                        excess,
                        selected,
                    )
                    record_retention(
                        con,
                        experiment_id,
                        "synthetic_only",
                        baseline,
                        prepared,
                        baseline_thresholds,
                        (X_elbit, elbit_labels),
                        val_indices,
                        heads,
                        args.device,
                    )
                    X_train = X_real_all[torch.as_tensor(np.flatnonzero(train_mask), device=args.device)]
                    adapted, history, wall = fine_tune(
                        cfg,
                        prepared,
                        X_train,
                        heads,
                        source_weights,
                        args.device,
                        args.steps,
                        args.batch,
                        args.lr,
                        args.synthetic_share,
                        seed=100000 * seed + SCENES.index(heldout),
                    )
                    adapted_thresholds = calibrate_thresholds(
                        adapted, prepared, background_indices, heads, args.device
                    )
                    adapted_scores = model_scores(adapted, X_real_all, heads, args.device)
                    record_scores_and_metrics(
                        con,
                        experiment_id,
                        "adapted_positive_replay",
                        heldout,
                        adapted_scores,
                        adapted_thresholds,
                        audit,
                        valid,
                        eligible,
                        scenes,
                        terrains,
                        distances,
                        excess,
                        selected,
                    )
                    record_retention(
                        con,
                        experiment_id,
                        "adapted_positive_replay",
                        adapted,
                        prepared,
                        adapted_thresholds,
                        (X_elbit, elbit_labels),
                        val_indices,
                        heads,
                        args.device,
                    )
                    con.execute("DELETE FROM loss_history WHERE experiment_id=?", (experiment_id,))
                    con.executemany(
                        "INSERT INTO loss_history VALUES (?,?,?,?)",
                        [(experiment_id, step, loss, elapsed) for step, loss, elapsed in history],
                    )
                    adapted_path = None
                    if not args.no_save_models:
                        adapted_path = args.model_dir / f"{run_id}_holdout_{heldout}.pt"
                        torch.save(adapted.state_dict(), adapted_path)
                    con.execute(
                        "UPDATE experiments SET status='complete',wall_seconds=?,adapted_weights=?,"
                        "finished_utc=?,error=NULL WHERE experiment_id=?",
                        (
                            wall,
                            str(adapted_path) if adapted_path else None,
                            datetime.now(timezone.utc).isoformat(),
                            experiment_id,
                        ),
                    )
                    con.commit()
                    before = con.execute(
                        "SELECT vehicle_recall,human_head_rate FROM metrics WHERE experiment_id=? "
                        "AND condition='synthetic_only' AND subset='signal_ge6' AND terrain='ALL'",
                        (experiment_id,),
                    ).fetchone()
                    after = con.execute(
                        "SELECT vehicle_recall,human_head_rate FROM metrics WHERE experiment_id=? "
                        "AND condition='adapted_positive_replay' AND subset='signal_ge6' AND terrain='ALL'",
                        (experiment_id,),
                    ).fetchone()
                    log(
                        f"{run_id} holdout {heldout}: vehicle {before[0]:.3f}->{after[0]:.3f}; "
                        f"human FP {before[1]:.3f}->{after[1]:.3f}; {wall:.1f}s"
                    )
                    del adapted, X_train
                    torch.cuda.empty_cache()
                except Exception as exc:
                    con.execute(
                        "UPDATE experiments SET status='failed',finished_utc=?,error=? WHERE experiment_id=?",
                        (datetime.now(timezone.utc).isoformat(), f"{type(exc).__name__}: {exc}", experiment_id),
                    )
                    con.commit()
                    raise
            del baseline, prepared, X_real_all, X_elbit
            torch.cuda.empty_cache()

    export_summaries(con, args.output.parent / f"{args.output.stem}_exports")
    integrity = con.execute("PRAGMA integrity_check").fetchone()[0]
    if integrity != "ok":
        raise RuntimeError(f"result database integrity check failed: {integrity}")
    complete = con.execute("SELECT COUNT(*) FROM experiments WHERE status='complete'").fetchone()[0]
    log(f"complete experiments={complete}; database={args.output}")
    campaign.close()
    con.close()


if __name__ == "__main__":
    main()
