"""The score database.

One SQLite file holding every model's confidence on every window it was tested on, so any
post-processing -- hysteresis, dynamic thresholding, event-level scoring, a new chart -- runs
from stored numbers and never re-runs a model.

Layout, and why:

  runs                 one row per trained model: its configuration and everything measured
                       about it (size, operations, firing rate, energy, wall clock, status).

  windows              one row per window per dataset, with the window's START TIME inside its
                       recording. Hysteresis reads consecutive windows in order, so without
                       that column the database cannot serve the thing it exists for.
                       Written once when a dataset is prepared, shared by every model.

  score_blocks         per (run, dataset): the RAW per-head presence score for every window of
                       that dataset, in window order, as one float32 blob. Blobs rather than
                       58 million rows -- a hysteresis sweep is then a single read and a numpy
                       reshape instead of a query, and the whole table is ~700 MB rather than
                       ~4 GB. Scores, never decisions: a stored decision fixes the threshold
                       forever, a stored score lets any threshold be applied later.

  background_quantiles per (run, head): 1000 quantiles of that head's score distribution on
                       SYNTHETIC background windows. Thresholds are calibrated on synthetic
                       background only, so re-deriving an operating point later needs this
                       distribution -- and 1000 quantiles is a few hundred thousand rows
                       instead of the tens of millions storing every synthetic score would be.

  metrics              per (run, dataset, operating point): accuracy / recall / precision / F1,
                       plus the threshold-free score, which is recorded but flagged diagnostic.

Window identifiers are assigned once per dataset and are contiguous within it, which is what
lets the score blob be stored positionally.
"""
import json
import os
import random
import sqlite3
import time
import numpy as np

HEADS = ("human", "animal", "vehicle")
N_QUANTILES = 1000

SCHEMA = """
CREATE TABLE IF NOT EXISTS runs (
    run_id        TEXT PRIMARY KEY,
    grp           TEXT,
    label         TEXT,
    config_json   TEXT,
    status        TEXT,          -- ok | failed | skipped
    error         TEXT,
    params        INTEGER,
    macs          INTEGER,       -- multiply-accumulates of the matched ordinary network
    sops          REAL,          -- spike-driven accumulate operations
    energy_uj     REAL,          -- microjoules per window, 45 nm constants
    energy_ratio  REAL,          -- ordinary-network energy / this model's energy
    fire_json     TEXT,          -- firing rate per layer
    dead_json     TEXT,          -- never-firing fraction per layer
    tau_json      TEXT,          -- learned leak per layer
    steps         INTEGER,
    wall_min      REAL,
    peak_mib      REAL,
    best_val      REAL,
    started       TEXT,
    finished      TEXT
);

CREATE TABLE IF NOT EXISTS windows (
    window_id   INTEGER PRIMARY KEY,
    dataset     TEXT NOT NULL,
    recording   TEXT NOT NULL,   -- source file / scene / station, whatever identifies a stream
    pos         INTEGER NOT NULL,-- index of this window inside that recording
    t0_s        REAL NOT NULL,   -- start time in seconds inside that recording
    label       TEXT,            -- human | vehicle | animal | nothing | unknown
    distance_m  REAL,
    station     TEXT,
    person      TEXT,
    surface     TEXT,
    terrain     TEXT,
    extra_json  TEXT
);
CREATE UNIQUE INDEX IF NOT EXISTS ix_win_unique  ON windows(dataset, recording, pos);
CREATE INDEX        IF NOT EXISTS ix_win_dataset ON windows(dataset, window_id);

CREATE TABLE IF NOT EXISTS score_blocks (
    run_id     TEXT NOT NULL,
    dataset    TEXT NOT NULL,
    first_wid  INTEGER NOT NULL, -- window_id of row 0 of the blob
    n_windows  INTEGER NOT NULL,
    n_heads    INTEGER NOT NULL,
    heads_json TEXT NOT NULL,
    data       BLOB NOT NULL,    -- float32, shape (n_windows, n_heads), window_id order
    PRIMARY KEY (run_id, dataset)
);

CREATE TABLE IF NOT EXISTS background_quantiles (
    run_id TEXT NOT NULL,
    head   TEXT NOT NULL,
    data   BLOB NOT NULL,        -- float32, N_QUANTILES values, quantile 0..1 ascending
    PRIMARY KEY (run_id, head)
);

CREATE TABLE IF NOT EXISTS metrics (
    run_id    TEXT NOT NULL,
    dataset   TEXT NOT NULL,
    op_point  TEXT NOT NULL,     -- e.g. far_0.1pct | far_1pct | argmax
    cls       TEXT NOT NULL,     -- overall | human | vehicle | animal | nothing
    n         INTEGER,
    accuracy  REAL,
    recall    REAL,
    precision_ REAL,
    f1        REAL,
    auc       REAL,              -- threshold-free, DIAGNOSTIC ONLY (see report)
    extra_json TEXT,
    PRIMARY KEY (run_id, dataset, op_point, cls)
);

CREATE TABLE IF NOT EXISTS dataset_manifest (
    dataset    TEXT PRIMARY KEY,
    n_windows  INTEGER,
    first_wid  INTEGER,
    last_wid   INTEGER,
    info_json  TEXT
);
"""


def connect(path):
    """Open (creating if needed) the campaign database."""
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    con = sqlite3.connect(path, timeout=60.0)
    con.execute("PRAGMA journal_mode=WAL")
    con.execute("PRAGMA synchronous=NORMAL")
    con.executescript(SCHEMA)
    con.commit()
    return con




LOCK_RETRY_S = 900.0


def _write(con, sql, params=(), many=False):
    """Execute one write, waiting out 'database is locked' instead of failing.

    Several processes write to this file -- the loop, the ordinary-model harness, the review --
    and SQLite's own busy wait can starve a writer under back-to-back transactions from another
    process. A lost training run costs minutes; waiting costs nothing."""
    t0 = time.time()
    while True:
        try:
            return con.executemany(sql, params) if many else con.execute(sql, params)
        except sqlite3.OperationalError as e:
            if "locked" not in str(e) or time.time() - t0 > LOCK_RETRY_S:
                raise
            time.sleep(random.uniform(0.2, 1.5))


def _commit(con):
    t0 = time.time()
    while True:
        try:
            return con.commit()
        except sqlite3.OperationalError as e:
            if "locked" not in str(e) or time.time() - t0 > LOCK_RETRY_S:
                raise
            time.sleep(random.uniform(0.2, 1.5))


# ---------------------------------------------------------------- writes

def upsert_run(con, run_id, **fields):
    """Insert or update one run row. Unknown keys are ignored so callers can pass a superset."""
    cols = [r[1] for r in _write(con, "PRAGMA table_info(runs)")]
    fields = {k: v for k, v in fields.items() if k in cols}
    fields["run_id"] = run_id
    keys = list(fields)
    _write(con,
        f"INSERT INTO runs ({','.join(keys)}) VALUES ({','.join('?' * len(keys))}) "
        f"ON CONFLICT(run_id) DO UPDATE SET {','.join(k + '=excluded.' + k for k in keys if k != 'run_id')}",
        [fields[k] for k in keys])
    _commit(con)


def add_windows(con, dataset, rows, info=None):
    """Register every window of a dataset. `rows` is a list of dicts with at least
    recording / pos / t0_s / label. Returns (first_window_id, count).

    Idempotent: if the dataset is already registered with the same count it is left alone, so
    re-preparing a cached dataset costs nothing."""
    cur = _write(con, "SELECT n_windows, first_wid FROM dataset_manifest WHERE dataset=?", (dataset,))
    got = cur.fetchone()
    if got and got[0] == len(rows):
        return got[1], got[0]
    if got:
        _write(con, "DELETE FROM windows WHERE dataset=?", (dataset,))
        _write(con, "DELETE FROM score_blocks WHERE dataset=?", (dataset,))

    nxt = _write(con, "SELECT COALESCE(MAX(window_id), -1) + 1 FROM windows").fetchone()[0]
    payload = []
    for i, r in enumerate(rows):
        payload.append((
            nxt + i, dataset, str(r["recording"]), int(r["pos"]), float(r["t0_s"]),
            r.get("label"), r.get("distance_m"), r.get("station"), r.get("person"),
            r.get("surface"), r.get("terrain"),
            json.dumps(r["extra"]) if r.get("extra") else None))
    _write(con,
        "INSERT INTO windows (window_id,dataset,recording,pos,t0_s,label,distance_m,station,"
        "person,surface,terrain,extra_json) VALUES (?,?,?,?,?,?,?,?,?,?,?,?)", payload, many=True)
    _write(con,
        "INSERT INTO dataset_manifest (dataset,n_windows,first_wid,last_wid,info_json) "
        "VALUES (?,?,?,?,?) ON CONFLICT(dataset) DO UPDATE SET n_windows=excluded.n_windows,"
        "first_wid=excluded.first_wid,last_wid=excluded.last_wid,info_json=excluded.info_json",
        (dataset, len(rows), nxt, nxt + len(rows) - 1, json.dumps(info or {})))
    _commit(con)
    return nxt, len(rows)


def put_scores(con, run_id, dataset, scores, heads=HEADS):
    """Store one model's raw per-head scores for one dataset.

    `scores` is (n_windows, n_heads) float, in the same order the windows were registered."""
    first, n = _write(con,
        "SELECT first_wid, n_windows FROM dataset_manifest WHERE dataset=?", (dataset,)).fetchone()
    a = np.ascontiguousarray(np.asarray(scores, dtype=np.float32))
    if a.shape[0] != n:
        raise ValueError(f"{dataset}: {a.shape[0]} scores for {n} registered windows")
    _write(con,
        "INSERT OR REPLACE INTO score_blocks (run_id,dataset,first_wid,n_windows,n_heads,"
        "heads_json,data) VALUES (?,?,?,?,?,?,?)",
        (run_id, dataset, first, n, a.shape[1], json.dumps(list(heads)), a.tobytes()))
    _commit(con)


def put_background_quantiles(con, run_id, head, values):
    """Store the synthetic-background score distribution for one head, as N_QUANTILES points."""
    v = np.asarray(values, dtype=np.float64)
    q = np.quantile(v, np.linspace(0.0, 1.0, N_QUANTILES)).astype(np.float32)
    _write(con, "INSERT OR REPLACE INTO background_quantiles (run_id,head,data) VALUES (?,?,?)",
                (run_id, head, np.ascontiguousarray(q).tobytes()))
    _commit(con)


def put_metric(con, run_id, dataset, op_point, cls, **m):
    _write(con,
        "INSERT OR REPLACE INTO metrics (run_id,dataset,op_point,cls,n,accuracy,recall,"
        "precision_,f1,auc,extra_json) VALUES (?,?,?,?,?,?,?,?,?,?,?)",
        (run_id, dataset, op_point, cls, m.get("n"), m.get("accuracy"), m.get("recall"),
         m.get("precision"), m.get("f1"), m.get("auc"),
         json.dumps(m["extra"]) if m.get("extra") else None))
    _commit(con)


# ---------------------------------------------------------------- reads

def completed_runs(con):
    """Run identifiers already finished, so the loop can resume without repeating work."""
    return {r[0] for r in con.execute("SELECT run_id FROM runs WHERE status IN ('ok','failed')")}


def dataset_windows(con, dataset):
    """Every window of a dataset, in window_id order, as a dict of numpy arrays.

    Order matches the score blob exactly, so post-processing joins by position, not by query."""
    cur = con.execute(
        "SELECT window_id,recording,pos,t0_s,label,distance_m,station,person,surface,terrain "
        "FROM windows WHERE dataset=? ORDER BY window_id", (dataset,))
    rows = cur.fetchall()
    if not rows:
        return None
    cols = list(zip(*rows))
    return {
        "window_id": np.array(cols[0], dtype=np.int64),
        "recording": np.array(cols[1], dtype=object),
        "pos": np.array(cols[2], dtype=np.int64),
        "t0_s": np.array(cols[3], dtype=np.float64),
        "label": np.array(cols[4], dtype=object),
        "distance_m": np.array([np.nan if v is None else v for v in cols[5]], dtype=np.float64),
        "station": np.array(cols[6], dtype=object),
        "person": np.array(cols[7], dtype=object),
        "surface": np.array(cols[8], dtype=object),
        "terrain": np.array(cols[9], dtype=object),
    }


def get_scores(con, run_id, dataset):
    """One model's scores for one dataset: (n_windows, n_heads) float32, plus the head names.

    Rows line up positionally with dataset_windows(), which is already sorted by recording and
    position -- so a hysteresis sweep is this read, a groupby on `recording`, and nothing else."""
    got = con.execute(
        "SELECT n_windows,n_heads,heads_json,data FROM score_blocks WHERE run_id=? AND dataset=?",
        (run_id, dataset)).fetchone()
    if not got:
        return None, None
    n, k, heads_json, blob = got
    a = np.frombuffer(blob, dtype=np.float32).reshape(n, k)
    return a, json.loads(heads_json)


def threshold_at_far(con, run_id, head, far):
    """The score threshold giving `far` false-alarm rate on synthetic background.

    Derived from the stored quantiles, so any operating point can be chosen after the fact
    without re-running the model or the synthetic validation pass."""
    got = con.execute("SELECT data FROM background_quantiles WHERE run_id=? AND head=?",
                      (run_id, head)).fetchone()
    if not got:
        return None
    q = np.frombuffer(got[0], dtype=np.float32)
    # background scores above the threshold are the false alarms -> take the (1-far) quantile
    idx = np.clip(int(round((1.0 - far) * (len(q) - 1))), 0, len(q) - 1)
    return float(q[idx])


def runs_table(con):
    """Every run row as a list of dicts -- what the report iterates over."""
    cur = con.execute("SELECT * FROM runs ORDER BY started")
    cols = [d[0] for d in cur.description]
    return [dict(zip(cols, r)) for r in cur.fetchall()]


def metrics_table(con):
    cur = con.execute("SELECT * FROM metrics")
    cols = [d[0] for d in cur.description]
    return [dict(zip(cols, r)) for r in cur.fetchall()]
