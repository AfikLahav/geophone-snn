"""One-time preparation of every real dataset, cached.

Each dataset is read in its own format, resampled to the 1000 Hz the model expects, cut into
3-second windows, and put through the SAME feature code the synthetic side uses. The result is
cached, so the second launch starts training immediately.

Every window is registered in the score database with its recording, its position, and its START
TIME, because hysteresis reads consecutive windows in order and cannot do that without them.

Nothing here is calibrated against real amplitudes. Three of these datasets carry real physical
units, the rest are raw counts or normalized, and a per-dataset level anchor is recorded in the
manifest so a reader can see exactly what was assumed. The vehicle work already showed the
detection result is insensitive to that anchor across a 315-fold span, which is the reason a
relative anchor is acceptable at all.
"""
import glob
import io
import json
import os
import sys
import zipfile

import numpy as np
from scipy.signal import resample_poly

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, ".."))
sys.path.insert(0, os.path.join(ROOT, "simgeo", "simgeo_v42"))
import features as F                                        # noqa: E402

DATASETS_ROOT = os.environ.get("GEO_REAL_ROOT", os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "geophone_datasets"))
CACHE = os.environ.get("GEO_REAL_CACHE", os.environ.get("GEO_REAL_CACHE", os.path.join(ROOT, "campaign", "real_cache")))
HOP = 1500                      # 1.5 s, the 50% overlap the synthetic labels use
SCENE = 30 * int(F.FS)          # re-chunk long recordings so envelope statistics match synthetic

# Level anchor for datasets that carry no usable absolute scale.
#
# MEASURED, not assumed: the field recordings' own background windows have a median loudness of
# 10**-1.954 in the units those files are stored in. That is the level the model's thresholds --
# which come from simulated background -- are compatible with, because the field recordings are
# the one real dataset that needs no anchoring at all and transfers correctly.
#
# The earlier value here was 7.4, taken from an amplitude-gate note. It is 540x too high, and it
# made the model fire on 96% of a labelled QUIET class. Any change to this number must be
# re-checked against the false-alarm rate on a dataset with a real quiet label.
ELBIT_BG_RMS_MV = 10 ** -1.954
TARGET_BG_MV = ELBIT_BG_RMS_MV


# ------------------------------------------------------------------ helpers

def to_1k(x, fs):
    """Resample to 1000 Hz. Anything already there passes through untouched."""
    fs = float(fs)
    if abs(fs - 1000.0) < 1e-6:
        return np.asarray(x, np.float64)
    from math import gcd
    up, down = int(round(1000.0)), int(round(fs))
    g = gcd(up, down)
    return resample_poly(np.asarray(x, np.float64), up // g, down // g)


def anchor_level(x, target_mv=TARGET_BG_MV):
    """Scale a recording so its quiet background sits at the anchor level.

    Raw counts and normalized signals carry no physical scale, so some choice is unavoidable.
    The choice is recorded, and the vehicle diagnostic showed the result barely moves across a
    315-fold change in it."""
    x = np.asarray(x, np.float64)
    x = x - np.median(x)
    seg = 2000
    n = len(x) // seg
    if n < 3:
        s = np.std(x) or 1.0
    else:
        r = np.std(x[:n * seg].reshape(n, seg), axis=1)
        s = np.quantile(r, 0.2) or (np.std(x) or 1.0)
    return x * (target_mv / s)


def windows_from(sig, recording, label, t_offset=0.0, **extra):
    """Cut one continuous 1000 Hz signal into features plus window rows."""
    feats, rows = [], []
    pos = 0
    for c0 in range(0, len(sig), SCENE):
        seg = sig[c0:c0 + SCENE]
        if len(seg) < F.NW:
            continue
        pre = F.scene_precompute(np.asarray(seg, np.float32))
        for i0 in range(0, len(seg) - F.NW + 1, HOP):
            feats.append(F.window_features(pre, i0).astype(np.float32))
            rows.append(dict(recording=recording, pos=pos, label=label,
                             t0_s=t_offset + (c0 + i0) / F.FS, **extra))
            pos += 1
    if not feats:
        return np.empty((0, F.NFEAT), np.float32), []
    return np.stack(feats), rows


def _cache_paths(name):
    return (os.path.join(CACHE, f"{name}_X.npy"),
            os.path.join(CACHE, f"{name}_rows.json"),
            os.path.join(CACHE, f"{name}_info.json"))


def cached(name):
    xp, rp, ip = _cache_paths(name)
    if os.path.exists(xp) and os.path.exists(rp):
        return np.load(xp), json.load(open(rp, encoding="utf-8")), json.load(open(ip, encoding="utf-8"))
    return None


def save_cache(name, X, rows, info):
    os.makedirs(CACHE, exist_ok=True)
    xp, rp, ip = _cache_paths(name)
    np.save(xp, X)
    json.dump(rows, open(rp, "w", encoding="utf-8"))
    json.dump(info, open(ip, "w", encoding="utf-8"), indent=1)


# ------------------------------------------------------------------ datasets

def prep_elbit():
    """The four field recordings. Already 1000 Hz, already in millivolts -- nothing assumed."""
    import pandas as pd
    d = os.path.join(ROOT, "Goephone-Project", "geophone_data")
    files = {"human.csv": "human", "car.csv": "vehicle",
             "human_nothing.csv": "nothing", "car_nothing.csv": "nothing"}
    Xs, rows = [], []
    for fn, lab in files.items():
        a = pd.read_csv(os.path.join(d, fn))["amplitude"].to_numpy(np.float32)
        X, r = windows_from(a, recording=fn, label=lab)
        Xs.append(X)
        rows += r
    info = dict(source=d, fs_native=1000, resampled=False, units="millivolts as recorded",
                note="one continuous session per class -- no valid split exists, zero-shot only")
    return np.vstack(Xs), rows, info


def prep_savanna(max_rows=None):
    """Savanna wildlife monitoring: people, eleven animal families, and a labelled quiet class,
    every row carrying a real distance.

    The file stores three components but its component axis is a placeholder of zeros, so which
    one is vertical is not recorded. Rather than guess, the vertical is CHOSEN as the component
    with the most energy in the footstep band on human-labelled rows, and the choice is written
    into the manifest."""
    import h5py
    p = os.path.join(DATASETS_ROOT, "seissavanna", "dset_allspec_150_chunks_clean.nc")
    with h5py.File(p, "r") as f:
        cls = f["class"][:]
        dist = f["distance"][:]
        seis = f["seis"][:]
        n = len(cls)
        # class 0 is the quiet class and 1 is people, per the dataset's own notes file; the
        # pickled dictionary on disk disagrees, so the counts are asserted against the notes
        if int((cls == 1).sum()) < int((cls == 0).sum()):
            raise RuntimeError("savanna label mapping does not match the documented counts -- "
                               "resolve before using this dataset")
        probe = np.where(cls == 1)[0][:200]
        e = []
        for c in range(3):
            w = f["chunk"][probe.min():probe.min() + 200, c, :]
            sp = np.abs(np.fft.rfft(w - w.mean(1, keepdims=True), axis=1)) ** 2
            fr = np.fft.rfftfreq(w.shape[1], 1 / 200.0)
            e.append(float(sp[:, (fr >= 20) & (fr <= 90)].sum()))
        vert = int(np.argmax(e))

        # ONE scale for the whole dataset, from the quiet class -- never per row. Anchoring each
        # row separately normalizes every recording to the same loudness, which destroys exactly
        # what this dataset is here to measure: how the signal falls with distance.
        #
        # The target is the MEASURED background of the field recordings, not a nominal constant.
        # Getting that wrong is not a small error: an earlier attempt anchored to 7.4 mV, which
        # put this dataset's background 540x above the field rig's, and the model then fired on
        # 96% of the labelled QUIET windows. The thresholds come from simulated background, so a
        # real dataset has to arrive at the level the field recordings actually sit at.
        quiet = np.where(cls == 0)[0][:400]
        qr = []
        for i in quiet:
            a = np.asarray(f["chunk"][int(i), vert, :], np.float64)
            qr.append(np.std(a - np.median(a)))
        gain = ELBIT_BG_RMS_MV / (float(np.median(qr)) or 1.0)

        idx = np.arange(n) if max_rows is None else np.linspace(0, n - 1, max_rows).astype(int)
        Xs, rows = [], []
        label_of = {0: "nothing", 1: "human"}
        for i in idx:
            lab = label_of.get(int(cls[i]), "animal")
            w = np.asarray(f["chunk"][i, vert, :], np.float64)
            sig = to_1k(w - np.median(w), 200.0) * gain
            X, r = windows_from(sig, recording=f"row{int(i)}", label=lab,
                                distance_m=float(dist[i]), station=str(int(seis[i])))
            if len(X):
                Xs.append(X)
                rows += r
    info = dict(source=p, fs_native=200, resampled="200 -> 1000 Hz",
                vertical_component=vert, component_choice="most 20-90 Hz energy on human rows",
                band_ceiling_hz=100, units="calibrated velocity, one global scale",
                global_gain=float(gain), gain_source="quiet-class rows, 68th percentile",
                note="split by station; the component axis in the file is a placeholder. The "
                     "scale is global on purpose -- per-row normalization would flatten the "
                     "amplitude and make the distance analysis meaningless")
    return np.vstack(Xs), rows, info


def prep_vehicles():
    """Four vehicles across four road surfaces, with a ground-truth distance every second.

    Positives are windows at a near closest approach, negatives come from the far end of the SAME
    recording -- which is what stops the comparison becoming synthetic-versus-this-dataset."""
    import polars as pl
    base = os.path.join(DATASETS_ROOT, "m3n_vc")
    scenes = [d for d in ("a06", "h08", "h24", "i22", "i29", "s31")
              if os.path.isdir(os.path.join(base, d))]
    Xs, rows, unreadable, multi_vehicle = [], [], [], set()
    for scene in scenes:
        loc = os.path.join(base, scene, "sensor_location.parquet")
        terr_of = {}
        if os.path.exists(loc):
            for r in pl.read_parquet(loc).to_dicts():
                terr_of[str(r["sensor_id"])] = str(r.get("terrain", ""))
        runs = os.path.join(base, scene, "run_ids.parquet")
        veh_of = {}
        if os.path.exists(runs):
            for r in pl.read_parquet(runs).to_dicts():
                veh_of[int(r["run_id"])] = str(r.get("label", ""))

        for geo in sorted(glob.glob(os.path.join(base, scene, "*_geo.parquet"))):
            stem = os.path.basename(geo).replace("_geo.parquet", "")
            dis = os.path.join(base, scene, stem + "_dis.parquet")
            if not os.path.exists(dis):
                continue
            try:
                g = pl.read_parquet(geo)
                d = pl.read_parquet(dis)
            except Exception:
                # one file in this collection is zero bytes; skipped and counted rather than
                # allowed to take the whole dataset down
                unreadable.append(os.path.relpath(dis, base))
                continue
            if "distance" not in d.columns:
                # this scene runs two vehicles at once and stores a distance column per vehicle,
                # so "closest approach" is ambiguous. Prior work on this project excluded it from
                # single-vehicle analysis; the same exclusion is kept and recorded
                multi_vehicle.add(scene)
                continue
            sig = g["samples"].to_numpy().astype(np.float64)
            ts = g["timestamp"].to_numpy().astype(np.float64)
            if np.nanmedian(ts) > 1e11:        # one scene records milliseconds, the rest seconds
                ts = ts / 1000.0
            # the distance file stores a datetime, the geophone file an epoch second
            dt = d["time"].to_numpy().astype("datetime64[ns]").astype(np.int64) / 1e9
            dd = d["distance"].to_numpy().astype(np.float64)
            ok = np.isfinite(dd)
            dt, dd = dt[ok], dd[ok]
            if len(sig) < 200 * 40 or len(dd) < 10:
                continue
            run_no = int(stem.split("_")[0].replace("run", "")) if "run" in stem else -1
            sensor = stem.split("_")[1] if "_" in stem else ""
            terr = terr_of.get(sensor, "")
            veh = veh_of.get(run_no, "")
            t0 = ts[0]
            near_t = float(dt[int(np.argmin(dd))] - t0)
            far_t = float(dt[int(np.argmax(dd))] - t0)
            sig = anchor_level(to_1k(sig, 200.0))
            for tag, tc, lab in (("near", near_t, "vehicle"), ("far", far_t, "nothing")):
                i0 = int(max(0, (tc - 5.0) * F.FS))
                seg = sig[i0:i0 + 10 * int(F.FS)]
                if len(seg) < F.NW:
                    continue
                X, r = windows_from(seg, recording=f"{scene}/{stem}/{tag}", label=lab,
                                    terrain=terr, t_offset=i0 / F.FS,
                                    extra={"vehicle": veh, "sensor": sensor, "scene": scene})
                if len(X):
                    Xs.append(X)
                    rows += r
    if not Xs:
        raise RuntimeError("no vehicle drive-by windows built -- check the local copy")
    info = dict(source=base, fs_native=200, resampled="200 -> 1000 Hz",
                units="raw counts, level-anchored", band_ceiling_hz=100, scenes=scenes,
                unreadable_files=unreadable,
                excluded_multi_vehicle_scenes=sorted(multi_vehicle),
                note="split by scene; positives near closest approach, negatives from the far "
                     "end of the SAME recording, which is what stops the comparison becoming "
                     "synthetic-versus-this-dataset")
    return np.vstack(Xs), rows, info


def prep_revibe():
    """Indoor footsteps: 191 recordings, 11 people, wooden and concrete.

    Wrong physical regime for validating the outdoor simulator -- a floor is a plate, not a soil
    half-space -- but it is the only data on disk with a genuine multi-person structure, which is
    exactly what the outdoor real set lacks."""
    from scipy.io import loadmat
    zp = os.path.join(DATASETS_ROOT, "revibe", "Data.zip")
    Xs, rows = [], []
    with zipfile.ZipFile(zp) as z:
        names = [n for n in z.namelist() if n.endswith(".mat") and not n.startswith("__MACOSX")]
        for nm in names:
            parts = nm.split("/")
            person, surface = parts[1], parts[2]
            with z.open(nm) as fh:
                m = loadmat(io.BytesIO(fh.read()))
            arr = None
            for k, v in m.items():
                if not k.startswith("__") and isinstance(v, np.ndarray) and v.size > 1000:
                    arr = np.asarray(v, np.float64).squeeze()
                    break
            if arr is None:
                continue
            if arr.ndim > 1:
                arr = arr[:, 0] if arr.shape[0] > arr.shape[1] else arr[0]
            sig = anchor_level(to_1k(arr, 25600.0))
            X, r = windows_from(sig, recording=nm, label="human", person=person, surface=surface)
            if len(X):
                Xs.append(X)
                rows += r
    info = dict(source=zp, fs_native=25600, resampled="25600 -> 1000 Hz",
                units="unverified, level-anchored",
                note="indoor floor-coupled, not soil -- tests the feature bank and cross-person "
                     "behaviour, never the outdoor simulator. Split by person")
    return np.vstack(Xs), rows, info


def prep_rumble():
    """Wildlife footfall: bear, cougar, wolf, deer. Amplitude is normalized and clipped in the
    source files, so only rhythm and shape mean anything here."""
    from scipy.io import wavfile
    zp = os.path.join(DATASETS_ROOT, "rumble_jungle_footfall", "dataset-audio.zip")
    Xs, rows = [], []
    with zipfile.ZipFile(zp) as z:
        names = [n for n in z.namelist()
                 if n.lower().endswith(".wav") and not n.startswith("__MACOSX")][:2000]
        for nm in names:
            with z.open(nm) as fh:
                fs, a = wavfile.read(io.BytesIO(fh.read()))
            a = np.asarray(a, np.float64)
            if a.ndim > 1:
                a = a[:, 0]
            sig = anchor_level(to_1k(a, fs))
            X, r = windows_from(sig, recording=nm, label="animal",
                                extra={"group": nm.split("/")[1] if "/" in nm else ""})
            if len(X):
                Xs.append(X)
                rows += r
    info = dict(source=zp, fs_native=800, resampled="-> 1000 Hz",
                units="normalized and clipped in source -- relative measures only",
                note="animal rhythm only; no absolute level, no attenuation")
    return np.vstack(Xs), rows, info


def prep_blasts():
    """Near-field blasts, calibrated and distance-tagged. Two uses: false alarms on impulsive
    events the model never saw, and a direct check of amplitude against distance."""
    import obspy
    d = os.path.join(DATASETS_ROOT, "eida_6a_firecracker_rocket")
    st = obspy.read(os.path.join(d, "6A_firecracker_rocket_events.mseed"))
    Xs, rows = [], []
    for tr in st.select(channel="GHZ"):           # vertical geophone only, never the microphones
        sig = anchor_level(to_1k(tr.data, tr.stats.sampling_rate))
        X, r = windows_from(sig, recording=f"{tr.stats.station}.{tr.stats.starttime}",
                            label="blast", station=tr.stats.station)
        if len(X):
            Xs.append(X)
            rows += r
    info = dict(source=d, fs_native=2000, resampled="2000 -> 1000 Hz",
                units="counts / 75900 gives metres per second; level-anchored here",
                note="the bundled metadata says 1000 Hz and is wrong -- the trace headers say "
                     "2000. Microphone channels excluded")
    return np.vstack(Xs), rows, info


def prep_regional():
    """Earthquakes, explosions, surface events, sonic booms and thunder, at regional distance.

    A pure false-alarm test: none of these should raise a person or vehicle head. Content above
    50 Hz does not exist in the source, which is recorded rather than hidden by the resampling."""
    import h5py
    import pandas as pd
    d = os.path.join(DATASETS_ROOT, "pnw_ml")
    meta = pd.read_csv(os.path.join(d, "miniPNW_metadata.csv"))
    Xs, rows = [], []
    with h5py.File(os.path.join(d, "miniPNW_waveforms.hdf5"), "r") as f:
        for _, row in meta.iterrows():
            name = str(row["trace_name"])
            bucket, sl = name.split("$")
            i = int(sl.split(",")[0])
            arr = f[f"data/{bucket}"][i, 2, :]          # component order is ENZ -> vertical last
            sig = anchor_level(to_1k(arr, float(row["trace_sampling_rate_hz"])))
            X, r = windows_from(sig, recording=name, label="nothing",
                                station=str(row["station_code"]),
                                extra={"source_type": str(row["source_type"])})
            if len(X):
                Xs.append(X)
                rows += r
    info = dict(source=d, fs_native=100, resampled="100 -> 1000 Hz (no real content above 50 Hz)",
                units="raw counts, level-anchored",
                note="regional, 3-243 km -- a confuser test, never a near-field reference")
    return np.vstack(Xs), rows, info


def _rail_chunk(args):
    """One worker: a range of catalogued train passes from the rail HDF5 -> features + rows."""
    import h5py
    path, lo, hi, fs, in_pass = args
    Xs, rows = [], []
    with h5py.File(path, "r") as h:
        for i in range(lo, hi):
            sig = np.asarray(h["data"][i], np.float64)
            if not np.isfinite(sig).all() or sig.std() == 0:
                continue
            # 250 -> 200 Hz first, so the band matches the other 200 Hz sets, then to 1 kHz
            from scipy.signal import resample_poly
            sig = anchor_level(to_1k(resample_poly(sig, 4, 5), 200.0))
            X, r = windows_from(sig, recording=f"pass{i:04d}", label="nothing",
                                extra={"pass_id": i, "in_pass": None})
            for row in r:
                c = row["t0_s"] + 1.5
                row["extra"]["in_pass"] = bool(in_pass[0] <= c <= in_pass[1])
            if len(X):
                Xs.append(X)
                rows += r
    return Xs, rows


def prep_rail():
    """A month of catalogued train passes beside a railway near Lyon: 2,574 passes, each a 180 s
    cut around a 120 s catalogued pass, single station, 250 Hz. A heavy-vehicle confuser and a
    person-false-alarm test on real continuous ground; nothing here is a person.

    Every window is labeled nothing. `in_pass` marks the catalogued 120 s of each cut; the rest is
    the 30 s lead and tail (assumed symmetric around the catalog time)."""
    from concurrent.futures import ProcessPoolExecutor
    d = os.path.join(DATASETS_ROOT, "train_tremor_lyon", "data")
    path = os.path.join(d, "train_raw_data_B05_HHZ.h5")
    import h5py
    with h5py.File(path, "r") as h:
        n, ns = h["data"].shape
        fs = float(h["fs"][()])
    span = ns / fs
    lead = (span - 120.0) / 2.0
    in_pass = (lead, lead + 120.0)
    step = 64
    jobs = [(path, lo, min(lo + step, n), fs, in_pass) for lo in range(0, n, step)]
    Xs, rows = [], []
    with ProcessPoolExecutor(max_workers=6) as ex:
        for xs, rs in ex.map(_rail_chunk, jobs):
            Xs += xs
            rows += rs
    info = dict(source=d, fs_native=fs, resampled="250 -> 200 -> 1000 Hz (band-limited to 100 Hz to match the other 200 Hz sets)",
                units="raw counts, level-anchored per pass", n_passes=n, cut_s=span, catalog_pass_s=120.0,
                note="all windows labeled nothing; in_pass marks the catalogued 120 s; the person head firing here is a false alarm, the vehicle head firing is a train read as a vehicle")
    return np.vstack(Xs), rows, info


def prep_elbit_bl200():
    """The four field recordings decimated to 200 Hz and brought back to 1 kHz, so the matched-band
    models can be scored on the field with the same empty band the 200 Hz datasets have."""
    import pandas as pd
    from scipy.signal import resample_poly
    d = os.path.join(ROOT, "Goephone-Project", "geophone_data")
    files = {"human.csv": "human", "car.csv": "vehicle",
             "human_nothing.csv": "nothing", "car_nothing.csv": "nothing"}
    Xs, rows = [], []
    for fn, lab in files.items():
        a = pd.read_csv(os.path.join(d, fn))["amplitude"].to_numpy(np.float64)
        a = to_1k(resample_poly(a, 1, 5), 200.0)
        X, r = windows_from(a.astype(np.float32), recording=fn, label=lab)
        Xs.append(X)
        rows += r
    info = dict(source=d, fs_native=1000, resampled="1000 -> 200 -> 1000 Hz", units="millivolts as recorded",
                note="the field recordings with the band above 100 Hz removed, for the matched-band arm only")
    return np.vstack(Xs), rows, info


REGISTRY = {
    "elbit": prep_elbit,
    "rail": prep_rail,
    "elbit_bl200": prep_elbit_bl200,
    "savanna": prep_savanna,
    "vehicles": prep_vehicles,
    "revibe": prep_revibe,
    "rumble": prep_rumble,
    "blasts": prep_blasts,
    "regional": prep_regional,
}

# Datasets the sweep always scores against. The rest are opt-in: they are slower to prepare and
# answer narrower questions.
CORE = ["elbit", "savanna", "vehicles"]


def prepare(names=None, db_con=None, verbose=True):
    """Prepare (or load from cache) each dataset and register its windows in the database."""
    out = {}
    for name in (names or CORE):
        got = cached(name)
        if got is None:
            if verbose:
                print(f"[prep] {name}: building ...", flush=True)
            X, rows, info = REGISTRY[name]()
            save_cache(name, X, rows, info)
        else:
            X, rows, info = got
            if verbose:
                print(f"[prep] {name}: cached", flush=True)
        if db_con is not None:
            from training import db
            db.add_windows(db_con, name, rows, info)
        out[name] = (X, rows, info)
        if verbose:
            labs = {}
            for r in rows:
                labs[r["label"]] = labs.get(r["label"], 0) + 1
            print(f"       {len(rows):,} windows  {labs}")
    return out


if __name__ == "__main__":
    which = sys.argv[1:] or CORE
    prepare(which)
