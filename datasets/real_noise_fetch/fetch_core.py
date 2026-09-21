"""Shared core for the Stage-R2 real-noise fetch (see datasets/FETCH_PLAN.md).

Data lands under $GEO_NOISE_RAW/<target>/ as one float32 npz per station-day
(ground VELOCITY, m/s, native rate) + a global manifest.csv row per unit.
Restartable: existing npz files are skipped.
"""
import os, io, time, json, traceback
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.environ.get("GEO_NOISE_RAW", os.path.abspath(os.path.join(HERE, "..", "..", "..", "geophone_real_noise")))
MANIFEST = os.path.join(ROOT, "manifest.csv")

MANIFEST_COLS = ["target", "net", "sta", "cha", "fs", "date", "label",
                 "wind_max_ms", "wind_mean_ms", "gust_max_ms", "rain_sum_mm",
                 "rain_max_mmh", "gap_frac", "rms_vel", "npz", "fetched_utc"]


# ---------------------------------------------------------------- weather
def get_weather_daily(lat, lon, start_date, end_date):
    """Hourly Open-Meteo ERA5 weather -> daily summary DataFrame (index=date)."""
    import openmeteo_requests, requests_cache
    from retry_requests import retry
    sess = retry(requests_cache.CachedSession(os.path.join(ROOT, ".wx_cache"),
                                              expire_after=-1),
                 retries=5, backoff_factor=0.5)
    om = openmeteo_requests.Client(session=sess)
    params = {"latitude": lat, "longitude": lon,
              "start_date": start_date, "end_date": end_date,
              "hourly": ["wind_speed_10m", "wind_gusts_10m", "precipitation"],
              "wind_speed_unit": "ms", "timezone": "UTC"}
    r = om.weather_api("https://archive-api.open-meteo.com/v1/archive", params=params)[0]
    h = r.Hourly()
    idx = pd.date_range(start=pd.to_datetime(h.Time(), unit="s", utc=True),
                        end=pd.to_datetime(h.TimeEnd(), unit="s", utc=True),
                        freq=pd.Timedelta(seconds=h.Interval()), inclusive="left")
    wx = pd.DataFrame({"wind": h.Variables(0).ValuesAsNumpy(),
                       "gust": h.Variables(1).ValuesAsNumpy(),
                       "precip": h.Variables(2).ValuesAsNumpy()}, index=idx)
    d = wx.resample("1D").agg(wind_max=("wind", "max"), wind_mean=("wind", "mean"),
                              gust_max=("gust", "max"), rain_sum=("precip", "sum"),
                              rain_max=("precip", "max"))
    d.index = d.index.date
    return d, wx


def label_day(row):
    """Condition label from a daily weather summary row."""
    if row.rain_sum >= 5.0:
        return "rain_heavy"
    if row.rain_sum >= 0.5:
        return "rain_light"
    if row.wind_max >= 12:
        return "wind_high"
    if row.wind_max >= 7:
        return "wind_mid"
    if row.wind_max >= 4:
        return "wind_low"
    return "calm"


# ---------------------------------------------------------------- seismic
class RawFDSNClient:
    """Minimal FDSN client over plain requests — bypasses obspy's service
    discovery, which is unreliable against the PH5 endpoints even when the
    services themselves respond. Same get_stations/get_waveforms surface as
    the obspy Client (only the kwargs we use)."""

    def __init__(self, base_url):
        import requests
        self.base = base_url.rstrip("/")
        self.sess = requests.Session()

    def get_stations(self, network, channel, level, starttime, endtime, **_):
        from obspy import read_inventory
        r = self.sess.get(f"{self.base}/station/1/query",
                          params=dict(net=network, cha=channel, level=level,
                                      starttime=starttime, endtime=endtime,
                                      format="xml"), timeout=300)
        r.raise_for_status()
        return read_inventory(io.BytesIO(r.content))

    def get_waveforms(self, net, sta, loc, cha, t1, t2):
        from obspy import read, Stream
        r = self.sess.get(f"{self.base}/dataselect/1/query",
                          params=dict(net=net, sta=sta, loc=loc, cha=cha,
                                      starttime=str(t1).replace("Z", ""),
                                      endtime=str(t2).replace("Z", "")),
                          timeout=1200)
        if r.status_code == 204:
            return Stream()
        r.raise_for_status()
        return read(io.BytesIO(r.content))


def make_client(name):
    # obspy uses urllib, whose default trust store misses some archive cert chains
    # (e.g. GEOFON) — point SSL at certifi's bundle before any client is built.
    import certifi
    os.environ.setdefault("SSL_CERT_FILE", certifi.where())
    from obspy.clients.fdsn import Client
    if name == "IRISPH5":             # discovery-free path (see RawFDSNClient)
        return RawFDSNClient("http://service.iris.edu/ph5ws")
    urls = {"GEOFON": "https://geofon.gfz.de",
            "EARTHSCOPE": "http://service.iris.edu"}
    last = None
    for attempt in range(5):
        try:
            if name in urls:
                return Client(base_url=urls[name], timeout=120)
            return Client(name)
        except Exception as e:
            last = e
            time.sleep(15 * (attempt + 1))
    raise last


def pick_stations(inv, n):
    """Evenly-spaced spatial subset of station codes from an Inventory."""
    stas = sorted({(s.code, s.latitude, s.longitude)
                   for net in inv for s in net.stations})
    if len(stas) <= n:
        return [s[0] for s in stas]
    step = len(stas) / n
    return [stas[int(i * step)][0] for i in range(n)]


def fetch_day_velocity(client, inv, net, sta, cha, day_str):
    """Fetch one station-day -> (velocity float32 array, fs, gap_frac). None if no data."""
    from obspy import UTCDateTime
    t1 = UTCDateTime(day_str)
    t2 = t1 + 86400
    st = client.get_waveforms(net, sta, "*", cha, t1, t2)
    if len(st) == 0:
        return None
    expected = 86400 * st[0].stats.sampling_rate
    have = sum(tr.stats.npts for tr in st)
    gap_frac = max(0.0, 1.0 - have / expected)
    st.merge(method=1, fill_value=0)
    tr = st[0]
    fs = tr.stats.sampling_rate
    ny = fs / 2.0
    tr.detrend("demean"); tr.detrend("linear")
    tr.taper(max_percentage=0.005, type="hann")
    tr.remove_response(inventory=inv, output="VEL",
                       pre_filt=(0.5, 1.0, 0.85 * ny, 0.95 * ny), water_level=60)
    return np.asarray(tr.data, np.float32), fs, gap_frac


def append_manifest(row):
    new = not os.path.exists(MANIFEST)
    pd.DataFrame([row], columns=MANIFEST_COLS).to_csv(
        MANIFEST, mode="a", header=new, index=False)


def run_units(target, client, inv, units, wx_daily, max_units=None, log=print):
    """units: list of (net, sta, cha, day_str, label). Skips existing; appends manifest."""
    outdir = os.path.join(ROOT, target)
    os.makedirs(outdir, exist_ok=True)
    done = fail = 0
    for i, (net, sta, cha, day, label) in enumerate(units):
        if max_units and done >= max_units:
            break
        npz = os.path.join(outdir, f"{net}.{sta}.{cha}.{day}.npz")
        if os.path.exists(npz):
            continue
        try:
            res = fetch_day_velocity(client, inv, net, sta, cha, day)
            if res is None:
                log(f"  [{i}] {sta} {day}: no data"); continue
            v, fs, gap = res
            if gap > 0.5:
                log(f"  [{i}] {sta} {day}: gap_frac {gap:.2f} > 0.5, skip"); continue
            np.savez_compressed(npz, v=v, fs=fs)
            w = wx_daily.loc[pd.to_datetime(day).date()]
            append_manifest(dict(
                target=target, net=net, sta=sta, cha=cha, fs=fs, date=day, label=label,
                wind_max_ms=round(float(w.wind_max), 2),
                wind_mean_ms=round(float(w.wind_mean), 2),
                gust_max_ms=round(float(w.gust_max), 2),
                rain_sum_mm=round(float(w.rain_sum), 2),
                rain_max_mmh=round(float(w.rain_max), 2),
                gap_frac=round(gap, 4), rms_vel=float(np.std(v)),
                npz=os.path.relpath(npz, ROOT),
                fetched_utc=pd.Timestamp.utcnow().isoformat()))
            done += 1
            log(f"  [{i}] {sta} {day} [{label}]: OK  ({v.size/fs/3600:.1f} h, "
                f"gap {gap:.3f}, {os.path.getsize(npz)/1e6:.0f} MB)")
        except Exception as e:
            fail += 1
            log(f"  [{i}] {sta} {day}: FAIL {type(e).__name__}: {e}")
            if fail > 50:
                log("  too many failures, aborting target"); break
            time.sleep(2)
    return done, fail
