"""Stage-R2 driver. Usage:
    python fetch_run.py <yw|lasso|zg|is_il|all> [--smoke]
--smoke fetches 1 unit per target to validate the path end-to-end.
"""
import os, sys, datetime as dt
import numpy as np
import pandas as pd
import fetch_core as fc


def daterange(d1, d2):
    a = pd.date_range(d1, d2, freq="1D")
    return [d.strftime("%Y-%m-%d") for d in a]


def build_units(stas, days_labels, net, cha):
    return [(net, sta, cha, day, lab) for sta in stas for day, lab in days_labels]


# ---------------------------------------------------------------- targets
def target_yw(smoke):
    """IRIS Community Wavefield OK 2016 — 5 Hz ZLand nodes, 250 Hz, PH5 archive
    (NOT the standard FDSN archive — the YW code there is other epochs).
    Actual deployment: 2016-06-21 .. 2016-07-26. Wind/rain/diurnal anchor."""
    client = fc.make_client("IRISPH5")
    inv = client.get_stations(network="YW", channel="DPZ", level="response",
                              starttime="2016-06-21", endtime="2016-07-27")
    lat = np.mean([s.latitude for n in inv for s in n.stations])
    lon = np.mean([s.longitude for n in inv for s in n.stations])
    wxd, _ = fc.get_weather_daily(lat, lon, "2016-06-22", "2016-07-25")
    stas = fc.pick_stations(inv, 2 if smoke else 16)
    days = daterange("2016-06-23", "2016-07-17")          # 25 full days, all nodes live
    dl = [(d, fc.label_day(wxd.loc[pd.to_datetime(d).date()])) for d in days]
    return client, inv, build_units(stas, dl, "YW", "DPZ"), wxd


def target_lasso(smoke):
    """LASSO 2A 2016 — 10 Hz nodes, 500 Hz, PH5 archive. Rain anchor (radar-confirmed era)."""
    client = fc.make_client("IRISPH5")
    inv = client.get_stations(network="2A", channel="DPZ", level="response",
                              starttime="2016-04-14", endtime="2016-05-11")
    lat = np.mean([s.latitude for n in inv for s in n.stations])
    lon = np.mean([s.longitude for n in inv for s in n.stations])
    wxd, _ = fc.get_weather_daily(lat, lon, "2016-04-14", "2016-05-10")
    stas = fc.pick_stations(inv, 2 if smoke else 12)
    # top-10 rain days + 5 calmest controls
    rainy = wxd.sort_values("rain_sum", ascending=False).head(10).index
    calm = wxd[wxd.rain_sum == 0].sort_values("wind_max").head(5).index
    dl = [(d.strftime("%Y-%m-%d"), fc.label_day(wxd.loc[d]))
          for d in sorted(set(rainy) | set(calm))]
    return client, inv, build_units(stas, dl, "2A", "DPZ"), wxd


def target_zg(smoke):
    """San Jacinto ZG 2014 — 10 Hz nodes, 500 Hz, desert. Wind-on-desert + coupling anchor."""
    client = fc.make_client("EARTHSCOPE")
    inv = client.get_stations(network="ZG", channel="DPZ", level="response",
                              starttime="2014-05-07", endtime="2014-06-14")
    lat = np.mean([s.latitude for n in inv for s in n.stations])
    lon = np.mean([s.longitude for n in inv for s in n.stations])
    wxd, _ = fc.get_weather_daily(lat, lon, "2014-05-08", "2014-06-12")
    stas = fc.pick_stations(inv, 2 if smoke else 10)
    windy = wxd.sort_values("wind_max", ascending=False).head(7).index
    calm = wxd.sort_values("wind_max").head(5).index
    dl = [(d.strftime("%Y-%m-%d"), fc.label_day(wxd.loc[d]))
          for d in sorted(set(windy) | set(calm))]
    return client, inv, build_units(stas, dl, "ZG", "DPZ"), wxd


def target_is_il(smoke):
    """Israel Seismic Network via GEOFON. Open data ends 2021 (TRUAA era restricted);
    best open channels: SHZ @ 40 sps (1-20 Hz band), BHZ @ 20 sps fallback.
    Khamsin (Apr-May) + winter storms (Jan-Feb) 2021. Low-band anchor only —
    rain-band anchoring comes from LASSO/YW."""
    client = fc.make_client("GEOFON")
    inv = client.get_stations(network="IS", channel="SHZ,BHZ", level="response",
                              starttime="2021-01-01", endtime="2021-06-01")
    stas = {}
    for n in inv:
        for s in n.stations:
            chans = {c.code for c in s.channels}
            cha = "SHZ" if "SHZ" in chans else "BHZ"
            stas[s.code] = (s.code, s.latitude, s.longitude, cha)
    stas_all = sorted(stas.values())

    def probe(code, cha):
        """Keep only stations that actually deliver 2021 waveforms."""
        from obspy import UTCDateTime
        for day in ("2021-01-15", "2021-04-15"):
            try:
                t = UTCDateTime(day) + 12 * 3600
                st = client.get_waveforms("IS", code, "*", cha, t, t + 300)
                if len(st) and sum(tr.stats.npts for tr in st) > 0:
                    return True
            except Exception:
                continue
        return False

    # role-ordered candidates: Negev (khamsin) / coastal (surf) / north (rural)
    south = sorted(stas_all, key=lambda s: s[1])[:6]
    coastal = sorted([s for s in stas_all if s[2] < 35.0 and 31.4 <= s[1] <= 33.1],
                     key=lambda s: s[2])[:6]
    north = sorted(stas_all, key=lambda s: -s[1])[:6]
    if "HRFI" in stas:
        south = [stas["HRFI"]] + south
    chosen = []
    for role in (south, coastal, north):
        for s in role:
            if s[0] in [c[0] for c in chosen]:
                continue
            print(f"  probing IS.{s[0]}.{s[3]} ...", flush=True)
            if probe(s[0], s[3]):
                chosen.append(s); break
    chosen = chosen[: (1 if smoke else 3)]
    print(f"  IS stations selected: {[(c[0], c[3]) for c in chosen]}", flush=True)

    units, wxd_all = [], None
    for code, lat, lon, cha in chosen:
        wxd1, _ = fc.get_weather_daily(lat, lon, "2021-01-01", "2021-02-28")  # winter storms
        wxd2, _ = fc.get_weather_daily(lat, lon, "2021-04-01", "2021-05-31")  # khamsin season
        wxd = pd.concat([wxd1, wxd2])
        wxd_all = wxd if wxd_all is None else pd.concat(
            [wxd_all, wxd[~wxd.index.isin(wxd_all.index)]])
        days = [d.strftime("%Y-%m-%d") for d in pd.to_datetime(wxd.index)]
        if smoke:
            days = days[:1]
        units += [("IS", code, cha, d,
                   fc.label_day(wxd.loc[pd.to_datetime(d).date()])) for d in days]
    return client, inv, units, wxd_all


TARGETS = {"yw": target_yw, "lasso": target_lasso, "zg": target_zg, "is_il": target_is_il}


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    smoke = "--smoke" in sys.argv
    shard_i, shard_n = 0, 1
    if "--shard" in sys.argv:                      # --shard i n
        k = sys.argv.index("--shard")
        shard_i, shard_n = int(sys.argv[k + 1]), int(sys.argv[k + 2])
    names = list(TARGETS) if (not args or args[0] == "all") else [args[0]]
    os.makedirs(fc.ROOT, exist_ok=True)
    for name in names:
        # every process writes its own manifest (concurrent CSV appends interleave);
        # merged into manifest.csv by the QA step.
        fc.MANIFEST = os.path.join(fc.ROOT, f"manifest_{name}_{shard_i}.csv")
        print(f"=== target {name} (smoke={smoke}, shard {shard_i}/{shard_n}) ===", flush=True)
        try:
            client, inv, units, wxd = TARGETS[name](smoke)
            units = units[shard_i::shard_n]
            print(f"  {len(units)} units planned", flush=True)
            done, fail = fc.run_units(name, client, inv, units, wxd,
                                      max_units=1 if smoke else None,
                                      log=lambda m: print(m, flush=True))
            print(f"=== {name}: {done} fetched, {fail} failed ===", flush=True)
        except Exception as e:
            import traceback
            print(f"=== {name}: TARGET-LEVEL FAILURE {type(e).__name__}: {e} ===", flush=True)
            traceback.print_exc()


if __name__ == "__main__":
    main()
