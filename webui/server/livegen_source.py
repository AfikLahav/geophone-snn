"""Dataset mode source: live-generates FRESH, never-trained v3 scenes via
`simgeo_v3.gen_scene` (seeds >= 9_000_000, disjoint from the training corpus) and
streams their windows with the ground-truth labels alongside the model prediction.

Needs `simgeo_v3` + `simgeo_banks/library_v3.json` + the per-profile .npz banks on
the host (this workstation). `available()` reports presence so the UI can hide the
mode where they're absent.
"""
import sys, os, json, random
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
# prefer the vendored copy (self-contained); fall back to the parent repo for dev.
# gen_scene loads its banks from <SIMGEO_V3>/../simgeo_banks, so the bank dir auto-follows.
if os.path.isdir(os.path.join(HERE, "simgeo_v3")):
    SIMGEO_V3 = os.path.join(HERE, "simgeo_v3")
    LIB_PATH = os.path.join(HERE, "simgeo_banks", "library_v3.json")
else:
    _ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
    SIMGEO_V3 = os.path.join(_ROOT, "simgeo_v3")
    LIB_PATH = os.path.join(_ROOT, "simgeo_banks", "library_v3.json")
CLASSES = ["human", "vehicle", "animal", "nothing", "mixed"]

_gc = None
_BANKS = None


def available():
    return os.path.isdir(SIMGEO_V3) and os.path.exists(LIB_PATH)


def _load():
    global _gc, _BANKS
    if _gc is None:
        if SIMGEO_V3 not in sys.path:
            sys.path.insert(0, SIMGEO_V3)
        import generate_corpus as gc
        _gc = gc
        _BANKS = [p for p in json.load(open(LIB_PATH)) if not p.get("modal")]
    return _gc, _BANKS


class LiveGenSource:
    _next_sid = 9_000_000

    def __init__(self, coarse=None, family=None, subkind=None, loop=True):
        self.gc, self.banks = _load()
        self.coarse = coarse if coarse in CLASSES else None
        self.family = family or None
        self.subkind = subkind or None
        self.loop = loop
        self.name = "livegen"
        self.last_truth = None
        self.wins = []
        self._gen_new_scene()

    def _pick_spec(self):
        gc = self.gc
        # demo-friendly weighting: favour source classes, keep nothing/mixed for variety
        coarse = self.coarse or random.choices(CLASSES, weights=[0.26, 0.26, 0.20, 0.14, 0.14])[0]
        pool = [p for p in self.banks if (self.family is None or p["family"] == self.family)] or self.banks
        if coarse in ("vehicle", "mixed"):
            pool = [p for p in pool if p["vs_top_ms"] >= gc.VEH_MIN_VS_V3] or pool
        pid = random.choice(pool)["profile_id"]
        menu = {"human": gc.HUMAN, "vehicle": gc.VEHICLE, "animal": gc.ANIMAL,
                "nothing": gc.NOTHING, "mixed": gc.MIXED}[coarse]
        sub = self.subkind or random.choice(menu)
        sid = LiveGenSource._next_sid
        LiveGenSource._next_sid += 1
        return (sid, pid, "live", coarse, sub)

    def _gen_new_scene(self):
        # retry: short scenes can yield 0 full windows
        for _ in range(8):
            spec = self._pick_spec()
            sr, wr, wrows = self.gc.gen_scene(spec)
            out_mv = (np.frombuffer(wr[4], np.float32) + np.frombuffer(wr[5], np.float32)).astype(np.float32)
            wins = [self._win_truth(r) for r in wrows if int(r[1] * 1000) + 3000 <= len(out_mv)]
            if wins:
                self.out_mv, self.wins, self.meta, self.i = out_mv, wins, self._meta(sr, spec), 0
                return
        self.out_mv, self.wins, self.meta, self.i = out_mv, wins, self._meta(sr, spec), 0

    @staticmethod
    def _win_truth(row):
        lv = {"human": row[2], "car": row[5], "animal": row[8]}      # vehicle_level -> "car"
        active = [c for c, l in lv.items() if l != "none"]
        coarse = "mixed" if len(active) >= 2 else (active[0] if active else "nothing")
        return {"t0": float(row[1]),
                "truth": {"coarse": coarse, "active": active,
                          "human_level": row[2], "car_level": row[5], "animal_level": row[8],
                          "common_snr": round(float(row[11]), 1), "zone": row[12], "activity": int(row[13])}}

    @staticmethod
    def _meta(sr, spec):
        try:
            cfg = json.loads(sr[30])          # config_json is the LAST scene col (index 30)
        except Exception:
            cfg = {}
        return {"family": sr[2], "terrain_vs": round(float(sr[3])), "coarse": sr[5], "subkind": str(sr[6]),
                "n_subjects": int(sr[26]) if sr[26] is not None else 0, "dur_s": float(sr[7]),
                "closest_approach_m": sr[10], "speed_ms": sr[11], "path_style": sr[12],
                "sid": spec[0], "counts": cfg.get("counts", {})}

    def read_window(self):
        if self.i >= len(self.wins):
            if not self.loop:
                return None
            self._gen_new_scene()
        w = self.wins[self.i]
        self.last_truth = w["truth"]
        s = int(w["t0"] * 1000)
        seg = self.out_mv[s:s + 3000]
        self.i += 1
        return seg if len(seg) == 3000 else None

    def scene_meta(self):
        return self.meta

    def close(self):
        pass
