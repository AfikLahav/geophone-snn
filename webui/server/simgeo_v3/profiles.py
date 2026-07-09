"""Soil-profile library sampler — K=300 exactly (spec §1 + §1b; v3 TERRAIN-CATALOG, 2026-06-15).

282 half-space families -> pyprop8 banks; 18 modal floors (no bank).
Per-profile draws (spec §1b): log-uniform Vs; 2-4 layers (<=5% uniform half-space);
per-family top-layer thickness priors; ~18% bedrock-contact jumps (x3-8); Vp/Vs
1.6-2.5; Gardner density +-10% (snow overrides); Q log-uniform/family; coupling
Q_c 1-20; inverse branch for asphalt/concrete/paving/sabkha; frozen = scaled base;
splits 15% val + 15% test PER FAMILY, assigned here.
Units for pyprop8: km, km/s, g/cm^3.

v3 TERRAIN-CATALOG fixes (gap_study/V3_SYNTHETIC_DATASET_PLAN.md TIER D1/D2,
gap_study/v3_verify/V4_terrain_soil.md):
  D1a cut frozen+snow to ~0.5% of catalog (no Israeli permafrost; snow Hermon-only).
  D1b add terra_rossa (LITERATURE-ANALOG Vs, FLAGGED), cover_basalt (670-2400),
      chalk_marl (marl 350-560 / chalk 620-1220) families with Israeli-measured Vs.
  D1c cap half-space Vs at 2580 m/s (Israeli in-situ ceiling = compact dolomite;
      3300 was Vp/lab, not in-situ Vs).  Was 3300/3200.
  D1d kurkar floor 600 -> 475; sabkha mud 80 -> 120.
  D2  ~12% BURIED velocity-inversion variant in _stack_normal (one interior layer
      x U(0.5,0.8), monotonicity guard lifted for that realization only).
  H1  build_library() is SELF-CONTAINED: it now emits the pavement `plate` SDOF
      key (f0/zeta/h) that scenes.py consumes, so library.json regenerates from
      this file alone (the prior version relied on an external plate injector and
      had externally-added keys it could not reproduce).
  CONFOUND GUARD (V4 critical, Zaslavsky 2012): site f0 / layering are varied
      INDEPENDENTLY of Vs30 — coupling_fc, layer count, thickness, bedrock-jump
      and buried-inversion draws do NOT key off vs_top, so terrain != Vs30.

Vs PROVENANCE (disclosure ledger):
  LITERATURE-GROUNDED (Israeli field Vs): clay, sand, gravel, wet_soil, rock,
    kurkar, cover_basalt, chalk_marl, sabkha, bedrock-jump structure, 2580 cap
    -- Gorstein & Ezersky 2015 (IJGE); Zaslavsky et al. 2012 (Nat.Sci.4:631);
    Ezersky & Livne 2013 / Ezersky & Legchenko 2014 (Dead Sea); Darvasi & Agnon
    2019 (Solid Earth 10:379) + Darvasi 2021 (Earthquake Spectra) Cover Basalt.
  FLAGGED UNVALIDATED (literature-analog, no Israeli field Vs): terra_rossa,
    loess -- Singer 2007; Durn 2003; Crouvi et al. 2008; MASW silt analogs.
  Frozen/snow Vs are retained as DENSE-snow analogs (disclose; H5) but down-weighted.
"""
import numpy as np

# family: n, vs_top range (m/s), Q range, coupling fc (Hz), top-layer thickness (m)
# flags: wet / inverse / frozen / snow / sabkha / plate (pavement SDOF) / vs_flag
#   vs_flag="analog" marks families whose Vs is literature-analog, not Israeli field Vs.
FAMILIES = {
    "soft_soil":   dict(n=28, vs=(100, 300),   q=(5, 20),    fc=(30, 150),  top=(0.5, 8)),
    "loess":       dict(n=22, vs=(150, 400),   q=(5, 25),    fc=(30, 150),  top=(2, 25),
                        vs_flag="analog"),
    "sand":        dict(n=26, vs=(100, 500),   q=(10, 40),   fc=(30, 150),  top=(1, 15)),
    "gravel":      dict(n=24, vs=(300, 750),   q=(20, 60),   fc=(60, 250),  top=(0.5, 6)),
    "clay":        dict(n=28, vs=(150, 800),   q=(5, 30),    fc=(50, 200),  top=(1, 20)),
    "wet_soil":    dict(n=18, vs=(200, 600),   q=(20, 60),   fc=(80, 300),  top=(1, 10), wet=True),
    "rock":        dict(n=24, vs=(800, 2500),  q=(50, 200),  fc=(100, 400), top=(0.2, 3)),
    "kurkar":      dict(n=10, vs=(475, 1500),  q=(30, 80),   fc=(80, 300),  top=(1, 10)),
    # NEW (D1b) -------------------------------------------------------------
    "terra_rossa": dict(n=16, vs=(250, 500),   q=(5, 30),    fc=(50, 200),  top=(0.2, 4),
                        vs_flag="analog", bedrock_p=0.65),  # thin stiff red clay over carbonate
    "cover_basalt":dict(n=10, vs=(670, 2400),  q=(40, 180),  fc=(100, 400), top=(0.3, 8),
                        bimodal=(670, 1390, 1680, 2400)),    # weathered / fresh split
    "chalk_marl":  dict(n=14, vs=(350, 1220),  q=(20, 100),  fc=(80, 350),  top=(1, 15),
                        bimodal=(350, 560, 620, 1220)),      # soft marl / firm chalk split
    # ----------------------------------------------------------------------
    "dirt_road":   dict(n=14, vs=(250, 600),   q=(15, 50),   fc=(60, 250),  top=(0.1, 0.5)),
    "asphalt":     dict(n=20, vs=(1000, 1800), q=(20, 60),   fc=(150, 400), inverse=True,
                        plate=dict(f0=(40, 80), zeta=(0.10, 0.245), h=(0.05, 0.20))),
    "concrete":    dict(n=16, vs=(2000, 2500), q=(50, 150),  fc=(150, 500), inverse=True,
                        plate=dict(f0=(48, 82), zeta=(0.03, 0.05), h=(0.12, 0.30))),
    "paving":      dict(n=6,  vs=(800, 1800),  q=(30, 80),   fc=(100, 400), inverse=True,
                        plate=dict(f0=(58, 80), zeta=(0.08, 0.18), h=(0.06, 0.16))),
    # frozen + snow cut to ~0.5% of the 300-profile catalog (D1a) -----------
    "frozen":      dict(n=1,  vs=(100, 500),   q=(5, 40),    fc=(100, 400), top=(0.5, 8), frozen=True),
    "snow":        dict(n=1,  vs=(100, 300),   q=(5, 20),    fc=(20, 100),  top=(0.1, 1.0), snow=True),
    "sabkha":      dict(n=4,  vs=(400, 800),   q=(5, 20),    fc=(50, 200),  inverse=True, sabkha=True),
}
K_BANKS = sum(f["n"] for f in FAMILIES.values())         # 282
assert K_BANKS == 282, f"K_BANKS={K_BANKS}, expected 282"

N_FLOORS = {"susp_concrete": 10, "wood_floor": 8}        # modal branch, no banks
K_TOTAL = K_BANKS + sum(N_FLOORS.values())               # 300

VS_HS_CAP = 2580.0    # D1c: Israeli in-situ Vs ceiling (compact dolomite); was 3300
_GROWTH_CAP = 2600.0  # stop intra-stack stiffening just above the HS cap (was 3200)
P_BURIED_INV = 0.22   # D2: buried-inversion draw on eligible (>=3-layer) stacks;
                      # realized fraction lands ~10-13% of all banks (only multi-layer
                      # stacks are eligible) -- inside the spec's 10-15% target.


def _loguni(rng, lo, hi):
    return float(np.exp(rng.uniform(np.log(lo), np.log(hi))))


def _vp_rho(vs_ms, rng, wet=False, frozen=False):
    vp = vs_ms * rng.uniform(1.6, 2.5)
    if wet:
        vp = max(vp, rng.uniform(1500, 1900))
    if frozen:
        vp = max(vp, rng.uniform(1500, 2500))            # ice-cemented
    vp = min(vp, 6000.0)                                 # granite-class ceiling; Vp also
    rho = np.clip(310 * (vp ** 0.25) * rng.uniform(0.9, 1.1), 1300, 2700)
    return vp, rho                                       # drives the ghost-damping cost


def _draw_vs_top(fam, rng):
    """Top-layer Vs draw. Bimodal families (cover_basalt, chalk_marl) deliberately
    sample BOTH lobes (weathered/fresh, marl/chalk) so the family is not collapsed
    to one sub-material; otherwise log-uniform over the family range."""
    bi = fam.get("bimodal")
    if bi:
        lo1, hi1, lo2, hi2 = bi
        return _loguni(rng, lo1, hi1) if rng.random() < 0.5 else _loguni(rng, lo2, hi2)
    return _loguni(rng, *fam["vs"])


def _stack_normal(fam, rng):
    """Normal (stiffening-with-depth) stack in m/s + m. Returns ([(thick_m, vs_ms)...], buried_inv).

    CONFOUND GUARD: layer count, thicknesses, the bedrock-jump draw and the buried-
    inversion draw are independent of vs_top, so f0/layering vary independently of Vs30.
    D2: with prob P_BURIED_INV, one interior layer is softened (x U(0.5,0.8)) and the
    monotonicity guard is lifted for that realization only (buried low-velocity layer)."""
    vs_top = _draw_vs_top(fam, rng)
    nlay = int(rng.choice([1, 2, 3, 4], p=[0.05, 0.40, 0.35, 0.20]))
    t_lo, t_hi = fam.get("top", (0.5, 8))
    # buried-LVL only meaningful with >=3 layers (an interior layer must exist)
    buried_inv = (nlay >= 3 and not fam.get("no_inv") and rng.random() < P_BURIED_INV)
    stack, vs = [], vs_top
    for j in range(nlay - 1):
        th = _loguni(rng, t_lo, t_hi) if j == 0 else _loguni(rng, 0.5, 10.0)
        stack.append((th, vs))
        nxt = vs * rng.uniform(1.3, 2.2)
        if nxt >= _GROWTH_CAP:                            # cap would flatten growth:
            break                                        # go straight to half-space
        vs = nxt
    # D2: plant a buried low-velocity layer. Soften one interior layer RELATIVE to its
    # stiffer predecessor (x U(0.5,0.8)) so vs[k] < vs[k-1] is guaranteed; the half-space
    # appended below is stiffer, so a genuine soft-between-stiff valley results. The
    # monotonicity guard is lifted for this realization only.
    if buried_inv and len(stack) >= 2:
        k = int(rng.integers(1, len(stack)))             # interior layer (not the surface)
        th_k, _ = stack[k]
        vs_prev = stack[k - 1][1]
        stack[k] = (th_k, vs_prev * rng.uniform(0.5, 0.8))
    else:
        buried_inv = False                               # guard: no interior layer to soften
    # bedrock-contact jump (Israeli thin-soil-over-limestone signature).
    # terra_rossa: thin stiff soil over hard carbonate -> elevated jump probability.
    bedrock_p = fam.get("bedrock_p", 0.18)
    if len(stack) >= 1 and rng.random() < bedrock_p:
        vs = stack[-1][1] * rng.uniform(3.0, 8.0)
    vs_hs = float(min(max(vs, stack[-1][1] * 1.05 if stack else vs), VS_HS_CAP))
    if stack and not buried_inv and vs_hs <= stack[-1][1]:  # keep monotone unless buried-inv
        stack = stack[:-1]
    stack.append((np.inf, vs_hs))
    return stack, buried_inv


def _stack_inverse(fam, rng):
    """Stiff thin top over soft base (asphalt/concrete/paving/sabkha)."""
    vs_top = _loguni(rng, *fam["vs"])
    if fam.get("sabkha"):
        t_top = rng.uniform(0.1, 0.5)                    # salt crust
        vs_base = rng.uniform(120, 250)                  # D1d: Dead Sea brine mud (was 80-200)
    else:
        t_top = rng.uniform(0.05, 0.3)
        vs_base = rng.uniform(150, 400)
    stack = [(t_top, vs_top)]
    vs = vs_base
    for _ in range(int(rng.integers(1, 3))):
        th = _loguni(rng, 0.5, 6.0)
        stack.append((th, vs))
        vs *= rng.uniform(1.3, 2.0)
    stack.append((np.inf, float(min(vs, VS_HS_CAP))))
    return stack


def _plate_sdof(fam, rng):
    """Pavement SDOF plate-resonance filter params consumed by scenes.py
    (H = f0^2/((f0^2-f^2)+2i*zeta*f0*f)). Self-contained here for reproducibility (H1)."""
    p = fam["plate"]
    return dict(f0=round(float(rng.uniform(*p["f0"])), 1),
                zeta=round(float(rng.uniform(*p["zeta"])), 3),
                h=round(float(rng.uniform(*p["h"])), 3))


def sample_profile(family, fam, idx, rng):
    buried_inv = False
    if fam.get("inverse"):
        stack_ms = _stack_inverse(fam, rng)
    elif fam.get("snow"):
        # snow blanket over (often frozen) ground
        t_snow = _loguni(rng, *fam["top"])
        vs_snow = _loguni(rng, *fam["vs"])
        base, _ = _stack_normal(dict(vs=(200, 600), top=(0.5, 6)), rng)
        scale = rng.uniform(1.0, 3.0)                    # base may be frozen
        stack_ms = [(t_snow, vs_snow)] + [(t, min(v * scale, VS_HS_CAP)) for t, v in base]
    else:
        stack_ms, buried_inv = _stack_normal(fam, rng)
        if fam.get("frozen"):
            scale = rng.uniform(2.0, 4.0)
            stack_ms = [(t, min(v * scale, VS_HS_CAP)) for t, v in stack_ms]

    wet = bool(fam.get("wet") or fam.get("sabkha"))
    layers = []
    for i, (th_m, vs_ms) in enumerate(stack_ms):
        if fam.get("snow") and i == 0:                   # Gardner invalid for snow
            vp = rng.uniform(400, 1500)
            rho = rng.uniform(100, 500)
        else:
            vp, rho = _vp_rho(vs_ms, rng,
                              wet=(wet and i > 0) or fam.get("wet", False),
                              frozen=bool(fam.get("frozen")))
        layers.append((th_m / 1000.0 if np.isfinite(th_m) else np.inf,
                       vp / 1000.0, vs_ms / 1000.0, rho / 1000.0))

    q = _loguni(rng, *fam["q"])
    if fam.get("frozen"):
        q *= rng.uniform(2.0, 5.0)
    fc = _loguni(rng, *fam["fc"])
    vr_src = stack_ms[1][1] if fam.get("inverse") else stack_ms[0][1]
    rec = dict(profile_id=f"{family}_{idx}", family=family,
               layers=[list(l) for l in layers],
               q=float(q), coupling_fc=float(fc),
               coupling_q=float(rng.uniform(1, 20)),
               vr_ms=float(0.92 * vr_src), vs_top_ms=float(stack_ms[0][1]),
               wet=wet, inverse=bool(fam.get("inverse", False)),
               buried_inversion=bool(buried_inv),
               vs_analog=bool(fam.get("vs_flag") == "analog"),
               modal=False)
    if fam.get("plate"):                                 # H1: emit pavement SDOF here
        rec["plate"] = _plate_sdof(fam, rng)
        rec["plate_model"] = True
    return rec


def build_library(seed=20260611):
    rng = np.random.default_rng(seed)
    lib = []
    for family, fam in FAMILIES.items():
        for i in range(fam["n"]):
            lib.append(sample_profile(family, fam, i, rng))
    # modal floors (D4 not in TERRAIN-CATALOG scope; left as-is)
    for i in range(N_FLOORS["susp_concrete"]):
        lib.append(dict(profile_id=f"susp_concrete_{i}", family="susp_concrete",
                        modal=True, f0=float(rng.uniform(5, 15)),
                        zeta=float(rng.uniform(0.02, 0.05)),
                        nmodes=int(rng.integers(4, 9)),
                        coupling_fc=float(rng.uniform(80, 300)),
                        coupling_q=float(rng.uniform(1, 20)), vs_top_ms=2000.0))
    for i in range(N_FLOORS["wood_floor"]):
        lib.append(dict(profile_id=f"wood_floor_{i}", family="wood_floor",
                        modal=True, f0=float(rng.uniform(10, 25)),
                        zeta=float(rng.uniform(0.02, 0.08)),
                        nmodes=int(rng.integers(3, 7)),
                        coupling_fc=float(rng.uniform(60, 250)),
                        coupling_q=float(rng.uniform(1, 20)), vs_top_ms=600.0))
    # splits: 15% val + 15% test PER FAMILY (shuffled, deterministic, self-contained)
    fams = {}
    for p in lib:
        fams.setdefault(p["family"], []).append(p)
    for family, members in fams.items():
        order = rng.permutation(len(members))
        n = len(members)
        n_test = max(1, int(round(0.15 * n)))
        n_val = max(1, int(round(0.15 * n)))
        for rank, oi in enumerate(order):
            members[oi]["split"] = ("test" if rank < n_test else
                                    "val" if rank < n_test + n_val else "train")
    return lib


if __name__ == "__main__":
    lib = build_library()
    banks = [p for p in lib if not p["modal"]]
    print(f"total={len(lib)} banks={len(banks)} floors={len(lib)-len(banks)}")
