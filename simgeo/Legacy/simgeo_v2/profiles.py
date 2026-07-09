"""Soil-profile library sampler — K=300 exactly (spec §1 + §1b, amended 2026-06-11).

282 half-space families -> pyprop8 banks; 18 modal floors (no bank).
Per-profile draws (spec §1b): log-uniform Vs; 2-4 layers (<=5% uniform half-space);
per-family top-layer thickness priors; ~18% bedrock-contact jumps (x3-8); Vp/Vs
1.6-2.5; Gardner density +-10% (snow overrides); Q log-uniform/family; coupling
Q_c 1-20; inverse branch for asphalt/concrete/paving/sabkha; frozen = scaled base;
splits 15% val + 15% test PER FAMILY, assigned here.
Units for pyprop8: km, km/s, g/cm^3.
"""
import numpy as np

# family: n, vs_top range (m/s), Q range, coupling fc (Hz), top-layer thickness (m)
FAMILIES = {
    "soft_soil": dict(n=30, vs=(100, 300),   q=(5, 20),    fc=(30, 150),  top=(0.5, 8)),
    "loess":     dict(n=24, vs=(150, 400),   q=(5, 25),    fc=(30, 150),  top=(2, 25)),
    "sand":      dict(n=30, vs=(100, 500),   q=(10, 40),   fc=(30, 150),  top=(1, 15)),
    "gravel":    dict(n=28, vs=(300, 750),   q=(20, 60),   fc=(60, 250),  top=(0.5, 6)),
    "clay":      dict(n=32, vs=(200, 800),   q=(5, 30),    fc=(50, 200),  top=(1, 20)),
    "wet_soil":  dict(n=20, vs=(200, 600),   q=(20, 60),   fc=(80, 300),  top=(1, 10), wet=True),
    "rock":      dict(n=30, vs=(800, 2500),  q=(50, 200),  fc=(100, 400), top=(0.2, 3)),
    "kurkar":    dict(n=10, vs=(600, 1500),  q=(30, 80),   fc=(80, 300),  top=(1, 10)),
    "dirt_road": dict(n=14, vs=(250, 600),   q=(15, 50),   fc=(60, 250),  top=(0.1, 0.5)),
    "asphalt":   dict(n=20, vs=(1000, 1800), q=(20, 60),   fc=(150, 400), inverse=True),
    "concrete":  dict(n=16, vs=(2000, 2500), q=(50, 150),  fc=(150, 500), inverse=True),
    "paving":    dict(n=6,  vs=(800, 1800),  q=(30, 80),   fc=(100, 400), inverse=True),
    "frozen":    dict(n=12, vs=(100, 500),   q=(5, 40),    fc=(100, 400), top=(0.5, 8), frozen=True),
    "snow":      dict(n=6,  vs=(100, 300),   q=(5, 20),    fc=(20, 100),  top=(0.1, 1.0), snow=True),
    "sabkha":    dict(n=4,  vs=(400, 800),   q=(5, 20),    fc=(50, 200),  inverse=True, sabkha=True),
}
K_BANKS = sum(f["n"] for f in FAMILIES.values())         # 282

N_FLOORS = {"susp_concrete": 10, "wood_floor": 8}        # modal branch, no banks
K_TOTAL = K_BANKS + sum(N_FLOORS.values())               # 300


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


def _stack_normal(fam, rng):
    """Normal (stiffening-with-depth) stack in m/s + m. Returns [(thick_m, vs_ms)...]."""
    vs_top = _loguni(rng, *fam["vs"])
    nlay = int(rng.choice([1, 2, 3, 4], p=[0.05, 0.40, 0.35, 0.20]))
    t_lo, t_hi = fam.get("top", (0.5, 8))
    stack, vs = [], vs_top
    for j in range(nlay - 1):
        th = _loguni(rng, t_lo, t_hi) if j == 0 else _loguni(rng, 0.5, 10.0)
        stack.append((th, vs))
        nxt = vs * rng.uniform(1.3, 2.2)
        if nxt >= 3200.0:                                # cap would flatten growth:
            break                                        # go straight to half-space
        vs = nxt
    # bedrock-contact jump (Israeli thin-soil-over-limestone signature)
    if len(stack) >= 1 and rng.random() < 0.18:
        vs = stack[-1][1] * rng.uniform(3.0, 8.0)
    vs_hs = float(min(max(vs, stack[-1][1] * 1.05 if stack else vs), 3300.0))
    if stack and vs_hs <= stack[-1][1]:                  # keep strictly monotone
        stack = stack[:-1]
    stack.append((np.inf, vs_hs))
    return stack


def _stack_inverse(fam, rng):
    """Stiff thin top over soft base (asphalt/concrete/paving/sabkha)."""
    vs_top = _loguni(rng, *fam["vs"])
    if fam.get("sabkha"):
        t_top = rng.uniform(0.1, 0.5)                    # salt crust
        vs_base = rng.uniform(80, 200)                   # saturated mud
    else:
        t_top = rng.uniform(0.05, 0.3)
        vs_base = rng.uniform(150, 400)
    stack = [(t_top, vs_top)]
    vs = vs_base
    for _ in range(int(rng.integers(1, 3))):
        th = _loguni(rng, 0.5, 6.0)
        stack.append((th, vs))
        vs *= rng.uniform(1.3, 2.0)
    stack.append((np.inf, float(min(vs, 3300.0))))
    return stack


def sample_profile(family, fam, idx, rng):
    if fam.get("inverse"):
        stack_ms = _stack_inverse(fam, rng)
    elif fam.get("snow"):
        # snow blanket over (often frozen) ground
        t_snow = _loguni(rng, *fam["top"])
        vs_snow = _loguni(rng, *fam["vs"])
        base = _stack_normal(dict(vs=(200, 600), top=(0.5, 6)), rng)
        scale = rng.uniform(1.0, 3.0)                    # base may be frozen
        stack_ms = [(t_snow, vs_snow)] + [(t, min(v * scale, 3300.0)) for t, v in base]
    else:
        stack_ms = _stack_normal(fam, rng)
        if fam.get("frozen"):
            scale = rng.uniform(2.0, 4.0)
            stack_ms = [(t, min(v * scale, 3300.0)) for t, v in stack_ms]

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
    return dict(profile_id=f"{family}_{idx}", family=family,
                layers=[list(l) for l in layers],
                q=float(q), coupling_fc=float(fc),
                coupling_q=float(rng.uniform(1, 20)),
                vr_ms=float(0.92 * vr_src), vs_top_ms=float(stack_ms[0][1]),
                wet=wet, inverse=bool(fam.get("inverse", False)),
                modal=False)


def build_library(seed=20260611):
    rng = np.random.default_rng(seed)
    lib = []
    for family, fam in FAMILIES.items():
        for i in range(fam["n"]):
            lib.append(sample_profile(family, fam, i, rng))
    # modal floors
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
    # splits: 15% val + 15% test PER FAMILY (shuffled, deterministic)
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
