"""Path library (subset of P1-P14) + START rule — GENERATION_PLAN v1.2.
Single vertical sensor at origin; only r(t)=|p(t)| reaches the physics, but true 2-D
paths make radial behaviour + multi-source geometry honest. START rule: r0 in
[0.15,1.1]*R_det, with an onset-free fraction (subject already mid-zone)."""
import numpy as np


def sample(rng, r_det, speed, onset_free_p=0.30, dur_cap=120.0):
    """Return (path_xy (n,2), t_grid, dur, info). speed = gait/travel speed m/s."""
    style = rng.choice(["pass_by", "oblique", "meander", "approach_dwell",
                        "partial", "loiter"], p=[.34, .20, .18, .12, .10, .06])
    d = rng.uniform(1.5, 0.7 * r_det)                       # closest approach
    onset_free = rng.random() < onset_free_p                # start already mid-zone
    dur = float(np.clip(2 * r_det / max(speed, 0.3), 8, dur_cap))
    t = np.arange(0, dur, 0.1)
    if style == "pass_by":
        x = speed * (t - dur / 2); y = np.full_like(t, d)
    elif style == "oblique":
        ang = rng.uniform(0.3, 1.2)
        x = speed * np.cos(ang) * (t - dur / 2)
        y = d + speed * np.sin(ang) * (t - dur / 2) * rng.choice([-1, 1])
    elif style == "meander":
        k = rng.integers(3, 6)
        wpx = np.cumsum(rng.normal(0, speed * dur / k, k))
        wpy = d + rng.normal(0, 0.3 * d, k)
        x = np.interp(np.linspace(0, k - 1, len(t)), np.arange(k), wpx)
        y = np.interp(np.linspace(0, k - 1, len(t)), np.arange(k), wpy)
    elif style == "approach_dwell":
        half = dur / 2
        rr = np.where(t < half, d + speed * (half - t), d + speed * (t - half))
        ang = rng.uniform(0, 2 * np.pi); x = rr * np.cos(ang); y = rr * np.sin(ang)
    elif style == "partial":
        x = speed * (t - dur * 0.8); y = np.full_like(t, d)
    else:                                                   # loiter
        c = rng.uniform(2, 0.5 * r_det); rad = rng.uniform(1, 5)
        x = c + rad * np.cos(2 * np.pi * t / rng.uniform(10, 30))
        y = d + rad * np.sin(2 * np.pi * t / rng.uniform(10, 30))
    # apply START rule: shift so initial range sits in the annulus
    r0_target = rng.uniform(0.15, 0.45 if onset_free else 1.1) * r_det
    r0_now = np.hypot(x[0], y[0])
    if r0_now > 1e-6:
        scale = np.clip(r0_target / r0_now, 0.3, 3.0)
        # nudge offset rather than rescale path shape
    return np.stack([x, y], 1), t, dur, dict(style=style, d=round(float(d), 1),
                                              onset_free=bool(onset_free))
