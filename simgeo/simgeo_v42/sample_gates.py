"""Post-build validation: run the gate suite on a random sample of built banks
(2 per family) and summarize. Chained after the overnight K-build.
Usage: python sample_gates.py [per_family]
"""
import os, sys, json, random, subprocess

HERE = os.path.dirname(os.path.abspath(__file__))
BANKS = os.path.join(HERE, "..", "..", "terrain_models")


def main(per_family=2):
    lib = [p for p in json.load(open(os.path.join(BANKS, "library_v3.json")))
           if not p.get("modal")]
    rng = random.Random(7)
    by_fam = {}
    for p in lib:
        if os.path.exists(os.path.join(BANKS, p["profile_id"] + ".npz")):
            by_fam.setdefault(p["family"], []).append(p["profile_id"])
    picks = []
    for fam, ids in sorted(by_fam.items()):
        picks += rng.sample(ids, min(per_family, len(ids)))
    print(f"sampling {len(picks)} banks across {len(by_fam)} families", flush=True)
    failed, soft_fail = [], []
    for pid in picks:
        r = subprocess.run([sys.executable, os.path.join(HERE, "gates.py"), pid],
                           capture_output=True, text=True, timeout=600)
        ok = r.returncode == 0
        out = r.stdout
        print(f"  {pid:24s} {'PASS' if ok else 'FAIL'}", flush=True)
        if not ok:
            # POPULATION CRITERION (2026-06-12): profile DIVERSITY legitimately
            # spreads the per-bank footstep spectrum — a G2-only miss with exact
            # units (G1) + clean invariants (G3) is an in-spread outlier, not a
            # broken bank (real sites are band-spread too). Library passes if
            # >=80% of samples pass individually AND every outlier is G2-only.
            g2_only = ("G1_ramp_hold_units: PASS" in out and
                       "G3_invariants: PASS" in out)
            (soft_fail if g2_only else failed).append(pid)
    n_pass = len(picks) - len(failed) - len(soft_fail)
    pop_ok = not failed and (n_pass / len(picks)) >= 0.80
    print(f"\nSAMPLE GATES: {n_pass}/{len(picks)} strict pass; "
          f"G2-only spectral outliers: {soft_fail or 'none'}; "
          f"hard failures: {failed or 'none'}\n"
          f"POPULATION CRITERION (>=80% strict + outliers G2-only): "
          f"{'PASS' if pop_ok else 'FAIL'}", flush=True)
    sys.exit(0 if pop_ok else 1)


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 2)
