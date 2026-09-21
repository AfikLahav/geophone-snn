"""Stall-aware supervisor for fetch_run.py (tonight's lesson: archive connections
can hang silently for hours without tripping client timeouts).

Runs fetch_run.py <target> [--shard i n] as a child; progress = new files
appearing under $GEO_NOISE_RAW/<dir>. If no new file for STALL_MIN
minutes, kill + restart (the fetcher skips already-downloaded units). Exits 0
when the child completes normally; exits 1 after MAX_RESTARTS.
Usage: python fetch_watchdog.py <yw|lasso|zg|is_il> [--shard i n]
"""
import os, sys, time, glob, subprocess

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.environ.get("GEO_NOISE_RAW", os.path.abspath(os.path.join(HERE, "..", "..", "..", "geophone_real_noise")))
DIRS = {"yw": "yw", "lasso": "lasso", "zg": "zg", "is_il": "is_il"}
STALL_MIN = int(os.environ.get("STALL_MIN", "25"))   # per-target override:
# LASSO full-day PH5 extractions can queue server-side >25 min; killing them
# mid-queue made an infinite restart loop (12 attempts, 2026-06-11 night)
MAX_RESTARTS = 12


def n_files(target):
    return len(glob.glob(os.path.join(ROOT, DIRS[target], "*.npz")))


def main():
    target = sys.argv[1]
    extra = sys.argv[2:]
    tag = f"{target} {' '.join(extra)}".strip()
    for attempt in range(1, MAX_RESTARTS + 1):
        print(f"[watchdog] start {tag} (attempt {attempt})", flush=True)
        child = subprocess.Popen([sys.executable, os.path.join(HERE, "fetch_run.py"),
                                  target] + extra)
        last_n = n_files(target)
        last_progress = time.time()
        while True:
            rc = child.poll()
            if rc is not None:
                if rc == 0:
                    print(f"[watchdog] {tag} completed", flush=True)
                    return 0
                print(f"[watchdog] {tag} exited rc={rc}; restarting", flush=True)
                break
            time.sleep(60)
            now_n = n_files(target)
            if now_n > last_n:
                last_n, last_progress = now_n, time.time()
            elif time.time() - last_progress > STALL_MIN * 60:
                print(f"[watchdog] {tag} STALLED ({STALL_MIN} min, "
                      f"{now_n} files) — killing child", flush=True)
                child.kill()
                child.wait()
                break
        time.sleep(30)
    print(f"[watchdog] {tag} gave up after {MAX_RESTARTS} attempts", flush=True)
    return 1


if __name__ == "__main__":
    sys.exit(main())
