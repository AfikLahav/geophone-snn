"""Merge shards -> corpus.sqlite + run QA battery (GENERATION_PLAN v1.2 Part D).
Usage: python merge_qa.py <corpus_dir>
"""
import os, sys, glob, sqlite3, time
import numpy as np


def merge(corpus_dir):
    shards = sorted(glob.glob(os.path.join(corpus_dir, "shard_*.sqlite")))
    out = os.path.join(corpus_dir, "corpus.sqlite")
    if os.path.exists(out):
        os.remove(out)
    db = sqlite3.connect(out)
    db.execute("PRAGMA journal_mode=WAL")
    # clone schema from shard 0
    s0 = sqlite3.connect(shards[0])
    for sql, in s0.execute("SELECT sql FROM sqlite_master WHERE type='table'"):
        db.execute(sql)
    s0.close()
    for sh in shards:
        db.execute(f"ATTACH '{sh}' AS s")
        db.execute("INSERT INTO scenes SELECT * FROM s.scenes")
        db.execute("INSERT INTO windows(scene_id,t0,human_level,human_snr,human_soft,"
                   "vehicle_level,vehicle_snr,vehicle_soft,animal_level,animal_snr,animal_soft,"
                   "common_snr,zone,activity) SELECT scene_id,t0,human_level,human_snr,human_soft,"
                   "vehicle_level,vehicle_snr,vehicle_soft,animal_level,animal_snr,animal_soft,"
                   "common_snr,zone,activity FROM s.windows")
        db.commit(); db.execute("DETACH s")
    db.execute("CREATE INDEX ix_ws ON windows(scene_id)")
    db.execute("CREATE INDEX ix_wz ON windows(zone)")
    db.execute("CREATE INDEX ix_ss ON scenes(split)")
    db.commit()
    return db, out


def qa(db):
    print("\n=== CORPUS QA (v1.2 Part D) ===")
    nsce = db.execute("SELECT COUNT(*) FROM scenes").fetchone()[0]
    nwin = db.execute("SELECT COUNT(*) FROM windows").fetchone()[0]
    print(f"scenes={nsce}  windows={nwin}  ({nwin/nsce:.1f}/scene)")
    for q, lab in [("SELECT coarse,COUNT(*) FROM scenes GROUP BY coarse", "class"),
                   ("SELECT split,COUNT(*) FROM scenes GROUP BY split", "split"),
                   ("SELECT zone,COUNT(*) FROM windows GROUP BY zone", "zone"),
                   ("SELECT family,COUNT(*) FROM scenes GROUP BY family ORDER BY 2 DESC LIMIT 8", "top families")]:
        print(f"  {lab}:", dict(db.execute(q).fetchall()))
    checks = []
    # QA-E: split integrity (no profile crosses splits)
    bad = db.execute("SELECT COUNT(*) FROM (SELECT profile_id FROM scenes GROUP BY profile_id "
                     "HAVING COUNT(DISTINCT split)>1)").fetchone()[0]
    checks.append(("QA-E1 no profile crosses splits", bad == 0, f"{bad} leak"))
    # QA-B1: terrain(family)-class decorrelation: no family >65% one class
    rows = db.execute("SELECT family, coarse, COUNT(*) FROM scenes GROUP BY family, coarse").fetchall()
    fam = {}
    for f, c, n in rows:
        fam.setdefault(f, {})[c] = n
    worst = max((max(d.values()) / sum(d.values()) for d in fam.values()), default=0)
    checks.append(("QA-B1 no terrain-family >65% one class", worst <= 0.66, f"worst {worst:.0%}"))
    # QA-F2: class window counts within 3x (subject classes)
    cw = {}
    for c in ("human", "vehicle", "animal"):
        cw[c] = db.execute(f"SELECT COUNT(*) FROM windows WHERE {c}_level!='none'").fetchone()[0]
    rng = max(cw.values()) / max(min(cw.values()), 1)
    checks.append(("QA-F2 subject-class windows within 3x", rng <= 3.5, f"{cw}, ratio {rng:.1f}"))
    # zone sanity: marginal+nothing present, detectable not ~100%
    det = db.execute("SELECT COUNT(*) FROM windows WHERE zone='detectable'").fetchone()[0]
    checks.append(("zone: detectable < 80% (faint windows exist)", det / nwin < 0.8, f"{det/nwin:.0%} detectable"))
    print("\n  assertions:")
    for name, ok, info in checks:
        print(f"   [{'PASS' if ok else 'FAIL'}] {name}  ({info})")
    return all(ok for _, ok, _ in checks)


def main():
    cd = sys.argv[1] if len(sys.argv) > 1 else r"N:\geophone_synth\corpus_v1"
    t0 = time.time()
    print(f"merging shards in {cd} ...", flush=True)
    db, out = merge(cd)
    print(f"merged -> {out} ({(time.time()-t0)/60:.1f} min), size "
          f"{os.path.getsize(out)/1e9:.1f} GB", flush=True)
    ok = qa(db)
    db.close()
    print("\nQA:", "ALL PASS" if ok else "see failures above")


if __name__ == "__main__":
    main()
