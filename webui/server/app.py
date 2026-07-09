"""GeoSense backend — serves the web UI and streams geophone detections.

Run:  python -m uvicorn app:app --host 127.0.0.1 --port 8000   (from webui/server/)
or:   python webui/server/run.py

WebSocket /ws : client sends {source:"replay"|"live", csv?, speed?}; server streams
one `window` message per ~1.5 s (model cadence) with per-class probabilities, the
detected classes, a signal-strength readout, and a downsampled trace for the scope.
"""
import asyncio, math, os, sys
import numpy as np
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

HERE = os.path.dirname(os.path.abspath(__file__))
WEBUI = os.path.abspath(os.path.join(HERE, ".."))
sys.path.insert(0, HERE)
import inference, geophone, livegen_source

SCALE = 25.4                                    # real volts -> synthetic mV alignment (geophone)
SIM_SCALE = 1.0                                 # gen_scene out_mv is ALREADY synthetic mV
DATA = os.path.join(HERE, "data")               # vendored replay clips (self-contained)
det = inference.Detector()
app = FastAPI(title="GeoSense")


def downsample(w, n=160):
    idx = np.linspace(0, len(w) - 1, min(n, len(w))).astype(int)
    return [round(float(w[i]) * 1000.0, 3) for i in idx]      # volts -> mV


def strength(w):
    rms_mv = float(np.std(w)) * 1000.0
    db = 20 * math.log10(max(rms_mv, 1e-3) / 0.5)
    pct = int(max(0, min(100, round((db + 3) / 0.30))))
    return {"rms_mv": round(rms_mv, 3), "pct": pct}


# Dataset (gen_scene) mode is already in mV — no x1000 / x25.4
def strength_mv(w):
    rms_mv = float(np.std(w))
    db = 20 * math.log10(max(rms_mv, 1e-3) / 0.5)
    return {"rms_mv": round(rms_mv, 3), "pct": int(max(0, min(100, round((db + 3) / 0.30))))}


def downsample_mv(w, n=160):
    idx = np.linspace(0, len(w) - 1, min(n, len(w))).astype(int)
    return [round(float(w[i]), 3) for i in idx]


def make_source(cfg):
    if cfg.get("source") == "sim":
        return livegen_source.LiveGenSource(coarse=cfg.get("class"),
                                            family=cfg.get("family"), subkind=cfg.get("subkind"))
    if cfg.get("source") == "live":
        try:
            return geophone.SerialSource(port=cfg.get("port"))
        except Exception as e:
            print(f"[geophone] live unavailable ({e}); falling back to replay")
    csv = cfg.get("csv") or "car.csv"
    return geophone.ReplaySource(os.path.join(DATA, csv))


@app.get("/devices")
def devices():
    return {"ports": geophone.list_serial_ports(), "geophone": geophone.find_geophone_port(),
            "replays": sorted(os.path.basename(p) for p in
                              __import__("glob").glob(os.path.join(DATA, "*.csv")))}


@app.get("/sim")
def sim_status():
    return {"available": livegen_source.available()}


@app.websocket("/ws")
async def stream(ws: WebSocket):
    await ws.accept()
    src = None
    try:
        cfg = await ws.receive_json()
        is_sim = cfg.get("source") == "sim"
        src = make_source(cfg)
        speed = float(cfg.get("speed", 1.0))
        scale = SIM_SCALE if is_sim else SCALE
        sig_fn = strength_mv if is_sim else strength
        trace_fn = downsample_mv if is_sim else downsample
        mode = "dataset" if is_sim else "geophone"
        await ws.send_json({"type": "ready", "source": getattr(src, "name", "?"),
                            "live": cfg.get("source") == "live", "mode": mode})
        win = 0
        last_sid = None
        while True:
            w = src.read_window()
            if w is None:                                   # buffer not full yet (live)
                if not is_sim:
                    await ws.send_json({"type": "buffering"})
                await asyncio.sleep(0.2); continue
            if is_sim:                                      # new scene -> reset canvas + describe
                meta = src.scene_meta()
                if meta.get("sid") != last_sid:
                    last_sid = meta.get("sid")
                    await ws.send_json({"type": "scene", "meta": meta, "n_windows": len(src.wins)})
            res = await asyncio.to_thread(det.run_waveform, w, scale)
            r = res[0] if res else {"probs": {"human": 0, "car": 0, "animal": 0, "nothing": 1},
                                    "detected": []}
            signal = sig_fn(w)
            if is_sim and src.last_truth:               # meter from the true per-window SNR
                snr = src.last_truth.get("common_snr", -99.0)
                signal["pct"] = int(max(0, min(100, round((snr + 5) / 35 * 100))))
            msg = {"type": "window", "win": win, "mode": mode, "source": getattr(src, "name", "?"),
                   "probs": r["probs"], "detected": r["detected"], "levels": r.get("levels", {}),
                   "signal": signal, "trace": trace_fn(w)}
            if is_sim:
                msg["truth"] = src.last_truth
                msg["meta"] = src.scene_meta()
            await ws.send_json(msg)
            win += 1
            await asyncio.sleep(max(0.1, 1.5 / speed))
    except WebSocketDisconnect:
        pass
    finally:
        if src:
            src.close()


# serve ONLY the front-end (index.html + the asset dirs) — not server/, node_modules, etc.
@app.get("/")
def index():
    return FileResponse(os.path.join(WEBUI, "index.html"))


for _d in ("src", "styles", "assets"):
    app.mount(f"/{_d}", StaticFiles(directory=os.path.join(WEBUI, _d)), name=_d)
