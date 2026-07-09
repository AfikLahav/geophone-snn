"""Launch GeoSense: starts the backend (which also serves the UI) and opens a browser.

    python run.py            # http://localhost:8000/
"""
import os, sys, time, threading, webbrowser

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "server"))
PORT = int(os.environ.get("PORT", "8000"))


def _open_browser():
    time.sleep(2.0)
    webbrowser.open(f"http://localhost:{PORT}/")


if __name__ == "__main__":
    import uvicorn
    import app  # webui/server/app.py
    print(f"\n  GeoSense  ->  http://localhost:{PORT}/   (Ctrl+C to stop)\n")
    threading.Thread(target=_open_browser, daemon=True).start()
    uvicorn.run(app.app, host="127.0.0.1", port=PORT, log_level="warning")
