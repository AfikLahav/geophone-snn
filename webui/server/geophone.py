"""Geophone signal source.

Yields 3 s windows (NW=3000) at a 1.5 s hop (HOP=1500), in VOLTS (the recorded-CSV
/ Main.py convention). The backend applies x25.4 + features downstream.

  ReplaySource - replays a recorded CSV (demo default; the live floor is usually
                 quantization noise, so replay shows real detections).
  SerialSource - live CH340 device, background reader into a rolling deque.

Device detection is robust to a different machine: explicit port override first,
then CH340 VID/PID, then USB-serial description, then a content probe, then a
single-port fallback.
"""
import os, time, glob, threading, collections
import numpy as np

FS = 1000
NW = 3000
HOP = 1500
CH340_VID, CH340_PID = 0x1A86, 0x7523
BAUD = 460800


class ReplaySource:
    def __init__(self, csv_path, loop=True):
        import pandas as pd
        self.amp = pd.read_csv(csv_path)["amplitude"].to_numpy(np.float32)
        self.name = os.path.basename(csv_path)
        self.loop = loop
        self.pos = 0

    def read_window(self):
        """Next 3 s window (volts), advancing by HOP. None at end (if not looping)."""
        if self.pos + NW > len(self.amp):
            if not self.loop:
                return None
            self.pos = 0
        w = self.amp[self.pos:self.pos + NW].copy()
        self.pos += HOP
        return w

    def close(self):
        pass


def list_serial_ports():
    from serial.tools import list_ports
    return [{"device": p.device, "desc": p.description, "vid": p.vid, "pid": p.pid}
            for p in list_ports.comports()]


def find_geophone_port(prefer=None):
    """Return the device path of the geophone, or None."""
    if prefer:
        return prefer
    from serial.tools import list_ports
    ports = list(list_ports.comports())
    for p in ports:                                   # 1) exact CH340 VID/PID
        if (p.vid, p.pid) == (CH340_VID, CH340_PID):
            return p.device
    for p in ports:                                   # 2) description match (CH340/CH910x/USB-serial)
        d = ((p.description or "") + " " + (p.manufacturer or "")).upper()
        if any(k in d for k in ("CH340", "CH910", "USB-SERIAL", "USB SERIAL")):
            return p.device
    import serial                                     # 3) content probe: does it emit "int,float"?
    for p in ports:
        try:
            s = serial.Serial(); s.port = p.device; s.baudrate = BAUD; s.timeout = 1
            s.dtr = False; s.rts = False; s.open(); time.sleep(0.3); s.reset_input_buffer()
            ok = 0
            for _ in range(40):
                a = s.readline().decode("utf-8", "replace").strip().split(",")
                if len(a) == 2:
                    try: int(a[0]); float(a[1]); ok += 1
                    except ValueError: pass
            s.close()
            if ok >= 5:
                return p.device
        except Exception:
            pass
    if len(ports) == 1:                               # 4) only one port -> try it
        return ports[0].device
    return None


class SerialSource:
    def __init__(self, port=None, baud=BAUD):
        import serial
        self.port = find_geophone_port(port)
        if not self.port:
            raise RuntimeError("no geophone serial device found")
        self.ser = serial.Serial()
        self.ser.port = self.port; self.ser.baudrate = baud; self.ser.timeout = 1
        self.ser.dtr = False; self.ser.rts = False          # don't reset the ESP on open
        self.ser.open(); time.sleep(2); self.ser.reset_input_buffer()
        self.name = self.port
        self.buf = collections.deque(maxlen=NW)
        self.lock = threading.Lock()
        self._run = True
        self.thread = threading.Thread(target=self._reader, daemon=True)
        self.thread.start()

    def _reader(self):
        while self._run:
            try:
                a = self.ser.readline().decode("utf-8", "replace").strip().split(",")
                if len(a) == 2:
                    mv = float(a[1])                          # firmware reports mV
                    with self.lock:
                        self.buf.append(mv / 1000.0)         # -> volts (Main.py convention)
            except Exception:
                pass

    def read_window(self):
        with self.lock:
            if len(self.buf) < NW:
                return None
            return np.array(self.buf, np.float32)

    def close(self):
        self._run = False
        try: self.ser.close()
        except Exception: pass


if __name__ == "__main__":
    csv = os.path.join(os.path.dirname(__file__), "data", "human.csv")
    rep = ReplaySource(csv)
    w = rep.read_window()
    print(f"ReplaySource({rep.name}): window shape={None if w is None else w.shape}, "
          f"rms={None if w is None else round(float(np.std(w)), 5)} V")
    print("serial ports:", list_serial_ports())
    print("detected geophone port:", find_geophone_port())
