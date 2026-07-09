# 30 — Live geophone acquisition → streaming 3-second windows

**Scope of this doc:** the front of the pipeline only — get bytes off the live
geophone over USB serial, turn them into a rolling 3-second / 3000-sample window
in the exact units the model expects, and hand each window to feature extraction.
It stops at "window ready for `simgeo/features.py`"; featurization, the SNN, and
the WebUI wiring are downstream docs.

**Status:** PLANNING ONLY. No code here is to be written/modified yet. All file
paths are absolute from the project root
`S:\ALL PROJECTS\geophone sensor\finals project\finals project\`.

---

## 1. Acquisition facts (verified against firmware + existing scripts)

All current acquisition scripts agree on the same wire contract. Source of truth:

| Fact | Value | Cite |
|---|---|---|
| USB-serial chip | CH340 (WCH) | firmware uses an ESP8266 + CH340 USB bridge |
| VID / PID | `0x1A86` / `0x7523` | `Detect.py:12-13`; `Main.py:12-13`; `record15.py:8`; `capture_test.py:10`; `diag_geophone.py:5`; `gain_check.py:7` |
| Auto-detect rule | match VID&PID **or** `'CH340' in description` | `Detect.py:21-22`; `Main.py:19-20` |
| Baud | `460800` | firmware `Geophone_reader.ino:7` (`SERIAL_BAUD = 460800`); `Detect.py:14`; `Main.py:10`; `record15.py:5` |
| Open flags | `dtr = False`, `rts = False` **before** `open()` | `Detect.py:45-47`; `Main.py:38-40`; `record15.py:15`. Purpose: do NOT toggle EN/GPIO0 → do NOT reset the ESP on port open (comment `Main.py:38` "no-reset open keeps ESP running"). |
| Settle after open | `time.sleep(2)` then `reset_input_buffer()` | `Detect.py:48-49`; `Main.py:46-47`; `record15.py:16`. Drops the boot banner + partial first line. |
| Banner line(s) | `"Geophone - Direct Differential ..."` then header `"time_ms,voltage_mV"` | firmware `Geophone_reader.ino:22`, `:42` |
| Data line format | `"<time_ms>,<voltage_mV>"` ASCII, `\r\n` terminated | firmware `:53-59` (`Serial.print(millis()); print(","); println(raw*0.125, 3)`) |
| Parse rule | split on `,`, require exactly 2 fields, `int(ms)`, `float(mV)`; skip otherwise | `Main.py:53-57`; `Detect.py:55-61`; `record15.py:22-27` |
| Sample rate | ~1000 Hz (firmware `SAMPLE_INTERVAL_US = 1000`, `Geophone_reader.ino:10`) | confirmed live: `Detect.py:69` computes `1000/median(dt)`; newest CSVs have dt≈1 ms |
| ADC | ADS1015, `GAIN_SIXTEEN` (±0.256 V), **0.125 mV/bit** | firmware `:4`, `:8` (`GAIN_SIXTEEN`), `:58` (`raw * 0.125`) |
| Chip data rate | `RATE_ADS1015_3300SPS` (3300 SPS, decimated to ~1 kHz by the 1 ms loop) | firmware `:32` |

**`time_ms` semantics:** it is the Arduino `millis()` value, **not** wall clock.
`Main.py:58-60` and `record15.py:28` zero it on the first sample (`start_ms`) and
divide by 1000 to make a relative `time_s`. The reader must do the same and must
treat `time_ms` as authoritative for ordering/jitter, not the host receive time.

**Note — older vs newer CSVs differ in rate.** `geophone_20260524_*.csv` were
logged at ~100 Hz (dt = 0.010 s in the file). `geophone_20260615_*.csv` are
~1000 Hz (dt = 0.001 s). The current firmware is the 1 kHz one; the rolling reader
must be built for **1000 Hz** and should *measure* the live rate (median dt) rather
than assume it (see §5 jitter).

---

## 2. Units & scaling — the EXACT chain for one live sample

This is the part most likely to introduce a silent bug, so it is spelled out
end to end.

```
ADS1015 raw counts  --(firmware ×0.125)-->  voltage_mV on the wire   [Geophone_reader.ino:58]
voltage_mV          --(Main.py ÷1000)----->  amplitude in VOLTS in CSV [Main.py:61-62]
amplitude (volts)   --(model ×25.4)-------->  pseudo-mV fed to features [snn_real_baseline.py:38, finetune_5fold.py:39]
```

Key facts:

- **Firmware emits millivolts.** `raw * 0.125` with `GAIN_SIXTEEN` ⇒ 1 LSB =
  0.125 mV. So the smallest non-zero magnitude on the wire is `0.125` mV. (In the
  CSV that shows up as `amplitude = ±0.000125` V — visible literally in
  `geophone_20260615_*.csv` rows. That single-LSB value is exactly the quiet-floor
  signature; see §4.)
- **CSV is in volts.** `Main.py:61` `v_V = raw_mv / 1000.0`, written `%.6f`
  (`Main.py:62`). `record15.py:29` and `capture_test.py:32` do the identical
  `/1000` (the latter comments "amplitude in volts (Main.py convention)").
  CSV header is `time_s,amplitude` (`Main.py:44`, `record15.py:32`).
- **The model multiplies the CSV `amplitude` (volts) by `SCALE = 25.4`.** This is
  the *only* amplitude transform between CSV and features:
  - `snn_real_baseline.py:13` `SCALE = 25.4`; `:38`
    `a = pd.read_csv(p)["amplitude"].to_numpy(np.float32) * SCALE`
  - `finetune_5fold.py:12` `SCALE = 25.4`; `:39` identical line.
  - The product `volts × 25.4` is then fed to `F.scene_precompute` /
    `F.window_features` (`snn_real_baseline.py:41-43`). The features module treats
    its input as **millivolts** (its synthetic corpus is in mV, e.g. `noise_mv`,
    `clean_mv` in `test_floor.py:17`). So `×25.4` is a *units+gain bridge*: it
    rescales real-sensor volts up to sit on the synthetic mV noise floor the model
    was pretrained on. It is **not** a clean volts→mV conversion (that would be
    ×1000); 25.4 is an empirically chosen alignment constant (see §5 open question).

**Therefore the live reader's required output unit is: the SAME number that the
CSV `amplitude` column holds — i.e. VOLTS — so the existing `×25.4` step applies
unchanged.** Concretely, for each parsed wire sample:

```
voltage_mV          (parsed float from the line, e.g. -0.125)
amplitude_volts  =  voltage_mV / 1000.0        # mirror Main.py:61 EXACTLY
```

Accumulate `amplitude_volts` into the window buffer. When a 3000-sample window is
emitted, multiply the whole window by `25.4` (mirror `snn_real_baseline.py:38`)
*then* pass to `F.scene_precompute`/`F.window_features`. Do the `×25.4` at the
window boundary, not per sample, so the buffer stays in the same volts unit as the
CSVs (keeps replay-from-CSV and live identical — see §4).

**Do NOT** apply `×1000` and `×25.4` both. **Do NOT** feed raw mV to the model.
The one and only correct live featurization input is `(voltage_mV / 1000) × 25.4`
per sample, batched into 3000-sample windows.

---

## 3. Rolling-buffer streaming reader — spec (signatures only)

**There is no existing rolling/streaming reader.** Every current acquisition
script is a *fixed-duration one-shot*: `Detect.py` reads 5 s, `record15.py` reads
`DUR` (default 15 s) then exits, `capture_test.py`/`diag_geophone.py`/`test_floor.py`
read a fixed window then save+plot, `Main.py` streams to CSV but only prints rows —
none maintains a rolling deque or emits overlapping model windows. This section is
**new groundwork**, not a refactor of an existing component.

### 3.1 Window math (from `simgeo/features.py`)

- `F.FS = 1000.0`, `F.NW = 3000` (window = 3.0 s / 3000 samples) —
  `simgeo/features.py:30-31`, docstring `:22`.
- Buffer: `deque(maxlen=3000)` = exactly one feature window of history.
- Hop: **1500 samples = 1.5 s** ⇒ 50% overlap. This matches the model's training
  `HOP = 1500` (`snn_real_baseline.py:13`, `finetune_5fold.py:12`). Emit a window
  every 1500 newly-appended samples.
- One window's feature path (mirror of `snn_real_baseline.py:41-43`):
  `pre = F.scene_precompute(window_x); feats = F.window_features(pre, 0)` where
  `window_x` is the full 3000-sample array `× 25.4`. (Live uses a single window so
  `i0 = 0`; the per-scene loop in the training scripts is only for chopping long
  CSVs.) `F.NFEAT == 132` (`simgeo/features.py:144-145`); the model then selects
  its trained subset via `fidx` exactly as in the training scripts.

### 3.2 Proposed signatures (no implementation)

```python
# new module, suggested path: <root>/live_reader.py  (sibling of Main.py)

CH340_VID = 0x1A86
CH340_PID = 0x7523
BAUD      = 460800
FS        = 1000          # nominal; measured at runtime
NW        = 3000          # = simgeo.features.NW  (3 s)
HOP       = 1500          # 1.5 s, 50% overlap (= training HOP)
SCALE     = 25.4          # = snn_real_baseline.SCALE / finetune_5fold.SCALE
SETTLE_S  = 2.0           # mirror Main.py:46

def find_geophone() -> str | None:
    """COM port of the CH340 geophone, or None. Same rule as Main.py:15-26 /
    Detect.py:17-24 (VID&PID or 'CH340' in description)."""

class GeophoneStream:
    """Background serial reader → rolling 3 s windows. NOT in any current script."""

    def __init__(self, port: str | None = None, *,
                 on_window=None,         # callback(window_volts: np.ndarray[3000])
                 baud: int = BAUD): ...

    def open(self) -> None:
        """serial.Serial(); port/baud/timeout=1; dtr=False; rts=False BEFORE open();
        open(); sleep(SETTLE_S); reset_input_buffer().  (Detect.py:41-49 pattern.)"""

    def start(self) -> None:
        """Spawn the daemon reader thread (see _reader_loop). Idempotent."""

    def stop(self) -> None:
        """Signal thread to exit, join with timeout, close the port."""

    def _reader_loop(self) -> None:
        """ser.readline().decode('utf-8','replace').strip(); skip if ',' not in line
        or not exactly 2 fields; int(ms), float(mV); on ValueError continue
        (Detect.py:54-61 parse). Append mV/1000.0 (VOLTS, Main.py:61) to the deque.
        Every HOP appended samples, if len(buf)==NW: snapshot list(buf) → np.array,
        invoke _emit(window). Track first ms as start_ms for relative time & jitter."""

    def _emit(self, window_volts) -> None:
        """x = np.asarray(window_volts, np.float64) * SCALE     # snn_real_baseline.py:38
           pre = F.scene_precompute(x); feats = F.window_features(pre, 0)  # NFEAT=132
           hand feats to the downstream model adapter / WebUI callback."""

    def latest_window(self) -> "np.ndarray | None":
        """Thread-safe snapshot of the current 3000-sample buffer (for the scope /
        signal-strength meter), or None until the buffer first fills."""
```

### 3.3 Downstream handoff (what the window becomes)

The WebUI's simulated backend already defines the per-window output contract the
live path must reproduce — `Sim.sampleWindow()` returns
`{ win, probs:{human,car,animal,nothing}, detected, snr, kind, r }`
(`webui/src/sim.js:94-123`). The live chain
`GeophoneStream._emit → features (132) → model_ema.pt → probs dict` must emit the
**same shape** so the existing UI (`webui/src/histograms.js`, `detection.js`)
consumes it unchanged. (Model inference + the Python↔JS bridge are separate
wireup docs; this doc guarantees only that a correctly-scaled 132-feature window
is produced at the right cadence.)

### 3.4 Cadence / throughput (concrete numbers, N = 3000)

- Append rate ≈ 1000 samples/s → one window emitted every **1.5 s** (every 1500
  samples). ~40 windows/minute.
- Per-window compute: `scene_precompute` runs ~15 SOS band-filters + 2 envelopes +
  peak-picks over N=3000 (O(N) each, ≈ a few ms), then `window_features` is O(N log N)
  dominated by the rfft/welch/wavelet-packet work on 3000 samples — comfortably
  < 1.5 s on CPU, so the reader thread never falls behind the 1.5 s hop. Memory is
  O(NW) = 3000 floats for the buffer (negligible).

---

## 4. Live-quality caveat + recorded-CSV replay fallback

**Use the live device — it is connected — but expect a weak/near-floor signal on a
quiet indoor floor.** The Jun-15 diagnostics establish this:

- `gain_check.py:28-31`: latest live std ≈ sub-mV, only a handful of distinct ADC
  codes, `std/LSB` small — the trace barely exceeds one quantization step.
- `notch_analysis.py:14-23`: residual (non-mains) RMS is only a few × the **ideal
  ADC quantization floor** (`q_rms = LSB/√12`, `:15`), and 50/100/150 Hz mains
  dominate the raw power. I.e. on a quiet floor the live signal is
  **floor-noise / quantization limited**, not source-limited.
- `test_floor.py` / `test_floor_zoom.py`: the only consistent real "feature" is a
  ~7–8 Hz floor drum (`test_floor.py:42`, axvspan 6–9 Hz) — ambient, not a target.
- Confirmed by the raw CSVs: long runs of literal `-0.000125` (= −1 LSB) in
  `geophone_20260615_*.csv` ⇒ the sensor is sitting on the quantization floor when
  nothing is walking/driving near it.

**Consequence for the demo:** the live path will run and stream windows, but
**detection may be weak or empty unless there is a real vibration source (footsteps
/ a car) close to the sensor.** This is a physics/siting issue, not a code bug —
flag it to the user so a near-floor live demo isn't mistaken for a broken model.

**Replay fallback (recommended default for a reliable demo):** the reader should
support a **recorded-CSV source** that produces byte-identical windows to the live
path. Real labeled content with actual car/human energy lives in
`Goephone-Project/geophone_data/` — `car.csv`, `human.csv`, `car_nothing.csv`,
`human_nothing.csv` (these are exactly the files the model was validated on:
`snn_real_baseline.py:36`, `finetune_5fold.py:37`). They share the **same schema**
(`time_s,amplitude`, amplitude in volts) as the live CSVs, so:

- A `CsvReplaySource(path)` can stream `amplitude` rows into the *same* deque at a
  paced ~1000 Hz (or fast-forward), through the *same* `×25.4` `_emit` path.
- Because replay and live share the volts unit and the same scaling, **switching
  source must not change any downstream math** — this is the test that the unit
  handling in §2 is correct.

This also matches the WebUI's intended "Geophone mode = replay a recorded CSV (live
serial optional)" (`webui/README.md:68`).

---

## 5. Threading / async design, error handling, open questions

### 5.1 Threading model
- One **daemon background thread** owns the blocking `ser.readline()` loop and the
  deque; the main/UI thread never touches the serial port. `serial.Serial.timeout
  = 1` (as in every current script) keeps `readline()` from blocking forever so the
  stop flag is checked at least once/second.
- Hand-off to the UI is via the `on_window` callback (or a thread-safe queue). The
  callback runs feature extraction; if that ever approaches the 1.5 s budget, move
  featurization to a second worker and let the reader thread only fill the deque.
- `latest_window()` returns a copy under a lock so the scope renderer can read the
  buffer without tearing.

### 5.2 Error handling — disconnect / reconnect
- **Open failure / not found:** `find_geophone()` returns None → surface "geophone
  not connected" and fall back to CSV replay (§4) rather than crashing. (Current
  scripts just `sys.exit`, e.g. `record15.py:12`, `diag_geophone.py:9` — not
  acceptable for a long-running UI.)
- **Mid-stream USB unplug:** `readline()` raises `serial.SerialException` /
  `OSError`. Catch in `_reader_loop`, mark `connected = False`, close the handle,
  and enter a reconnect loop: periodically re-run `find_geophone()`; on success
  re-`open()` (with the same `dtr=False/rts=False` + 2 s settle), `reset_input_buffer()`,
  and **clear the deque** (stale pre-disconnect samples must not bridge into a new
  window). Push a status event to the UI on each transition.
- **Garbage / partial lines:** already handled by the parse guards (skip if `','`
  absent, not 2 fields, or `ValueError`) — keep them; these are why `probe_serial.py`
  / `probe2.py` exist (they were one-off baud/format sweeps; `probe_serial.py:2`
  hardcodes `COM4`, which is *not* reliable — the auto-detect path must always be
  used in production, never a hardcoded port).
- **First-window latency:** the buffer needs 3000 samples (~3 s) before the first
  window; show a "warming up" state for the first ~3 s after (re)connect.

### 5.3 Open questions / risks (flag to user)
1. **Is `×25.4` calibrated to *this* sensor on *this* floor?** SCALE = 25.4 was
   fixed against the `Goephone-Project` recordings (`snn_real_baseline.py:13`). If
   the live sensor's analog gain / coupling differs from those recordings, 25.4 may
   put live windows at the wrong point on the synthetic noise floor and bias every
   probability. **Recommend:** record a short live "nothing" clip and a live
   footstep clip, compare their RMS (in volts ×25.4) to the synthetic mV floor used
   in training before trusting live probabilities. This is a publication-relevant
   calibration step, not a cosmetic one.
2. **Sample-rate jitter.** Firmware targets 1 ms but it is a busy-wait `micros()`
   loop sharing the CPU with I2C + `Serial.print` (`Geophone_reader.ino:46-60`);
   `diag_geophone.py:28-33` already measures median dt and counts gaps ≥50 ms. The
   feature bank assumes an exact 1000 Hz grid (`F.FS = 1000`). **Risk:** if the live
   median rate drifts (e.g. 960–1010 Hz) the 3000-sample window is not exactly 3.000 s
   and band edges shift slightly. **Recommend:** measure live median dt at startup;
   if it deviates from 1.0 ms beyond a tolerance, warn (and optionally resample to a
   uniform 1 kHz grid before windowing — not in current scripts).
3. **Dropped samples vs. wall-clock.** The reader counts *received* samples for the
   1500-hop, not `time_ms`. If lines are dropped (USB hiccup, host stall), 3000
   received samples may span >3 s. Decide whether to gate windows on `time_ms` span
   (≈3000 ms) instead of raw count for rigor.
4. **Mains contamination.** 50/100/150 Hz dominate the quiet floor
   (`notch_analysis.py`). The training features keep site lines as *monitored*
   features (`simgeo/features.py:64-72`), so do **not** silently notch the live
   signal before featurization — that would diverge live from the trained
   distribution. Any notch must be applied to the training data too, or not at all.
5. **Multiple CH340 devices.** Current scripts just take `candidates[0]`
   (`Main.py:24-26`, `Detect.py:37-38`). If another CH340 dongle is present this is
   ambiguous — consider a serial-number pin for the demo machine.

---

## Summary of decisions to confirm with the user
- Reader output unit = **volts** (mirror `Main.py:61`); apply **×25.4 at the window
  boundary** only.
- Default the demo to **CSV replay** of `Goephone-Project/geophone_data/*.csv` for
  reliable detections; use the live device as a (likely near-floor) live mode.
- Build for **1000 Hz / NW=3000 / HOP=1500**, but **measure** the live rate and warn
  on drift.
- Validate the **25.4** constant against a fresh live nothing/footstep pair before
  trusting live probabilities.
