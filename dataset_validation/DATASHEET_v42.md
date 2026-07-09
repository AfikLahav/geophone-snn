# Corpus datasheet — v42

windows: 3,292,768

## Declared design
- **seed**: 20260706
- **labels**: 3 s / 1.5 s hop; presence=emission-activity (D3); per-class in-band SNR; zone3 at gates_3s
- **subkind_weights**: {"human": {"walk": 2}, "animal": {"slow_quad": 2}, "nothing_confuser_w": 1.8, "note": "declared tempering (Change 4); realized B measured in INVARIANT_CHECK"}
- **weather**: Change 1: CLASS-BLIND global tier axis. Marginal {calm .35, wind_low .20, wind_mid .15, wind_high .10, rain_light .12, rain_heavy .08}, dealt per-cell by largest-remainder [R3] -> identical across all classes (kills the v4 P/N-condition weather leak).
- **subkind_snr_offset_db**: Change 2 [R1]: per-subkind source-level offset (car/walk/horse ref) applied at the snr_to_distance CALL SITE keyed on the real subkind -> fills starved loud bins.
- **W_snr_density**: triangular(tau_lo, mode +1.5/+7.5/+4.5, hi +40/45) + 15% subfloor tail (unchanged from v4)
- **Q_rig**: {"gain_log10": "GEO_GAIN_LOG10 span (uniform-in-dB; bounds set by the amplitude pilot)", "coupling": "Change 3: MIX lowpass/bump 50/50, anchor_fc median 53 Hz (lognormal 0.35, clip 30-150)", "quantization": "GEO_QUANT_MIX categorical LSB {0=continuous, 0.125, 0.2, 0.25} mV (model input only)", "rail_mv": "GEO_RAIL_MIX categorical {inf, 512, 256} mV (model input only; clipping = randomized nuisance)", "mains": "v41 dB-above-floor, terrain-gated", "spur_150hz_p": 0.2}
- **documented_couplings**: ["coupling_fc ~ profile stiffness (physical)", "vehicle excluded on vs<95 m/s (physical scope)"]
- **floors_caps**: F floors 300/reachable cell; per-subkind ceilings from subkind_ceilings_v42.json (Change 2)

## Realized class mass (pi_train)
{"nothing": 0.5552, "animal": 0.1316, "vehicle": 0.1173, "human": 0.1008, "mixed": 0.0951}

## Realized subkind mass within class
{
 "human": {
  "walk": 0.2313,
  "stealth": 0.1833,
  "march": 0.1248,
  "loaded": 0.1206,
  "group": 0.1205,
  "child": 0.1115,
  "run": 0.108
 },
 "vehicle": {
  "car": 0.1765,
  "idle_exit": 0.1384,
  "convoy": 0.1202,
  "tractor": 0.1068,
  "tracked": 0.1007,
  "truck": 0.0935,
  "two_vehicle": 0.0917,
  "motorbike": 0.0866,
  "bicycle": 0.0855
 },
 "animal": {
  "slow_quad": 0.2571,
  "horse_rider": 0.1116,
  "horse": 0.1109,
  "boar": 0.1083,
  "herd": 0.1067,
  "sheep": 0.1065,
  "dog": 0.1016,
  "jackal": 0.0973
 },
 "nothing": {
  "machinery": 0.281,
  "traffic": 0.2809,
  "overflight": 0.2806,
  "ambient": 0.1574
 },
 "mixed": {
  "vehicle+animal": 0.3478,
  "human+animal": 0.338,
  "human+vehicle": 0.3142
 }
}

## Realized condition | class (neutrality view)
{
 "human": {
  "calm": 0.36,
  "wind_low": 0.199,
  "wind_mid": 0.144,
  "rain_light": 0.121,
  "wind_high": 0.094,
  "rain_heavy": 0.081
 },
 "vehicle": {
  "calm": 0.355,
  "wind_low": 0.199,
  "wind_mid": 0.147,
  "rain_light": 0.12,
  "wind_high": 0.098,
  "rain_heavy": 0.08
 },
 "animal": {
  "calm": 0.362,
  "wind_low": 0.2,
  "wind_mid": 0.14,
  "rain_light": 0.122,
  "wind_high": 0.094,
  "rain_heavy": 0.082
 },
 "nothing": {
  "calm": 0.35,
  "wind_low": 0.2,
  "wind_mid": 0.15,
  "rain_light": 0.12,
  "wind_high": 0.1,
  "rain_heavy": 0.08
 },
 "mixed": {
  "calm": 0.364,
  "wind_low": 0.198,
  "wind_mid": 0.14,
  "rain_light": 0.123,
  "wind_high": 0.093,
  "rain_heavy": 0.082
 }
}

## Realized Q_j sensor-DR spans (v4.2 Change 3)
{
 "tier": {
  "calm": 0.3547,
  "wind_low": 0.1996,
  "wind_mid": 0.1467,
  "rain_light": 0.1206,
  "wind_high": 0.0978,
  "rain_heavy": 0.0805
 },
 "gain_log10": {
  "min": 0.3,
  "max": 2.9,
  "median": 1.6,
  "span_db": 52.0
 },
 "quant_lsb": {
  "0.25": 0.2519,
  "0.125": 0.2506,
  "0.2": 0.2493,
  "0.0": 0.2482
 },
 "rail_mv": {
  "256.0": 0.3345,
  "512.0": 0.3333,
  "inf": 0.3322
 },
 "coupling_form": {
  "bump": 0.5011,
  "lowpass": 0.4989
 }
}