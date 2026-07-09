# Scene Hyperparameters (v0.2 — companion to SCENE_CATALOG; BINDING for generation)

> **AUTHORITY: `GENERATION_PLAN.md` v1.1 supersedes this file wherever they conflict.**
> v0.2 syncs the critical corrected values (2026-06-12 review); see GENERATION_PLAN Appendix S
> for the full list. Where a value here is marked **[v1.1]** it has been synced.

Distribution notation: U=uniform, LU=log-uniform, N=normal(μ,σ), LN=log-normal, Exp(mean),
Poi=Poisson. All clips explicit. Durations include lead-in/out.

## A. Per-family parameter envelopes

### A1. Single human transit (H01–H03, H13–H17, H20)
- duration: from residence-time; LU spread, typically 15–180 s; **cap 240 s** (H03/H13 tails)
- walk speed N(1.4, 0.25) clip [0.8, 2.2] m/s; run speed N(3.2, 0.7) clip [2.0, 5.5]
- cadence = speed / stride_len, stride_len N(0.72, 0.08) m; **clip walk [1.3, 2.2] [v1.1]**, run [2.6, 3.5] Hz
- **run strike pattern: ~70% rearfoot (double-hump impact 1.65+active 2.44 BW) / 30% fore-mid (single 2.0–3.1 BW); contact clip [0.16, 0.30] s [v1.1]**
- mass LN(ln 75, 0.22) clip [45, 110] kg; child LN(ln 28, 0.2) clip [18, 42]
- footwear categorical: boot .35 / trainer .35 / sandal .15 / barefoot .10 / heavy-boot .05
  → **heel-strike rise lookup [v1.1]: boot 5–15 ms, trainer 10–25, barefoot/cushioned 20–50, sandal 10–25 + slap** (sub-ms unsupported)
- closest approach LU(2, R_det) m; pauses (H13): Poi(0.05/s), dur Exp(12) clip [5, 30] s
- load (H14): U(20, 40) kg → speed −15–30%, cadence −10%
- limp (H15): L/R amp U(0.5, 0.85), interval U(0.7, 0.95)

### A2. Stealth/covert (H04, H05, H18, H19)
- duration LU(20, 300); H19 Exp(90) **cap 180 s**
- H04 speed U(0.2, 0.6) m/s, stride 0.35–0.45 m → **cadence [1.0, 1.5] [v1.1]**; **GRF ×U(0.5, 0.7) [v1.1, LOW-CONFIDENCE engineering estimate]**; soft soles only
- H05 crawl speed U(0.3, 0.8) m/s, limb impacts U(0.5, 1.0) Hz + drag term
- dwell Exp(45) cap 180 s; H19 micro-motion LU(0.05, 0.3)×walk-GRF; H19 r₀ U(0.1, 0.4)·R_det
- H04/H05 r₀ U(0.5, 1.1)·R_det

### A3. Group human (H06–H09, H12, H21, H23)
- N: Poi(3)+2 clip [2,10] (crowds) / DU[2,4] (patrol) / DU[4,10] (H21 crew)
- phases: decorrelated U(0, 1/cadence) (H06); near-in-step jitter N(0, 0.08/cadence) s (H07/H23)
- convergence point: sensor-offset U(−10, 10) m/axis; member spread N(point, σ), σ U(0.5, 5) m
- H12: radius LU(10, 0.8·R_det), angular speed U(0.2, 0.8) rad/s, duration Exp(120) **cap 240 s**
- H17 intervals: run U(30, 120) s / walk U(60, 180) s

### A4. Road vehicles (V01, V02, V04, V06, V12)
- duration: residence-time (8–40 s typical); car speed LN(ln 60, 0.35) clip [15, 130] km/h;
  dirt LU(**5**, 40); truck N(50, 15) clip [15, 90]; moto LN(ln 70, 0.45) clip [20, 140];
  bus N(40, 10) clip [20, 70]
- mass: car LN(ln 1600, 0.25) clip [900, 3500]; truck LU(3500, 40000) kg; axles: car 2, truck DU[2,5]
- road offset d_road LU(5, 250) m; roughness IRI: paved LU(0.5, 3), dirt LU(3, 12) m/km
- **joint sampling:** gear → RPM = f(speed, gear, wheel) → harmonic comb; speed cap ≈ min(v_max, 80/IRI)
- idle RPM N(750, 80)

### A5. Light/electric vehicles (V08–V11, V17, H24, V20)
- speeds: bike N(18, 5) [10, 30]; MTB N(15, 5) [8, 25]; e-scooter N(20, 5) [10, 30];
  ATV LN(ln 30, 0.4) [10, 60]; cart/robot U(2, 25) km/h
- masses: bike N(85, 12) [65, 120]; ATV LN(ln 350, 0.25) [200, 600]; cart U(250, 600) kg
- e-scooter roughness ×U(1.5, 4) of car-normalized; load↔speed negative corr.
- terrain locks per compatibility matrix (paved-only sets enforced)

### A6. Stop/idle/heavy (V03, V05, V07, V13–V16, V18, V19)
- V03 idle Exp(90) **cap 300 s**; scene 60–360 s
- V13 convoy gap Exp(8) s, N DU[2,4]
- V14 rate LU(0.5, 6) veh/min/direction, duration Exp(90) **HARD CAP 180 s, ≤200 scenes**
- V15 handoff gap U(5, 30) s
- **V16 agricultural crawler [v1.1]: speed U(1, 8) km/h, mass LU(3e3, 2.5e4) kg, pitch U(0.118, 0.160) m**
- **V19 military tracked [v1.1]: APC/IFV mass LU(1e4, 3e4) OR MBT LU(5e4, 7e4) kg (sample subtype), speed U(40, 70) km/h, pitch U(0.190, 0.194) m**; track-pitch comb f = speed/pitch
- **V18 forklift/telehandler [v1.1]: mass LU(3e3, 1.5e4) kg**; stop-go cycles Poi(4)/scene

### A7. Animal transit (A01, A05, A08–A10, A13–A15, A17)
- duration residence-time + 5 s; dog trot N(3.5, 0.8) [1.5, 7], gallop N(6, 1) [4, 9] m/s
- gazelle N(12, 3) [6, 20] m/s; pronk Poi(0.3)/s (impulse: 4-leg, 6–10 BW, 0.05–0.10 s)
- horse walk N(1.6, 0.3) / trot N(4.0, 0.8) m/s, transition at U(0.2, 0.8)·duration
- jackal U(0.3, 1.5) m/s + pauses Poi(0.3)/s dur Exp(4) s; porcupine U(0.2, 0.8) m/s + dig
  bursts (2–5 Hz scratch, 0.5–3 s bursts)
- allometric locks: f_stride ∝ **M^−0.148 [v1.1]**, t_contact ∝ M^0.148, GRF ∝ M (always enforced)
- **fore/hind asymmetry [v1.1]: forelimbs ~57% BW, peak ~30–40% higher than hind (4 feet NOT identical)**
- **double-hump GRF is WALK-ONLY [v1.1]; trot/canter/gallop = single-hump; fast-gait peak 1.4–1.7 BW (gallop max ~2.5)**

### A8. Animal forage/graze/static (A03, A04, A07, A11, A12, A16)
- duration: dwell Exp(120) **cap 300 s** (A12 cap 240 s)
- drift speed LU(0.05, 0.5) m/s; wander radius U(2, 15) m
- group: boar DU[2,5], herd DU[5,10], hyrax DU[3,8]; phases decorrelated
- rooting Poi(0.8)/s/animal; r₀ constraints per catalog (≤0.5–0.6·R_det)

### A9. Mixed/timeline (X01–X10)
- sequential gaps Exp(60) clip [10, 300] s; gap <15 s ⇒ transition windows flagged
- X01 gap U(30, 90); X08 = 3 events, total **cap 360 s**
- X06 wind ramp U(0.5, 2.0) m/s per 10 s + intruder-SNR guard (>6 dB first 20% windows)
- component durations drawn from their family caps FIRST, then sum-capped

### A10. Nothing — ambient/weather (N01–N15, N33, N38)
- duration LU(30, 600) s (stationary ⇒ long is cheap information); N14 U(120, 600) cap 300
  scenes; N15 LU(120, 3600) gated dist_to_coast ≤ 500 m
- wind tiers U(3,6)/U(7,12)/U(13,20) m/s; gust rate Poi(0.5·U)/min; N06 gust period U(1.5, 2.5) s
- **gust-AM intensity (R3-FITTED): envelope sigma/mu ~ U(0.5, 0.9)** (measured across
  ~2,000 wind hours; replaces the literature 0.15–0.3), tau ≈ 100/U s (confirmed 4–20 s)
- **meso-drift (binding): scenes >120 s add a slow level drift** sampled from the fitted
  hour-to-hour level variation of their condition bin (r3_fits p10–p90)
- rain LU(2, 10) mm/h steady; burst: onset U(5, 30) s → peak LU(15, 30) → decay Exp(60) s
- hail LU(0.5, 5)/s heavy-tailed; thunder inter-arrival Exp(40) s
- crack-pop rates (N12/N38): Poi(0.5–5)/min, peaking at dawn warming (N38) / night cooling (N12)

### A11. Nothing — infrastructure (N16–N23, N31, N34–N37, N39, N41)
- duration LU(60, 600) s
- mains LU(1, 10)% FS, drift U(0.01, 0.05) Hz/s; transformer 100 Hz ± U(0.5, 2) Hz sidebands
- pump ON U(5, 17) / OFF U(17, 34) min; sprinkler zones: surge every U(30, 120) min, zone U(5, 20) min
- **pile driver [v1.1]: impulsive broadband — per-impact spectrum 7–50 Hz, blow-repetition (comb spacing) U(0.5, 1.5) Hz, amplitude LU(0.3, 3)×walk-GRF-at-distance (NOT a 0.5–2 Hz tone)**
- **pump [v1.1]: BPF = (RPM/60)×N_vanes; RPM ~ pole-count motor (shaft ~25 Hz), N_vanes DU[3,7] → 75–175 Hz (NOT fixed 75 Hz)**
- jackhammer 8–15 Hz bursts, duty U(0.3, 0.7); generator N(1500, 50) RPM comb
- pipe thumps Poi(0.1)/s, amp LU(0.5, 5)×walk-GRF-equiv; gate slams Poi + shift-change bursts
- livestock pen (N34): 5–30 animals at fixed 10–30 m, feeding bursts 2–5 min, 2×/day phase

### A12. Nothing — overflights/transport (N24–N30, N40–N42)
- airliner N(90, 30) s [30, 180], alt LU(1, 10) km; prop alt LU(0.3, 3) km
- helicopter alt LU(50, 500) m, **BPF=N_blades·RPM/60, N_blades DU[2,5], RPM LU(300,600) → 8–30 Hz [v1.1]**, speed N(150, 30) km/h
- train U(60, 300) s, distance LU(0.5, 5) km, bogie thuds 1.5–3 Hz
- **DST micro-quake (N40) [v1.1]: ML U(1.5, 3.0), dist LU(20, 100) km, burst 5–30 s; amplitude via log10(M0)=1.5·ML+9.1 → regional GMPE → ground velocity, band 0.5–20 Hz**

## B. Corpus duration mix (binding)
Overall scene duration ~ **LN(ln 75 s, 0.8), clip [10, 600] s** (median ~75, mean ~120):
~20% in 10–30 s, 35% in 30–90, 25% in 90–180, 15% in 180–360, 5% in 360–600.
Nothing-class mean ≈ 150 s vs subject ≈ 75 s (delivers window-level nothing share without
extra scenes). Hard caps: V14 180 s · H19 180 s · V03 300 s · A07 300 s · A12 240 s ·
H12 240 s · N14 300 s · X08 360 s. Generation log warns when residence formula exceeds cap.

## C. Labeling & SNR definitions (binding; **synced to GENERATION_PLAN v1.1 — three-zone**)
- **SNR band: 5–90 Hz common [v1.1]** (was 5–80; broadband would hand vehicles a LF advantage);
  diagnostic bands stored: **human/animal 20–90, vehicle 5–25 [v1.1]**. SNR_dB =
  20·log10(in-band RMS_signal / RMS_noise), same window (signal & noise rendered separately).
- **THREE-ZONE labeling [v1.1] (replaces binary):**
  - `detectable` SNR ≥ 0 dB → hard positive (frozen test scores this).
  - `marginal` **−20 ≤ SNR < 0 dB → soft target = sigmoid(SNR/6)** (was −10 hard-nothing;
    widened per detection-theory integration gain — verified empirically by the SNR-vs-recall
    curve; ±3 dB flip-audit <15%; fallback −15 dB pre-authorized).
  - `nothing` SNR < −20 dB → hard nothing.
- **Activity label** per window: any source ≥ 0 dB (for the activity head).
- **Event table:** contiguous presence with ≥1 detectable window + attached marginal windows;
  event-level metrics (latency, false-events/day) score against this.
- Train/test mismatch acknowledged: soft training vs hard-binary test → report marginal-zone-
  disaggregated metrics + hard-label ablation. Soft labels are derived training VIEWS.
- Peak-SNR-at-closest-approach by tier: E U(10, 25) / M U(3, 15) / H U(0, 8) / A U(−3, 5)
  dB above threshold; stored per scene.
- Per-window manifest fields: `snr_db`, `scene_difficulty_tag`, `window_rank_within_scene`,
  `is_transition_window`, `is_mid_zone_open`, `multi_source_flag`, per-source SNRs (multi),
  `ambient_floor_percentile`. Curriculum score = (1 − snr/max_snr_in_scene) × {E .25, M .5,
  H .75, A 1.0}.

## D. Correlated-sampling registry (never sample independently)
Humans: speed↔cadence (stride link) · mass↔GRF · speed↔GRF factor · footwear↔τ_rise ·
load↔speed/cadence · stealth speed↔cadence.
Animals: mass↔stride freq (M^−0.14) · mass↔contact (M^0.14) · mass↔GRF · speed↔stride
within gait, jump at transitions · herd size↔scene SNR/duration.
Vehicles: gear→speed→RPM→harmonic comb (hard link) · mass↔axle load↔roughness amplitude ·
IRI↔speed cap · e-scooter load↔speed.
Weather/site: wind↔gust rate · wind↔vegetation level · rain rate↔duration (convective
short, stratiform long) · rain↔in-scene slow Vs drift · storm↔intruder-SNR guard ·
pile-driver rate↔amplitude.
Geometry: r₀↔duration (transits) and r₀-inward locks (static archetypes) · closest
approach↔peak SNR (the §7 continuum) · group N↔superposed energy.

## E. Per-terrain ambient-floor percentiles (anti terrain→class cue)
Sample ambient level percentile within the NLNM–NHNM range per scene, by family prior:
soft soil/loess 50–70th · rock/paving 30–50th · snow 20–40th · sabkha/frozen 40–60th ·
others 35–65th. Stored as `ambient_floor_percentile`.
