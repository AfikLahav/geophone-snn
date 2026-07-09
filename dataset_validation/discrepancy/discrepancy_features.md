# Feature-Space Discrepancy Report — v4 Real vs Synthetic

N_synth=23999, N_real=2353 windows | 132 features | 104 model features | clip=±8

## 1. Per-Family Domain AUROC (all classes pooled)

AUROC = P(real scores higher than synth). 0.5=indistinguishable, >0.7=notable gap.

| Family     | N_feat | Median AUROC | Max AUROC | Max |dev| |
|------------|--------|-------------|-----------|----------|
| NL1        |      1 | 0.8699      | 0.8699    | 0.3699   |
| WPE16      |     16 | 0.7587      | 0.8225    | 0.3225   |
| AR8        |      8 | 0.6171      | 0.9318    | 0.4399   |
| COOC3      |      3 | 0.5769      | 0.5781    | 0.0781   |
| III5       |      5 | 0.5435      | 0.6214    | 0.1214   |
| HOS3       |      3 | 0.5422      | 0.5454    | 0.1154   |
| CAD12      |     12 | 0.5356      | 0.6546    | 0.1662   |
| WX2        |      2 | 0.5218      | 0.5328    | 0.0328   |
| MOD7       |      7 | 0.5166      | 0.5944    | 0.1198   |
| TONAL7     |      7 | 0.5102      | 0.6581    | 0.1581   |
| CEP16      |     16 | 0.5086      | 0.8140    | 0.3667   |
| ENG8       |      8 | 0.4939      | 0.6653    | 0.1653   |
| PHYS7      |      7 | 0.4910      | 0.7298    | 0.4836   |
| LEGACY32   |     32 | 0.4494      | 0.7981    | 0.4837   |
| STAT5      |      5 | 0.3827      | 0.4956    | 0.1381   |

Overall median AUROC across all 132 features: **0.5289**

## 2. Top 25 Most Domain-Separable Features (|AUROC−0.5| rank)

| Rank | Feature                          | AUROC  | Family   |
|------|----------------------------------|--------|----------|
|    1 | variance                         | 0.0163 | LEGACY32 |
|    2 | rms_total                        | 0.0164 | LEGACY32 |
|    3 | log_rms                          | 0.0164 | PHYS7    |
|    4 | peak_to_peak                     | 0.0189 | LEGACY32 |
|    5 | energy_mid_gap                   | 0.0234 | LEGACY32 |
|    6 | energy_low_freq                  | 0.0263 | LEGACY32 |
|    7 | energy_car_approach              | 0.0284 | LEGACY32 |
|    8 | energy_car_tail                  | 0.0307 | LEGACY32 |
|    9 | energy_car_peak                  | 0.0316 | LEGACY32 |
|   10 | energy_human_peak                | 0.0356 | LEGACY32 |
|   11 | energy_human_tail                | 0.0437 | LEGACY32 |
|   12 | energy_high_freq                 | 0.0552 | LEGACY32 |
|   13 | ar_5                             | 0.0601 | AR8      |
|   14 | ar_6                             | 0.9318 | AR8      |
|   15 | ar_4                             | 0.9261 | AR8      |
|   16 | ar_3                             | 0.1052 | AR8      |
|   17 | ar_7                             | 0.1231 | AR8      |
|   18 | higuchi_fd                       | 0.8699 | NL1      |
|   19 | mfcc_1                           | 0.1333 | CEP16    |
|   20 | ar_8                             | 0.8322 | AR8      |
|   21 | wpe_13                           | 0.8225 | WPE16    |
|   22 | frac_foot_20_90                  | 0.1801 | PHYS7    |
|   23 | wpe_15                           | 0.8171 | WPE16    |
|   24 | lfcc_2                           | 0.8140 | CEP16    |
|   25 | event_count                      | 0.7981 | LEGACY32 |

## 3. OOD Analysis — Top 25 Features by Median |z| (scaler.json z-units)

|z|>2 = out-of-distribution for the model. Scaler covers 104 model features.

| Rank | Feature                          | Med|z| | P90|z| | Frac>2 | Frac>4 |
|------|----------------------------------|--------|--------|--------|--------|
|    1 | log_rms                          |   3.58 |   4.70 |  90.9% |  26.9% |
|    2 | peak_to_peak                     |   2.20 |   2.24 |  89.8% |   0.0% |
|    3 | higuchi_fd                       |   1.92 |   3.29 |  47.5% |   0.1% |
|    4 | frac_foot_20_90                  |   1.58 |   2.82 |  36.8% |   0.0% |
|    5 | dominant_freq                    |   1.55 |   1.66 |   6.5% |   1.5% |
|    6 | lfcc_2                           |   1.54 |   2.26 |  19.3% |   0.0% |
|    7 | frac_veh_5_25                    |   1.32 |   1.91 |   8.5% |   0.0% |
|    8 | spectral_rolloff                 |   1.28 |   2.61 |  26.4% |   2.1% |
|    9 | lfcc_1                           |   1.23 |   2.36 |  17.5% |   0.0% |
|   10 | zcr                              |   1.20 |   3.84 |  34.7% |   8.9% |
|   11 | spectral_centroid                |   1.16 |   2.31 |  20.9% |   0.0% |
|   12 | hop_band_frac                    |   1.13 |   2.77 |  15.6% |   2.8% |
|   13 | event_count                      |   1.08 |   3.07 |  24.4% |   2.7% |
|   14 | wpe_1                            |   1.05 |   1.45 |   0.0% |   0.0% |
|   15 | attack_sharpness                 |   0.95 |   0.97 |   0.0% |   0.0% |
|   16 | spec_kurt                        |   0.93 |   3.23 |  27.1% |   0.0% |
|   17 | iii_entropy                      |   0.93 |   1.14 |   0.0% |   0.0% |
|   18 | iii_cv                           |   0.92 |   1.72 |   6.1% |   0.2% |
|   19 | tonal_index                      |   0.89 |   1.31 |   0.9% |   0.0% |
|   20 | line_stability                   |   0.88 |   1.87 |   7.6% |   0.0% |
|   21 | gust_mod                         |   0.88 |   1.08 |   1.1% |   0.0% |
|   22 | band_occupancy                   |   0.87 |   2.24 |  16.4% |   0.0% |
|   23 | mod_e_1_3                        |   0.86 |   1.88 |   7.5% |   2.0% |
|   24 | mfcc_2                           |   0.85 |   1.92 |   8.5% |   0.0% |
|   25 | lfcc_3                           |   0.83 |   1.64 |   3.6% |   0.0% |

### Per-Family OOD (104-feature subset)

| Family     | Median Med|z| | Max Med|z| | MeanFrac>2 |
|------------|-------------|-----------|------------|
| NL1        |          1.92 |      1.92 |      47.5% |
| TONAL7     |          0.82 |      0.89 |       4.7% |
| PHYS7      |          0.81 |      3.58 |      26.7% |
| COOC3      |          0.78 |      0.87 |      12.3% |
| HOS3       |          0.73 |      0.93 |      14.3% |
| III5       |          0.69 |      0.93 |       6.6% |
| LEGACY32   |          0.67 |      2.20 |      10.8% |
| MOD7       |          0.66 |      0.86 |       5.7% |
| STAT5      |          0.65 |      0.95 |       1.6% |
| CAD12      |          0.61 |      0.92 |       6.0% |
| CEP16      |          0.60 |      1.54 |       5.4% |
| ENG8       |          0.60 |      1.13 |       5.2% |
| WX2        |          0.54 |      0.61 |       3.5% |
| AR8        |          0.49 |      0.50 |       2.3% |
| WPE16      |          0.42 |      1.05 |      14.4% |

## 4. CLIPZ Saturation — Top Offenders

Fraction of real windows where |z| ≥ 8 (model sees railed feature value).

| Rank | Feature                          | Real Clipped | Synth Clipped |
|------|----------------------------------|-------------|---------------|
|    1 | wpe_5                            |       11.8% |          0.2% |
|    2 | wpe_13                           |       10.2% |          0.3% |
|    3 | wpe_12                           |        7.4% |          0.3% |
|    4 | wpe_15                           |        2.3% |          0.3% |
|    5 | mod_ratio_low_mid                |        1.7% |          0.1% |
|    6 | wpe_7                            |        1.4% |          0.3% |
|    7 | wpe_10                           |        1.3% |          0.3% |
|    8 | wpe_11                           |        1.2% |          0.3% |
|    9 | cad_salience                     |        1.2% |          0.3% |
|   10 | skewness                         |        1.1% |          0.2% |
|   11 | wpe_14                           |        0.8% |          0.2% |
|   12 | kurtosis                         |        0.4% |          0.2% |
|   13 | frac_wind_1_5                    |        0.3% |          0.0% |

## 5. Class-Conditional Domain AUROC

Does the domain gap come from the class signal or the background?
AUROC computed separately for matched class pairs (real_human vs synth_human, etc.)

Overall (all classes pooled) median AUROC: **0.5289**

| Class   | Median AUROC | Frac>0.7 | Verdict                          |
|---------|-------------|----------|----------------------------------|
| human   | 0.4829      | 29.5%    | Moderate class-signal gap |
| vehicle | 0.4956      | 15.2%    | Moderate class-signal gap |
| nothing | 0.5263      | 28.8%    | Moderate class-signal gap |

### Top-3 Most Separable Features per Class

**human:**
  - energy_low_freq: AUROC=0.0034
  - variance: AUROC=0.0060
  - rms_total: AUROC=0.0060
**vehicle:**
  - energy_mid_gap: AUROC=0.0427
  - variance: AUROC=0.0493
  - rms_total: AUROC=0.0495
**nothing:**
  - variance: AUROC=0.0016
  - peak_to_peak: AUROC=0.0017
  - rms_total: AUROC=0.0018

---
*Generated by ft_run_all.py | elapsed=21s*
