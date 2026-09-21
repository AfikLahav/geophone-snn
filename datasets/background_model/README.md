# Background model

Real backgrounds vary with location and weather. `r3_fits.npz` stores measured distributions of power across frequency so the generator can vary its background noise without storing the original recordings in the repository.

## Sources

The archive keys and download scripts identify four source groups:

| Key | Source | Recording period requested by the downloader |
|---|---|---|
| `yw` | [IRIS Community Wavefield Experiment, Oklahoma](https://www.fdsn.org/networks/detail/YW_2016/) | June–July 2016 |
| `lasso` | [LArge-n Seismic Survey in Oklahoma](https://www.fdsn.org/networks/detail/2A_2016/) | April–May 2016 |
| `zg` | [Sage Brush Flats Nodal Experiment, San Jacinto fault zone](https://www.fdsn.org/networks/detail/ZG_2014/) | May–June 2014 |
| `is_il` | [Israel Seismic Network](https://www.fdsn.org/networks/detail/IS/) | January–February and April–May 2021 |

These include both US and Israeli recordings. They are not all measurements from the same type of sensor. The download code removes the instrument response to obtain ground velocity before storing the recordings. Weather information comes from the [Open-Meteo historical API](https://open-meteo.com/en/docs/historical-weather-api).

## Stored values

The file contains 21 source-and-weather groups. Each has a frequency axis (`f`), median power spectral density (`med`), and 10th and 90th percentiles (`p10`, `p90`). Power spectral density describes how signal power is distributed over frequency. Three additional arrays store time-of-day summaries for Israeli stations.

[r3_noise.py](../../simgeo/simgeo_v42/r3_noise.py) uses the frequency and median arrays. It selects a curve for the scene's weather, normalizes it over 1–20 Hz, and generates random background motion with that frequency distribution. Relative levels between weather groups are combined with the generator's amplitude calibration. The percentile and time-of-day arrays are retained as reference data and are not read by this generation path.

The measured frequency range differs between sources. Above a selected curve's range, the generator reduces power in proportion to the inverse square of frequency. That continuation is modeled, not measured. Separate code adds narrow frequency peaks, brief disturbances, slow level changes, and wind modulation. The output is ground velocity in metres per second, which then passes through the simulated geophone.

## Downloads

[real_noise_fetch](../real_noise_fetch) contains the recording downloader. It writes outside the repository under `GEO_NOISE_RAW` and skips files already present. Downloading is not needed to use the supplied fit file, and the downloader alone does not rebuild that file. Source attribution is listed in [THIRD_PARTY.md](../../THIRD_PARTY.md).

The optional downloader needs additional packages beyond the generation and training requirements:

```bash
python -m pip install obspy openmeteo-requests requests-cache retry-requests requests certifi
```
