# Third-party software and data

The [project license](LICENSE) covers original project code. External software, fonts, and data retain their own licenses and attribution requirements.

## Software

| Component | License | Use |
|---|---|---|
| [pyprop8](https://github.com/valentineap/pyprop8) | GPL-3.0 | Ground responses for layered media |
| [NumPy](https://numpy.org/) | BSD-3-Clause | Numerical arrays |
| [SciPy](https://scipy.org/) | BSD-3-Clause | Signal processing |
| [PyTorch](https://pytorch.org/) | BSD-3-Clause | Model training and inference |
| [SpikingJelly](https://github.com/fangwei123456/spikingjelly) | Apache-2.0 | Spiking neurons |
| [KaTeX](https://github.com/KaTeX/KaTeX/blob/main/LICENSE) | MIT | Formulas in the feature catalog |

Python dependencies are listed in [requirements.txt](requirements.txt). pyprop8 is installed as a dependency; its source is not copied into this repository. The feature catalog includes local KaTeX fonts.

## Background recordings

[r3_fits.npz](datasets/background_model/r3_fits.npz) contains aggregate background statistics associated with these recording groups:

| Source | Network | Reference |
|---|---|---|
| IRIS Community Wavefield Experiment in Oklahoma | YW, 2016 | [10.7914/SN/YW_2016](https://doi.org/10.7914/SN/YW_2016) |
| LArge-n Seismic Survey in Oklahoma | 2A, 2016 | [10.7914/SN/2A_2016](https://doi.org/10.7914/SN/2A_2016) |
| Sage Brush Flats Nodal Experiment | ZG, 2014 | [10.7914/SN/ZG_2014](https://doi.org/10.7914/SN/ZG_2014) |
| Israel Seismic Network | IS | [FDSN network record](https://www.fdsn.org/networks/detail/IS/) |

The downloader requests waveforms and instrument metadata through EarthScope/IRIS and GEOFON services. Associated weather data are requested from the [Open-Meteo historical API](https://open-meteo.com/en/docs/historical-weather-api). Raw recordings and weather downloads are not included. The [background-model notes](datasets/background_model/README.md) describe the saved statistics and their use.

## Other research data

[FootprintID](https://doi.org/10.5281/zenodo.4691144) and [PNW-ML](https://github.com/niyiyu/PNW-ML) were used as development references for footstep and background characteristics. They are distinct from the four source groups in `r3_fits.npz`.

Elbit supplied the field recordings used in the report. They are not distributed here. The external evaluation scripts also require separately obtained datasets, including SeisSavanna and M3N-VC. Public availability of a dataset does not place it under this project's license; consult its source record for its terms and citation.
