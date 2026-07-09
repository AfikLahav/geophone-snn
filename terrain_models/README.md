# terrain_models/ — soil-profile & Green's-function library

The layered-earth terrain definitions and their precomputed wave-response banks.
Shared by every generator version.

| file | what |
|---|---|
| `library_v3.json` | 340-profile v3/v4 catalog (**the active one**) |
| `library.json` | 300-profile v0/v1/v2 catalog |
| `splits_v3.json`, `splits_90_10.json` | train/val(/test) profile splits |
| `provenance_*.py` | snapshots of the samplers that produced the libraries |
| `*.npz` | the Green's-function banks (one per profile) |

Each profile is a stack of horizontal elastic layers `(thickness, Vp, Vs, ρ, Q)`.

The `*.npz` banks are **regenerable** from the libraries via
`simgeo/simgeo_v42/gfbank_build.py` + pyprop8, and are **gitignored** (large, ~156 MB).
