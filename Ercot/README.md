# ERCOT DuckDB Pipeline (Julia 1.11.6)

This project implements the "wire-it-once" workflow described in the trader briefings. It provides a reproducible Julia 1.11.6 environment that can scrape ERCOT Market Information System (MIS) index pages, download the public data products listed by EMIL ID, unpack archives, and load the results into a DuckDB database.

## Layout

```
ercot-pipeline/
  Project.toml / Manifest.toml   # Julia environment (Julia 1.11.6)
  fetch_and_ingest.jl            # Daily entrypoint (download → extract → ingest)
  config/datasets.json           # Declarative data catalog keyed by EMIL IDs
  data/
    raw/                         # As-downloaded source files (zip/csv/gz/xls)
    staging/                     # Extracted CSVs ready for DuckDB
    manifests/                   # Per-dataset SHA256 + metadata manifests
    duckdb/ercot.duckdb          # DuckDB file warehouse
```

## Configuring datasets

`config/datasets.json` enumerates the MIS products grouped exactly as described in **instructions 1.md**:

* **Core price feeds** (Δ = RT − DA foundation)
* **RTC+B co-optimization & scarcity pack**
* **Context drivers** (load, weather, renewables, outages)
* **Handy references & trials**

Each entry contains the EMIL ID, the MIS detail page to scrape, a regex that selects downloadable files, and the DuckDB destination table. Adjust the `index_url` if ERCOT publishes a more direct directory listing (e.g. Trials vs Production roots) or swap `mode` to `"urls"` when you have explicit download links.

## Running the pipeline

```
# Activate the Julia environment once
julia --project=. -e 'using Pkg; Pkg.instantiate()'

# Run the pipeline (downloads, extracts, ingests)
julia --project=. fetch_and_ingest.jl

# Provide an alternate config if needed
julia --project=. fetch_and_ingest.jl path/to/custom_config.json
```

The script keeps manifests keyed by URL + SHA256 to avoid re-downloading files and uses DuckDB's `read_csv_auto(..., union_by_name=TRUE)` to land everything into the `ercot.duckdb` database idempotently.

## Daily operations

* Schedule via cron: `0 10 * * * /usr/local/bin/julia --project /path/to/ercot-pipeline/fetch_and_ingest.jl >> /path/to/logs/ercot.log 2>&1`
* Add throttling or authentication by editing the helpers at the top of `fetch_and_ingest.jl`.
* Extend the config with additional EMIL IDs by appending new objects to `datasets.json`.

DuckDB keeps historical tables hot locally, so traders can query spreads, scarcity adders, and contextual drivers from a single file-backed database.

## GPU-aware state assimilation prototype

The `src/` directory now contains a small SciML-based prototype that can run on CPU or GPU:

- `Device.jl` detects whether CUDA (NVIDIA), ROCm (AMD), or Metal (macOS) is available at runtime and returns `CPUDevice` or `GPUDevice` wrappers accordingly. All state vectors and noise draws adapt through this abstraction.
- `AssimilationModel.jl` defines a simple RTC+B-inspired state ODE (load, wind, solar, thermal outages) and exposes `build_rtc_state_model`, `simulate_ensemble`, and `ensemble_matrix`. When a GPU is present the ODE integrates on CuArrays automatically.
- `EnKF.jl` provides a lightweight Ensemble Kalman Filter update that fuses an observation vector into the ensemble forecast.

Try it end-to-end:

```
julia scripts/run_assimilation.jl
```

The script activates the project, detects the device, simulates an ensemble, performs a synthetic EnKF update, and prints both the observation and posterior state mean. This scaffold will be extended with the PTDF learner, LMSR market layer, and RL control policy outlined in `docs/rtc_bayesian_market.md`.
