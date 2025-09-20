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

# Sweep manifests (drop missing files, keep the newest 10k URLs)
julia --project=. scripts/prune_manifests.jl --keep 10000

# Run the pipeline (downloads, extracts, ingests)
julia --project=. fetch_and_ingest.jl --manifest-keep 10000

# Provide an alternate config or dataset subset if needed
julia --project=. fetch_and_ingest.jl --manifest-keep 10000 path/to/custom_config.json
```

The script keeps manifests keyed by URL + SHA256 to avoid re-downloading files and uses DuckDB's `read_csv_auto(..., union_by_name=TRUE)` to land everything into the `ercot.duckdb` database idempotently.

## Daily operations

* Schedule the manifest sweep a few minutes before ingestion: `55 09 * * * /usr/local/bin/julia --project /path/to/ercot-pipeline/scripts/prune_manifests.jl --keep 10000 >> /path/to/logs/prune_manifests.log 2>&1`
* Then run the fetch with manifest retention and capped downloads: `0 10 * * * /usr/local/bin/julia --project /path/to/ercot-pipeline/fetch_and_ingest.jl --manifest-keep 10000 --max-new 3 >> /path/to/logs/ercot.log 2>&1`
* After the SCED ingest lands the latest `features.sced_mu` snapshot, run the logistic calibrator nightly to keep `ref.logistic_map_params` current: `30 10 * * * /usr/local/bin/julia --project /path/to/ercot-pipeline/scripts/calibrate_logistics.jl >> /path/to/logs/calibrate_logistics.log 2>&1`
* Add throttling or authentication by editing the helpers at the top of `fetch_and_ingest.jl`.
* Extend the config with additional EMIL IDs by appending new objects to `datasets.json`.

### Lagged training pipeline (cron / CI)

The lagged web dashboard and RL experiments expect yesterday’s data to be summarized each morning. Chain these jobs after the nightly ingest:

1. **Publish lag snapshot** – records the latest minute and event count in `mart.lag_snapshot_log`.
   ```
   5 10 * * * /usr/local/bin/julia /path/to/ercot-pipeline/scripts/publish_lag_snapshot.jl >> /path/to/logs/publish_lag_snapshot.log 2>&1
   ```
   Set `LAG_SNAPSHOT_SOURCE` to tag different environments (e.g. `prod`, `staging`).
2. **Optional automated backtests** – sanity-check the scenario stack via the lagged API.
   ```
   15 10 * * * BACKTEST_LIQUIDITY=5.0 /usr/local/bin/julia /path/to/ercot-pipeline/scripts/run_backtests.jl >> /path/to/logs/backtests.log 2>&1
   ```
   Override `BACKTEST_DATE`, `BACKTEST_TOP_CONSTRAINTS`, etc. to target specific windows; CI jobs can reuse the same script before deployments.

DuckDB keeps historical tables hot locally, so traders can query spreads, scarcity adders, and contextual drivers from a single file-backed database.

## Effective PTDF estimation

1. Materialize constraint and target features: `julia scripts/apply_views.jl`
2. Fit elastic-net sensitivities and persist outputs: `julia scripts/fit_effective_ptdfs.jl`
3. Optional: schedule the fitter after nightly ingestion to refresh `ref.estimated_ptdf*` tables.

The fitter now scans a small λ-grid with a 10% time-ordered hold-out and selects the λ that minimises validation RMSE/MAE before persisting. It writes:

- `ref.estimated_ptdf` with coefficients plus `abs_beta`, `avg_abs_mu`, and `expected_impact` columns
- `ref.estimated_ptdf_fit_metrics` for λ-grid diagnostics
- `ref.estimated_ptdf_intercepts`, `ref.estimated_ptdf_constraint_offsets`, `ref.estimated_ptdf_metadata`, and `ref.estimated_ptdf_node_rmse`
- `ref.estimated_ptdf_node_summary` (top-driver overview per node)

Only fits whose RMSE beats the no-μ baseline by at least 5% promote from staging into the live `ref.estimated_ptdf*` tables; otherwise the previous coefficients stay in place and the new run is retained under the `_staged` tables for inspection.

### PTDF scenario wiring

Run `julia scripts/run_ptdf_scenario.jl` to grab the most recent μ snapshot, apply the estimated PTDFs, emit DAG events (node > 25 $/MWh, constraint contribution shares), and quote them with the LMSR helpers. This gives a quick “what-if” view of constraint forecasts inside the market-of-models stack.

### PTDF REST service & trading loop

- Start the lightweight JSON service: `julia scripts/ptdf_service.jl` (configurable via `PTDF_SERVICE_PORT`/`PTDF_SERVICE_HOST`). Call `GET /scenario?nodes=HB_WEST,HB_HOUSTON&topk=3` to receive predicted prices, top drivers, and LMSR event prices.
- Drive a 5-minute trading loop prototype: `julia scripts/trading_loop.jl`. The loop checks model freshness/improvement, skips stale ticks, computes basis signals for configured nodes/hub, sizes via simple scenario shocks, and logs trade suggestions (no execution).

### Lagged web dashboard & RL sandbox

- Launch the Genie server: `julia scripts/start_webapp.jl` (override `WEBAPP_HOST`, `WEBAPP_PORT`, or DB settings via env vars). Visit `/dashboard` for a single-page UI that hits `/api/simulate`, `/api/trades`, and `/api/train` against the lagged dataset.
- The training card wraps `RLTradingEnv`; runs are logged into `mart.training_runs` (with optional notes in `mart.training_notes`). Tweak hyperparameters (risk budget, risk aversion, policy mix, scenario probabilities) from the form or call `/api/train` directly.
- Batch experiments from the CLI with `julia scripts/train_policy.jl --dates 2025-09-18 --episodes 5 --policy eps --epsilon 0.1`. The script uses the same lagged environment, records runs in DuckDB, and prints a JSON summary for CI assertions.

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
