module AssimilationRunner

using DataFrames
using Dates
using DuckDB
using LinearAlgebra
using Random

import ..Device: detect_device
import ..AssimilationModel: RTCStateModel, build_rtc_state_model, simulate_ensemble
import ..EnKF: build_enkf, update_ensemble!

export analyze_and_forecast!, build_state_labels

const DEFAULT_MU_LIMIT = 20
const DB_PATH = abspath(joinpath(@__DIR__, "..", "data", "duckdb", "ercot.duckdb"))

struct AssimilationStateConfig
    labels::Vector{Symbol}
    mu_pairs::Vector{Tuple{Symbol,String}}
end

function top_mu_pairs(db::DuckDB.DB; limit::Int=DEFAULT_MU_LIMIT)
    df = DataFrame(DuckDB.execute(db, """
        SELECT constraint_name
        FROM ref.top_constraints
        ORDER BY abs_mu DESC
        LIMIT ?
    """, [limit]))
    pairs = Tuple{Symbol,String}[]
    for name in df.constraint_name
        raw = String(name)
        sanitized = replace(raw, r"[^A-Za-z0-9_]" => "_")
        label = Symbol("CONSTRAINT_", sanitized)
        push!(pairs, (label, raw))
    end
    return pairs
end

function build_state_labels(db::DuckDB.DB; limit::Int=DEFAULT_MU_LIMIT)
    mu_pairs = top_mu_pairs(db; limit=limit)
    base = [:load, :wind, :solar, :thermal_outage, :lambda, :prc, :mcpc_ecrs, :mcpc_nspin]
    labels = vcat(base, first.(mu_pairs))
    return AssimilationStateConfig(labels, mu_pairs)
end

function build_state_model(config::AssimilationStateConfig;
                           device = detect_device(),
                           Δt = 300.0)
    n = length(config.labels)
    equilibrium = zeros(Float64, n)
    decay_rates = fill(0.10, n)
    process_scale = fill(50.0, n)
    return build_rtc_state_model(config.labels;
                                 device=device,
                                 Δt=Δt,
                                 equilibrium=equilibrium,
                                 decay_rates=decay_rates,
                                 process_scale=process_scale)
end

function latest_fact_row(db::DuckDB.DB)
    df = DataFrame(DuckDB.execute(db, """
        WITH latest AS (SELECT max(sced_ts_utc_minute) AS ts FROM mart.fact_rt_5min)
        SELECT *
        FROM mart.fact_rt_5min f
        JOIN latest l ON f.sced_ts_utc_minute = l.ts
        LIMIT 1
    """))
    isempty(df) && error("fact_rt_5min has no rows")
    return df[1, :]
end

function latest_mu_row(db::DuckDB.DB)
    df = DataFrame(DuckDB.execute(db, """
        SELECT *
        FROM features.sced_mu
        ORDER BY sced_ts_utc DESC
        LIMIT 1
    """))
    isempty(df) && error("features.sced_mu has no rows")
    return df[1, :]
end

function observation_from_latest(db::DuckDB.DB, config::AssimilationStateConfig)
    fact = latest_fact_row(db)
    mu_row = latest_mu_row(db)

    observed = Dict{Symbol,Tuple{Float64,Float64}}()
    observed[:load] = (Float64(fact.system_load_forecast_mw), 25_000.0)
    observed[:wind] = (Float64(fact.wind_system_mw), 10_000.0)
    observed[:solar] = (Float64(fact.solar_system_mw), 5_000.0)
    observed[:thermal_outage] = (Float64(get(fact, :outage_resource_south_mw, 0.0)), 10_000.0)
    observed[:lambda] = (Float64(fact.system_lambda), 1.0)
    observed[:prc] = (Float64(fact.prc), 4.0)
    observed[:mcpc_ecrs] = (Float64(fact.mcpc_ecrs), 4.0)
    observed[:mcpc_nspin] = (Float64(fact.mcpc_nspin), 4.0)

    for (label, column) in config.mu_pairs
        if hasproperty(mu_row, Symbol(column))
            value = Float64(coalesce(mu_row[Symbol(column)], 0.0))
            observed[label] = (value, 9.0)
        end
    end

    obs_labels = collect(keys(observed))
    n_obs = length(obs_labels)
    H = zeros(n_obs, length(config.labels))
    y = zeros(Float64, n_obs)
    R = zeros(Float64, n_obs)

    for (i, lbl) in enumerate(obs_labels)
        idx = findfirst(==(lbl), config.labels)
        idx === nothing && continue
        H[i, idx] = 1.0
        y[i] = observed[lbl][1]
        R[i] = observed[lbl][2]
    end

    meta = Dict{Symbol,Any}(
        :fact_timestamp => get(fact, :sced_ts_utc, missing),
        :fact_minute => get(fact, :sced_ts_utc_minute, missing),
        :mu_timestamp => get(mu_row, :sced_ts_utc, missing),
        :mu_minute => get(mu_row, :sced_ts_utc_minute, missing)
    )

    return H, y, Diagonal(R) |> Matrix, meta
end

function analyze_and_forecast!(; mu_limit::Int=DEFAULT_MU_LIMIT, ensemble_size::Int=64, rng::AbstractRNG=Random.default_rng())
    db = DuckDB.DB(DB_PATH)
    try
        config = build_state_labels(db; limit=mu_limit)
        model = build_state_model(config)
        Xf = simulate_ensemble(model; ensemble_size=ensemble_size, rng=rng)
        H, y, R, meta = observation_from_latest(db, config)
        filter = build_enkf(H=H, R=R)
        Xa = update_ensemble!(filter, Xf, y)
        return model, Xa, config, meta
    finally
        close(db)
    end
end

end
