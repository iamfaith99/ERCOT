#!/usr/bin/env julia
using Pkg
Pkg.activate(abspath(joinpath(@__DIR__, "..")))

using DataFrames
using Dates
using DuckDB
using Logging
using Printf
using Random
using Statistics

const DB_PATH = abspath(joinpath(@__DIR__, "..", "data", "duckdb", "ercot.duckdb"))
const SCALE_GRID = 10 .^ collect(-1:0.05:1)
const CENTER_GRID_POINTS = 25
const MIN_GLOBAL_SAMPLES = 500
const MIN_NODE_SAMPLES = 500

σ(z) = 1 / (1 + exp(-z))

function brier_loss(x::Vector{Float64}, y::Vector{Float64}; center::Float64, scale::Float64)
    scale <= 0 && return Inf
    p = @. σ((x - center) / scale)
    return mean((p .- y).^2)
end

function grid_search(x::Vector{Float64}, y::Vector{Float64}; centers::AbstractVector, scales::AbstractVector)
    best_center = centers[1]
    best_scale = scales[1]
    best_loss = Inf
    for c in centers
        for s in scales
            loss = brier_loss(x, y; center=c, scale=s)
            if loss < best_loss
                best_center, best_scale, best_loss = c, s, loss
            end
        end
    end
    return (center=best_center, scale=best_scale, loss=best_loss)
end

function center_grid(values::Vector{Float64})
    isempty(values) && error("Cannot build center grid for empty sample")
    lo = quantile(values, 0.1)
    hi = quantile(values, 0.9)
    if !isfinite(lo) || !isfinite(hi)
        lo, hi = minimum(values), maximum(values)
    end
    if hi <= lo
        span = max(abs(lo), 1.0)
        return [lo - span, lo, lo + span]
    end
    return collect(range(lo, hi; length=CENTER_GRID_POINTS))
end

function ensure_table!(db::DuckDB.DB)
    DuckDB.execute(db, "CREATE SCHEMA IF NOT EXISTS ref;")
    DuckDB.execute(db, """
        CREATE TABLE IF NOT EXISTS ref.logistic_map_params (
            event_kind TEXT,
            scope_key  TEXT,
            center     DOUBLE,
            scale      DOUBLE,
            sample_n   BIGINT,
            brier      DOUBLE,
            fitted_from TIMESTAMPTZ,
            fitted_to   TIMESTAMPTZ,
            updated_ts TIMESTAMPTZ DEFAULT NOW(),
            PRIMARY KEY (event_kind, scope_key)
        );
    """)
end

function write_params!(db::DuckDB.DB; event_kind::String, scope_key::String, center::Float64, scale::Float64,
                       sample_n::Int, brier::Float64, fitted_from::DateTime, fitted_to::DateTime)
    ensure_table!(db)
    DuckDB.execute(db, """
        INSERT OR REPLACE INTO ref.logistic_map_params
        (event_kind, scope_key, center, scale, sample_n, brier, fitted_from, fitted_to)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
    """, [event_kind, scope_key, center, scale, sample_n, brier, fitted_from, fitted_to])
end

function pull_training_window(db::DuckDB.DB; start::DateTime, stop::DateTime)
    mu = DataFrame(DuckDB.execute(db, """
        SELECT *
        FROM features.sced_mu
        WHERE sced_ts_utc_minute BETWEEN ? AND ?
        ORDER BY sced_ts_utc_minute
    """, [start, stop]))
    isempty(mu) && error("No records in features.sced_mu for selected window")

    beta_df = DataFrame(DuckDB.execute(db, "SELECT * FROM ref.estimated_ptdf"))
    isempty(beta_df) && error("ref.estimated_ptdf is empty")

    intercept_df = DataFrame(DuckDB.execute(db, "SELECT * FROM ref.estimated_ptdf_intercepts"))
    isempty(intercept_df) && error("ref.estimated_ptdf_intercepts is empty")

    y_df = DataFrame(DuckDB.execute(db, """
        SELECT DATE_TRUNC('minute', sced_ts_utc) AS sced_ts_utc_minute,
               settlement_point,
               rt_lmp - COALESCE(system_lambda, 0) AS y_congestion
        FROM mart.fact_rt_5min
        WHERE sced_ts_utc BETWEEN ? AND ?
    """, [start, stop]))

    return mu, beta_df, intercept_df, y_df
end

function train_node_gt25!(db::DuckDB.DB; start::DateTime, stop::DateTime, nodes_limit::Int=200)
    mu_df, beta_df, intercept_df, y_df = pull_training_window(db; start=start, stop=stop)

    nodes = String.(unique(beta_df.node))
    isempty(nodes) && error("No nodes in PTDF coefficients")
    nodes = nodes[1:min(nodes_limit, length(nodes))]

    intercept_map = Dict(String(row.node) => Float64(row.intercept) for row in eachrow(intercept_df))

    y_map = Dict{Tuple{String,DateTime},Float64}()
    for row in eachrow(y_df)
        key = (String(row.settlement_point), DateTime(row.sced_ts_utc_minute))
        y_map[key] = Float64(row.y_congestion)
    end

    feature_cols = filter(name -> !(name in ("sced_ts_utc", "sced_ts_utc_minute")), names(mu_df))

    X = Float64[]
    Y = Float64[]
    node_values = Dict{String,Vector{Float64}}()
    node_targets = Dict{String,Vector{Float64}}()

    for row in eachrow(mu_df)
        minute = DateTime(row.sced_ts_utc_minute)
        features = Dict{String,Float64}(col => Float64(coalesce(row[col], 0.0)) for col in feature_cols)
        for node in nodes
            rows = filter(:node => ==(node), beta_df)
            isempty(rows) && continue
            value = get(intercept_map, node, 0.0)
            for βrow in eachrow(rows)
                value += Float64(βrow.beta) * get(features, String(βrow.constraint_name), 0.0)
            end
            key = (node, minute)
            y = get(y_map, key, NaN)
            isnan(y) && continue
            label = y > 25 ? 1.0 : 0.0
            push!(X, value)
            push!(Y, label)
            node_vec = get!(node_values, node) do
                Float64[]
            end
            push!(node_vec, value)
            label_vec = get!(node_targets, node) do
                Float64[]
            end
            push!(label_vec, label)
        end
    end

    length(X) >= MIN_GLOBAL_SAMPLES || error("Insufficient samples for node_gt25 calibration")

    centers = center_grid(X)
    best = grid_search(X, Y; centers=centers, scales=SCALE_GRID)
    write_params!(db;
                  event_kind="node_gt25",
                  scope_key="global",
                  center=best.center,
                  scale=best.scale,
                  sample_n=length(X),
                  brier=best.loss,
                  fitted_from=start,
                  fitted_to=stop)
    @info "Calibrated node_gt25" scope="global" center=best.center scale=best.scale loss=best.loss samples=length(X)

    promoted = 0
    for (node, values) in node_values
        labels = get(node_targets, node, Float64[])
        length(values) == length(labels) || continue
        length(values) < MIN_NODE_SAMPLES && continue
        centers = center_grid(values)
        best_node = grid_search(values, labels; centers=centers, scales=SCALE_GRID)
        write_params!(db;
                      event_kind="node_gt25",
                      scope_key=node,
                      center=best_node.center,
                      scale=best_node.scale,
                      sample_n=length(values),
                      brier=best_node.loss,
                      fitted_from=start,
                      fitted_to=stop)
        promoted += 1
        @info "Calibrated node_gt25" scope=node center=best_node.center scale=best_node.scale loss=best_node.loss samples=length(values)
    end
    if promoted == 0
        @info "No node-specific calibrations promoted" min_samples=MIN_NODE_SAMPLES
    else
        @info "Promoted node-specific calibrations" count=promoted
    end
end

function train_contrib_pos!(db::DuckDB.DB; start::DateTime, stop::DateTime, sample_limit::Int=100_000)
    mu_df, beta_df, _, _ = pull_training_window(db; start=start, stop=stop)

    feature_cols = filter(name -> !(name in ("sced_ts_utc", "sced_ts_utc_minute")), names(mu_df))

    sampler = Random.default_rng()

    X = Float64[]
    Y = Float64[]

    β_rows = collect(eachrow(beta_df))
    isempty(β_rows) && error("Empty β rows for contrib calibration")

    for row in eachrow(mu_df)
        features = Dict{String,Float64}(col => Float64(coalesce(row[col], 0.0)) for col in feature_cols)
        shuffled = copy(β_rows)
        Random.shuffle!(sampler, shuffled)
        for βrow in Iterators.take(shuffled, min(length(β_rows), 2000))
            value = Float64(βrow.beta) * get(features, String(βrow.constraint_name), 0.0)
            push!(X, value)
            push!(Y, value > 0 ? 1.0 : 0.0)
            length(X) >= sample_limit && break
        end
        length(X) >= sample_limit && break
    end

    length(X) > 200 || error("Insufficient samples for contrib_pos calibration")

    centers = [0.0]
    scales = SCALE_GRID
    best = grid_search(X, Y; centers=centers, scales=scales)
    write_params!(db;
                  event_kind="contrib_pos",
                  scope_key="global",
                  center=best.center,
                  scale=best.scale,
                  sample_n=length(X),
                  brier=best.loss,
                  fitted_from=start,
                  fitted_to=stop)
    @info "Calibrated contrib_pos" center=best.center scale=best.scale loss=best.loss samples=length(X)
end

function main()
    stop_time = floor(now(Dates.UTC) - Day(1), Dates.Minute)
    start_time = stop_time - Day(7)

    db = DuckDB.DB(DB_PATH)
    try
        train_node_gt25!(db; start=start_time, stop=stop_time)
        train_contrib_pos!(db; start=start_time, stop=stop_time)
    finally
        close(db)
    end
end

main()
