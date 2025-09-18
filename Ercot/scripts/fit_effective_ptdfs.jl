#!/usr/bin/env julia
using Pkg
Pkg.activate(abspath(joinpath(@__DIR__, "..")))

using DataFrames
using Dates
using DuckDB
using GLMNet
using Logging
using Statistics

const DB_PATH = abspath(joinpath(@__DIR__, "..", "data", "duckdb", "ercot.duckdb"))
const EXTRA_REGRESSORS = [:scarcity_adder, :mcpc_regup, :mcpc_rrs]
const REG_PARAMS = (alpha = 0.1, lambda = 0.01)

function _first_or_zero(col)
    for v in col
        if !ismissing(v)
            return Float64(v)
        end
    end
    return 0.0
end

function _fetch_training_frames(db_path::AbstractString)
    db = DuckDB.DB(db_path)
    mu_df = DataFrame()
    target_df = DataFrame()
    try
        mu_df = DataFrame(DuckDB.execute(db, """
            SELECT DISTINCT mu.*
            FROM features.sced_mu AS mu
            INNER JOIN features.lmp_target AS lt USING (sced_ts_utc)
            ORDER BY sced_ts_utc
        """))

        target_df = DataFrame(DuckDB.execute(db, """
            SELECT
                lt.sced_ts_utc,
                lt.settlement_point,
                lt.y_congestion,
                lt.scarcity_adder,
                lt.mcpc_regup,
                lt.mcpc_rrs
            FROM features.lmp_target AS lt
            WHERE lt.sced_ts_utc IN (SELECT sced_ts_utc FROM features.sced_mu)
            ORDER BY lt.sced_ts_utc, lt.settlement_point
        """))
    finally
        close(db)
    end

    return mu_df, target_df
end

function _prepare_design_matrices(mu_df::DataFrame, target_df::DataFrame)
    isempty(mu_df) && error("features.sced_mu returned no rows")
    isempty(target_df) && error("features.lmp_target returned no rows")

    sort!(mu_df, :sced_ts_utc)

    scalars = combine(
        groupby(target_df, :sced_ts_utc),
        (reg => _first_or_zero => reg for reg in EXTRA_REGRESSORS)...
    )

    mu_aug = leftjoin(mu_df, scalars, on = :sced_ts_utc, order = :left)

    pivot_input = select(target_df, [:sced_ts_utc, :settlement_point, :y_congestion])
    y_pivot = unstack(pivot_input, :sced_ts_utc, :settlement_point, :y_congestion)

    sort!(mu_aug, :sced_ts_utc)
    sort!(y_pivot, :sced_ts_utc)

    common_times = intersect(mu_aug.sced_ts_utc, y_pivot.sced_ts_utc)
    isempty(common_times) && error("No overlapping timestamps between constraints and targets")

    common_set = Set(common_times)
    filter!(row -> row.sced_ts_utc in common_set, mu_aug)
    filter!(row -> row.sced_ts_utc in common_set, y_pivot)

    sort!(mu_aug, :sced_ts_utc)
    sort!(y_pivot, :sced_ts_utc)

    feature_cols = Symbol.(names(mu_aug, Not(:sced_ts_utc)))
    for col in feature_cols
        mu_aug[!, col] = Float64.(coalesce.(mu_aug[!, col], 0.0))
    end

    node_cols = Symbol.(names(y_pivot, Not(:sced_ts_utc)))
    for col in node_cols
        y_pivot[!, col] = Float64.(coalesce.(y_pivot[!, col], 0.0))
    end

    X = Matrix(select(mu_aug, feature_cols))
    Y = Matrix(select(y_pivot, node_cols))

    return (;
        X,
        Y,
        times = copy(mu_aug.sced_ts_utc),
        feature_cols = collect(feature_cols),
        node_cols = collect(node_cols)
    )
end

function _fit_glmnet(X::Matrix{Float64}, Y::Matrix{Float64})
    fit = glmnet(
        X,
        Y,
        GLMNet.MvNormal();
        alpha = REG_PARAMS.alpha,
        lambda = [REG_PARAMS.lambda],
        standardize = false,
        standardize_response = false
    )

    idx = length(fit.lambda)
    intercept = vec(fit.a0[:, idx])
    betas = fit.betas[:, :, idx]

    return intercept, betas
end

function _rebase_constraints!(betas::Matrix{Float64}, feature_cols::Vector{Symbol}, node_cols::Vector{Symbol})
    constraint_cols = [col for col in feature_cols if !(col in EXTRA_REGRESSORS)]
    offsets = Dict{Symbol,Float64}()

    for (idx, col) in enumerate(feature_cols)
        name = col
        name in constraint_cols || continue
        row_index = idx
        offset = mean(betas[row_index, :])
        betas[row_index, :] .-= offset
        offsets[name] = offset
    end

    return offsets
end

function _assemble_outputs(betas::Matrix{Float64}, intercept::Vector{Float64},
                           feature_cols::Vector{Symbol}, node_cols::Vector{Symbol},
                           offsets::Dict{Symbol,Float64})
    feature_names = String.(feature_cols)
    node_names = String.(node_cols)

    rows = Vector{NamedTuple{(:constraint_name, :node, :beta, :feature_type)}}()

    constraint_set = Set(String.(setdiff(feature_cols, EXTRA_REGRESSORS)))

    for (fi, fname) in enumerate(feature_names)
        feature_type = fname in constraint_set ? "constraint" : "regressor"
        for (ni, nname) in enumerate(node_names)
            push!(rows, (constraint_name = fname, node = nname, beta = betas[fi, ni], feature_type = feature_type))
        end
    end

    beta_df = DataFrame(rows)
    intercept_df = DataFrame(node = node_names, intercept = intercept)

    offsets_df = DataFrame(
        constraint_name = String.(collect(keys(offsets))),
        mean_offset = collect(values(offsets))
    )

    return beta_df, intercept_df, offsets_df
end

function _compute_residual_metrics(X::Matrix{Float64}, Y::Matrix{Float64}, betas::Matrix{Float64}, intercept::Vector{Float64})
    predictions = X * betas
    predictions .+= ones(size(predictions, 1)) * transpose(intercept)

    residuals = Y .- predictions
    rmse = sqrt(mean(residuals .^ 2))
    mae = mean(abs.(residuals))

    per_node_rmse = vec(sqrt.(mean(residuals .^ 2; dims = 1)))

    return (; residuals, rmse, mae, per_node_rmse)
end

function _metadata_frame(times, feature_cols, node_cols, betas, rmse, mae, per_node_rmse)
    window_start = minimum(times)
    window_end = maximum(times)
    constraint_mask = [!(col in EXTRA_REGRESSORS) for col in feature_cols]
    active_constraints = count(enumerate(constraint_mask)) do (idx, is_constraint)
        is_constraint && any(abs.(betas[idx, :]) .> 1e-6)
    end
    run_ts = Dates.now(Dates.UTC)

    meta = DataFrame(
        run_ts = [run_ts],
        window_start = [window_start],
        window_end = [window_end],
        observations = [length(times)],
        features = [length(feature_cols)],
        nodes = [length(node_cols)],
        rmse = [rmse],
        mae = [mae],
        active_constraints = [active_constraints],
        alpha = [REG_PARAMS.alpha],
        lambda = [REG_PARAMS.lambda]
    )

    per_node = DataFrame(
        node = String.(node_cols),
        rmse = per_node_rmse
    )

    return meta, per_node
end

function _persist_to_duckdb(db_path::AbstractString; beta_df, intercept_df, offsets_df, meta_df, per_node_df)
    db = DuckDB.DB(db_path)
    try
        DuckDB.execute(db, "CREATE SCHEMA IF NOT EXISTS ref;")

        DuckDB.register_data_frame(db, beta_df, "beta_tmp")
        DuckDB.execute(db, "CREATE OR REPLACE TABLE ref.estimated_ptdf AS SELECT * FROM beta_tmp;")
        DuckDB.unregister_data_frame(db, "beta_tmp")

        DuckDB.register_data_frame(db, intercept_df, "intercept_tmp")
        DuckDB.execute(db, "CREATE OR REPLACE TABLE ref.estimated_ptdf_intercepts AS SELECT * FROM intercept_tmp;")
        DuckDB.unregister_data_frame(db, "intercept_tmp")

        DuckDB.register_data_frame(db, offsets_df, "offset_tmp")
        DuckDB.execute(db, "CREATE OR REPLACE TABLE ref.estimated_ptdf_constraint_offsets AS SELECT * FROM offset_tmp;")
        DuckDB.unregister_data_frame(db, "offset_tmp")

        DuckDB.register_data_frame(db, meta_df, "meta_tmp")
        DuckDB.execute(db, "CREATE OR REPLACE TABLE ref.estimated_ptdf_metadata AS SELECT * FROM meta_tmp;")
        DuckDB.unregister_data_frame(db, "meta_tmp")

        DuckDB.register_data_frame(db, per_node_df, "per_node_tmp")
        DuckDB.execute(db, "CREATE OR REPLACE TABLE ref.estimated_ptdf_node_rmse AS SELECT * FROM per_node_tmp;")
        DuckDB.unregister_data_frame(db, "per_node_tmp")
    finally
        close(db)
    end
end

function main()
    @info "Loading training data" db = DB_PATH
    mu_df, target_df = _fetch_training_frames(DB_PATH)

    mats = _prepare_design_matrices(mu_df, target_df)

    @info "Fitting elastic-net" (observations = size(mats.X, 1), features = size(mats.X, 2), nodes = size(mats.Y, 2))
    intercept, betas = _fit_glmnet(mats.X, mats.Y)

    offsets = _rebase_constraints!(betas, mats.feature_cols, mats.node_cols)

    beta_df, intercept_df, offsets_df = _assemble_outputs(betas, intercept, mats.feature_cols, mats.node_cols, offsets)
    residuals = _compute_residual_metrics(mats.X, mats.Y, betas, intercept)

    meta_df, per_node_df = _metadata_frame(mats.times, mats.feature_cols, mats.node_cols,
                                           betas, residuals.rmse, residuals.mae, residuals.per_node_rmse)

    _persist_to_duckdb(DB_PATH;
        beta_df = beta_df,
        intercept_df = intercept_df,
        offsets_df = offsets_df,
        meta_df = meta_df,
        per_node_df = per_node_df
    )

    @info "Saved effective PTDF coefficients" (
        rmse = residuals.rmse,
        mae = residuals.mae,
        active_constraints = meta_df.active_constraints[1]
    )
end

main()
