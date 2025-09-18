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
const EXTRA_REGRESSORS = [:scarcity_adder, :mcpc_regup, :mcpc_rrs, :mcpc_ecrs, :mcpc_nspin]
const EXTRA_REGRESSORS_STR = String.(EXTRA_REGRESSORS)
const REG_PARAMS = (alpha = 0.1, lambda = 0.01)
const HOLDOUT_FRACTION = 0.1
const LAMBDA_GRID = [1e-3, 3e-3, 1e-2, 3e-2, 1e-1]
const MIN_IMPROVEMENT = 0.05

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
            SELECT *
            FROM features.sced_mu
            WHERE sced_ts_utc_minute IN (SELECT DISTINCT sced_ts_utc_minute FROM features.lmp_target)
            ORDER BY sced_ts_utc
        """))

        target_df = DataFrame(DuckDB.execute(db, """
            SELECT
                lt.sced_ts_utc_minute,
                lt.sced_ts_utc,
                lt.settlement_point,
                lt.y_congestion,
                lt.scarcity_adder,
                lt.mcpc_regup,
                lt.mcpc_rrs,
                lt.mcpc_ecrs,
                lt.mcpc_nspin
            FROM features.lmp_target AS lt
            WHERE lt.sced_ts_utc_minute IN (SELECT sced_ts_utc_minute FROM features.sced_mu)
            ORDER BY lt.sced_ts_utc_minute, lt.settlement_point
        """))
    finally
        close(db)
    end

    return mu_df, target_df
end

function _prepare_design_matrices(mu_df::DataFrame, target_df::DataFrame)
    isempty(mu_df) && error("features.sced_mu returned no rows")
    isempty(target_df) && error("features.lmp_target returned no rows")

    sort!(mu_df, :sced_ts_utc_minute)

    scalars = combine(
        groupby(target_df, :sced_ts_utc_minute),
        (reg => _first_or_zero => reg for reg in EXTRA_REGRESSORS)...
    )

    mu_aug = leftjoin(mu_df, scalars, on = :sced_ts_utc_minute, order = :left)

    pivot_input = select(target_df, [:sced_ts_utc_minute, :settlement_point, :y_congestion])
    y_pivot = unstack(pivot_input, :sced_ts_utc_minute, :settlement_point, :y_congestion)

    sort!(mu_aug, :sced_ts_utc_minute)
    sort!(y_pivot, :sced_ts_utc_minute)

    common_times = intersect(mu_aug[!, "sced_ts_utc_minute"], y_pivot[!, "sced_ts_utc_minute"])
    isempty(common_times) && error("No overlapping timestamps between constraints and targets")

    common_set = Set(common_times)
    filter!(row -> row.sced_ts_utc_minute in common_set, mu_aug)
    filter!(row -> row.sced_ts_utc_minute in common_set, y_pivot)

    sort!(mu_aug, :sced_ts_utc_minute)
    sort!(y_pivot, :sced_ts_utc_minute)

    feature_cols = [col for col in names(mu_aug) if col ∉ ("sced_ts_utc", "sced_ts_utc_minute")]
    for col in feature_cols
        mu_aug[!, col] = Float64.(coalesce.(mu_aug[!, col], 0.0))
    end

    node_cols = [col for col in names(y_pivot) if col != "sced_ts_utc_minute"]
    for col in node_cols
        y_pivot[!, col] = Float64.(coalesce.(y_pivot[!, col], 0.0))
    end

    X = Matrix(select(mu_aug, feature_cols))
    Y = Matrix(select(y_pivot, node_cols))

    return (;
        X,
        Y,
        times = copy(mu_aug[!, "sced_ts_utc"]),
        feature_cols = collect(feature_cols),
        node_cols = collect(node_cols),
        mu_aug = mu_aug,
        y_pivot = y_pivot
    )
end

function _fit_glmnet(X::Matrix{Float64}, Y::Matrix{Float64}; lambda::Float64 = REG_PARAMS.lambda)
    fit = glmnet(
        X,
        Y,
        GLMNet.MvNormal();
        alpha = REG_PARAMS.alpha,
        lambda = [lambda],
        standardize = false,
        standardize_response = false
    )

    idx = length(fit.lambda)
    intercept = vec(fit.a0[:, idx])
    betas = fit.betas[:, :, idx]

    return intercept, betas
end

function _rmse_mae(X::Matrix{Float64}, Y::Matrix{Float64}, betas::Matrix{Float64}, intercept::Vector{Float64})
    preds = X * betas
    preds .+= ones(size(preds, 1)) * transpose(intercept)
    residuals = Y .- preds
    rmse = sqrt(mean(residuals .^ 2))
    mae = mean(abs.(residuals))
    return rmse, mae
end

function _train_validation_indices(times::Vector{DateTime}; holdout_fraction::Float64 = HOLDOUT_FRACTION)
    n = length(times)
    n >= 2 || error("Need at least two observations for hold-out evaluation")
    holdout_n = max(1, Int(round(n * holdout_fraction)))
    holdout_n = min(holdout_n, n - 1)
    split_idx = n - holdout_n
    return 1:split_idx, (split_idx + 1):n
end

function _rebase_constraints!(betas::Matrix{Float64}, feature_cols, node_cols)
    constraint_mask = [!(String(col) in EXTRA_REGRESSORS_STR) for col in feature_cols]
    offsets = Dict{String,Float64}()

    for (idx, col) in enumerate(feature_cols)
        constraint_mask[idx] || continue
        offset = mean(betas[idx, :])
        betas[idx, :] .-= offset
        offsets[String(col)] = offset
    end

    return offsets
end

function _assemble_outputs(betas::Matrix{Float64}, intercept::Vector{Float64},
                           feature_cols::Vector{String}, node_cols::Vector{String},
                           offsets::Dict{String,Float64}, feature_stats::Dict{String,Float64})
    feature_names = String.(feature_cols)
    node_names = String.(node_cols)

    rows = Vector{NamedTuple{(:constraint_name, :node, :beta, :feature_type)}}()

    constraint_set = Set(filter(name -> !(name in EXTRA_REGRESSORS_STR), feature_names))

    for (fi, fname) in enumerate(feature_names)
        feature_type = fname in constraint_set ? "constraint" : "regressor"
        for (ni, nname) in enumerate(node_names)
            push!(rows, (constraint_name = fname, node = nname, beta = betas[fi, ni], feature_type = feature_type))
        end
    end

    beta_df = DataFrame(rows)
    beta_df.abs_beta = abs.(beta_df.beta)
    beta_df.avg_abs_mu = [get(feature_stats, row.constraint_name, 0.0) for row in eachrow(beta_df)]
    beta_df.expected_impact = beta_df.abs_beta .* beta_df.avg_abs_mu
    intercept_df = DataFrame(node = node_names, intercept = intercept)

    offsets_df = DataFrame(
        constraint_name = collect(keys(offsets)),
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

    return (; rmse, mae, per_node_rmse)
end

function _metadata_frame(times, feature_cols, node_cols, betas;
                         metrics,
                         holdout_fraction,
                         best_lambda,
                         train_count,
                         val_count)
    window_start = minimum(times)
    window_end = maximum(times)
    constraint_mask = [!(String(col) in EXTRA_REGRESSORS_STR) for col in feature_cols]
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
        rmse = [metrics.rmse],
        mae = [metrics.mae],
        train_rmse = [metrics.train_rmse],
        train_mae = [metrics.train_mae],
        val_rmse = [metrics.val_rmse],
        val_mae = [metrics.val_mae],
        baseline_rmse = [metrics.baseline_rmse],
        baseline_mae = [metrics.baseline_mae],
        improvement_rmse = [metrics.improvement_rmse],
        improvement_mae = [metrics.improvement_mae],
        holdout_fraction = [holdout_fraction],
        train_observations = [train_count],
        val_observations = [val_count],
        active_constraints = [active_constraints],
        alpha = [REG_PARAMS.alpha],
        lambda = [best_lambda]
    )

    per_node = DataFrame(
        node = String.(node_cols),
        rmse = metrics.per_node_rmse
    )

    return meta, per_node
end

function _persist_to_duckdb(db_path::AbstractString; beta_df, intercept_df, offsets_df, meta_df, per_node_df, node_summary_df, fit_metrics_df, publish::Bool)
    db = DuckDB.DB(db_path)
    try
        DuckDB.execute(db, "CREATE SCHEMA IF NOT EXISTS ref;")

        DuckDB.register_data_frame(db, beta_df, "beta_tmp")
        DuckDB.execute(db, "CREATE OR REPLACE TABLE ref.estimated_ptdf_staged AS SELECT * FROM beta_tmp;")
        DuckDB.unregister_data_frame(db, "beta_tmp")

        DuckDB.register_data_frame(db, intercept_df, "intercept_tmp")
        DuckDB.execute(db, "CREATE OR REPLACE TABLE ref.estimated_ptdf_intercepts_staged AS SELECT * FROM intercept_tmp;")
        DuckDB.unregister_data_frame(db, "intercept_tmp")

        DuckDB.register_data_frame(db, offsets_df, "offset_tmp")
        DuckDB.execute(db, "CREATE OR REPLACE TABLE ref.estimated_ptdf_constraint_offsets_staged AS SELECT * FROM offset_tmp;")
        DuckDB.unregister_data_frame(db, "offset_tmp")

        DuckDB.register_data_frame(db, meta_df, "meta_tmp")
        DuckDB.execute(db, "CREATE OR REPLACE TABLE ref.estimated_ptdf_metadata_staged AS SELECT * FROM meta_tmp;")
        DuckDB.unregister_data_frame(db, "meta_tmp")

        DuckDB.register_data_frame(db, per_node_df, "per_node_tmp")
        DuckDB.execute(db, "CREATE OR REPLACE TABLE ref.estimated_ptdf_node_rmse_staged AS SELECT * FROM per_node_tmp;")
        DuckDB.unregister_data_frame(db, "per_node_tmp")

        DuckDB.register_data_frame(db, node_summary_df, "node_summary_tmp")
        DuckDB.execute(db, "CREATE OR REPLACE TABLE ref.estimated_ptdf_node_summary_staged AS SELECT * FROM node_summary_tmp;")
        DuckDB.unregister_data_frame(db, "node_summary_tmp")

        DuckDB.register_data_frame(db, fit_metrics_df, "fit_metrics_tmp")
        DuckDB.execute(db, "CREATE OR REPLACE TABLE ref.estimated_ptdf_fit_metrics AS SELECT * FROM fit_metrics_tmp;")
        DuckDB.unregister_data_frame(db, "fit_metrics_tmp")

        if publish
            for (dest, src) in (
                "ref.estimated_ptdf" => "ref.estimated_ptdf_staged",
                "ref.estimated_ptdf_intercepts" => "ref.estimated_ptdf_intercepts_staged",
                "ref.estimated_ptdf_constraint_offsets" => "ref.estimated_ptdf_constraint_offsets_staged",
                "ref.estimated_ptdf_metadata" => "ref.estimated_ptdf_metadata_staged",
                "ref.estimated_ptdf_node_rmse" => "ref.estimated_ptdf_node_rmse_staged",
                "ref.estimated_ptdf_node_summary" => "ref.estimated_ptdf_node_summary_staged"
            )
                DuckDB.execute(db, "CREATE OR REPLACE TABLE $(dest) AS SELECT * FROM $(src);")
            end
        end
    finally
        close(db)
    end
end

function main()
    @info "Loading training data" db = DB_PATH
    mu_df, target_df = _fetch_training_frames(DB_PATH)

    mats = _prepare_design_matrices(mu_df, target_df)

    @info "Training coverage" start = first(mats.times) stop = last(mats.times) minutes = length(mats.times)
    @info "Fitting elastic-net" (observations = size(mats.X, 1), features = size(mats.X, 2), nodes = size(mats.Y, 2))

    train_idx, val_idx = _train_validation_indices(mats.times; holdout_fraction = HOLDOUT_FRACTION)
    X_train = mats.X[train_idx, :]
    Y_train = mats.Y[train_idx, :]
    X_val = mats.X[val_idx, :]
    Y_val = mats.Y[val_idx, :]

    fit_results = NamedTuple[]
    for λ in LAMBDA_GRID
        intercept, betas = _fit_glmnet(X_train, Y_train; lambda = λ)
        train_rmse, train_mae = _rmse_mae(X_train, Y_train, betas, intercept)
        val_rmse, val_mae = _rmse_mae(X_val, Y_val, betas, intercept)
        push!(fit_results, (lambda = λ,
                            intercept = intercept,
                            betas = betas,
                            train_rmse = train_rmse,
                            train_mae = train_mae,
                            val_rmse = val_rmse,
                            val_mae = val_mae))
    end

    fit_metrics_df = DataFrame(
        lambda = [res.lambda for res in fit_results],
        train_rmse = [res.train_rmse for res in fit_results],
        train_mae = [res.train_mae for res in fit_results],
        val_rmse = [res.val_rmse for res in fit_results],
        val_mae = [res.val_mae for res in fit_results]
    )

    scores = [(res.val_rmse, res.val_mae) for res in fit_results]
    _, best_idx = findmin(scores)
    best_result = fit_results[best_idx]
    intercept = best_result.intercept
    betas = best_result.betas

    feature_stats = Dict{String,Float64}()
    for col in mats.feature_cols
        feature_stats[String(col)] = mean(abs.(mats.mu_aug[!, col]))
    end

    offsets = _rebase_constraints!(betas, mats.feature_cols, mats.node_cols)

    beta_df, intercept_df, offsets_df = _assemble_outputs(betas, intercept, mats.feature_cols, mats.node_cols, offsets, feature_stats)

    full_metrics = _compute_residual_metrics(mats.X, mats.Y, betas, intercept)

    baseline_mean = vec(mean(mats.Y; dims = 1))
    baseline_pred = ones(size(mats.Y, 1)) * transpose(baseline_mean)
    baseline_res = mats.Y .- baseline_pred
    baseline_rmse = sqrt(mean(baseline_res .^ 2))
    baseline_mae = mean(abs.(baseline_res))
    improvement_rmse = baseline_rmse ≈ 0 ? 0.0 : 1 - (full_metrics.rmse / baseline_rmse)
    improvement_mae = baseline_mae ≈ 0 ? 0.0 : 1 - (full_metrics.mae / baseline_mae)

    metrics = (rmse = full_metrics.rmse,
               mae = full_metrics.mae,
               train_rmse = best_result.train_rmse,
               train_mae = best_result.train_mae,
               val_rmse = best_result.val_rmse,
               val_mae = best_result.val_mae,
               baseline_rmse = baseline_rmse,
               baseline_mae = baseline_mae,
               improvement_rmse = improvement_rmse,
               improvement_mae = improvement_mae,
               per_node_rmse = full_metrics.per_node_rmse)

    meta_df, per_node_df = _metadata_frame(mats.times, mats.feature_cols, mats.node_cols, betas;
                                           metrics = metrics,
                                           holdout_fraction = HOLDOUT_FRACTION,
                                           best_lambda = best_result.lambda,
                                           train_count = length(train_idx),
                                           val_count = length(val_idx))

    constraint_df = filter(:feature_type => ==("constraint"), beta_df)
    isempty(constraint_df) && (constraint_df = beta_df)
    node_summary_df = combine(groupby(constraint_df, :node),
        :abs_beta => maximum => :max_abs_beta,
        :expected_impact => maximum => :max_expected_impact,
        :expected_impact => sum => :total_expected_impact,
        nrow => :constraint_count)
    top_constraint_df = combine(groupby(constraint_df, :node)) do sdf
        sorted = sort(sdf, :expected_impact, rev = true)
        top = sorted[1, :]
        (node = top.node, top_constraint = top.constraint_name, top_expected_impact = top.expected_impact)
    end
    node_summary_df = leftjoin(node_summary_df, top_constraint_df, on = :node)

    publish = metrics.improvement_rmse >= MIN_IMPROVEMENT

    _persist_to_duckdb(DB_PATH;
        beta_df = beta_df,
        intercept_df = intercept_df,
        offsets_df = offsets_df,
        meta_df = meta_df,
        per_node_df = per_node_df,
        node_summary_df = node_summary_df,
        fit_metrics_df = fit_metrics_df,
        publish = publish)

    @info "Saved effective PTDF coefficients" (
        rmse = metrics.rmse,
        mae = metrics.mae,
        baseline_rmse = metrics.baseline_rmse,
        improvement_rmse = metrics.improvement_rmse,
        val_rmse = metrics.val_rmse,
        active_constraints = meta_df.active_constraints[1],
        best_lambda = best_result.lambda,
        published = publish
    )

    if !publish
        @warn "Model not published; improvement below threshold" improvement_rmse = metrics.improvement_rmse baseline_rmse = metrics.baseline_rmse rmse = metrics.rmse
    end
end

main()
