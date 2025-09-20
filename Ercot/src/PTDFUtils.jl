module PTDFUtils

using DataFrames
using DuckDB
using Dates
using Logging
using Statistics

import ..EventAlgebra: EventGraph, EventNode, add_event!, has_event, upsert_event!
import ..AssimilationModel: evaluate_event_priors, ensemble_event_priors
import ..MarketScoring
import ..MarketScoring: initialize_market, state_prices
import ..AssimilationRunner: analyze_and_forecast!

const EXTRA_REGRESSORS = [:scarcity_adder, :mcpc_regup, :mcpc_rrs, :mcpc_ecrs, :mcpc_nspin]
const EXTRA_REGRESSORS_STR = String.(EXTRA_REGRESSORS)

const BASIS_HUB = get(ENV, "SCENARIO_BASIS_HUB", "HB_HOUSTON")
const BASIS_POS_THRESHOLD = parse(Float64, get(ENV, "SCENARIO_BASIS_THRESHOLD", "15.0"))
const BASIS_NEG_THRESHOLD = parse(Float64, get(ENV, "SCENARIO_BASIS_THRESHOLD_NEG", "-15.0"))
const STATS_LOOKBACK = Day(parse(Int, get(ENV, "SCENARIO_STATS_LOOKBACK_DAYS", "14")))
const SCARCITY_PRC_THRESHOLD = parse(Float64, get(ENV, "SCENARIO_SCARCITY_PRC_THRESHOLD", "2000.0"))
const METRIC_EPS = 1e-6

sanitize_symbol(label::AbstractString) = Symbol(replace(label, r"[^A-Za-z0-9_]" => "_"))
_logistic(x; center = 30.0, scale = 10.0) = 1 / (1 + exp(-(x - center) / scale))

const _DOUBLE_QUOTE = string(Char(34))
const _DOUBLE_QUOTE_ESC = string(Char(34), Char(34))

function _empirical_quantile(values::Vector{Float64}, q::Float64)
    isempty(values) && error("Cannot compute quantile of empty collection")
    q < 0 || q > 1 && error("Quantile must be in [0,1]")
    sorted = sort(values)
    n = length(sorted)
    q == 1 && return sorted[end]
    q == 0 && return sorted[1]
    pos = (n - 1) * q + 1
    lower = floor(Int, pos)
    upper = ceil(Int, pos)
    lower < 1 && (lower = 1)
    upper > n && (upper = n)
    if lower == upper
        return sorted[lower]
    else
        weight = pos - lower
        return sorted[lower] + weight * (sorted[upper] - sorted[lower])
    end
end

function _parse_timestamp(ts_raw)
    ts_raw isa Dates.AbstractDateTime && return ts_raw
    ts_raw isa AbstractString || error("Unsupported timestamp type $(typeof(ts_raw))")
    try
        return DateTime(ts_raw)
    catch
        endswith(ts_raw, "Z") && return DateTime(ts_raw[1:end-1])
        error("Unable to parse timestamp string $(ts_raw)")
    end
end

function _load_logistic_params(db_path::AbstractString)
    params = Dict{Tuple{String,String},Tuple{Float64,Float64}}()
    db = DuckDB.DB(db_path)
    try
        df = DataFrame(DuckDB.execute(db, """
            SELECT event_kind, scope_key, center, scale
            FROM ref.logistic_map_params
        """))
        for row in eachrow(df)
            key = (String(row.event_kind), String(row.scope_key))
            params[key] = (Float64(row.center), Float64(row.scale))
        end
    catch err
        @debug "Failed loading logistic params" exception=(err, catch_backtrace())
    finally
        close(db)
    end
    return params
end

function _load_latest_fact_row(db_path::AbstractString)
    db = DuckDB.DB(db_path)
    try
        df = DataFrame(DuckDB.execute(db, """
            WITH latest AS (SELECT max(sced_ts_utc_minute) AS ts FROM mart.fact_rt_5min)
            SELECT *
            FROM mart.fact_rt_5min f
            JOIN latest l ON f.sced_ts_utc_minute = l.ts
            LIMIT 1
        """))
        isempty(df) && error("mart.fact_rt_5min has no rows")
        return df[1, :]
    finally
        close(db)
    end
end

function _load_actual_node_data(db_path::AbstractString, minute::DateTime)
    db = DuckDB.DB(db_path)
    price_map = Dict{String,Float64}()
    congestion_map = Dict{String,Float64}()
    try
        df = DataFrame(DuckDB.execute(db, """
            SELECT settlement_point,
                   rt_lmp,
                   rt_lmp - COALESCE(system_lambda, 0) AS congestion
            FROM mart.fact_rt_5min
            WHERE sced_ts_utc_minute = ?
        """, [minute]))
        for row in eachrow(df)
            node = String(row.settlement_point)
            price_map[node] = Float64(row.rt_lmp)
            congestion_map[node] = Float64(row.congestion)
        end
    finally
        close(db)
    end
    return price_map, congestion_map
end

function _float_default(value, fallback)
    if value === nothing || value === missing || isnan(Float64(value))
        return fallback
    end
    return Float64(value)
end

function _retrieve_params(params::Dict{Tuple{String,String},Tuple{Float64,Float64}}, kind::String, default_center::Float64, default_scale::Float64, keys::AbstractVector{<:AbstractString})
    for key in keys
        val = (kind, String(key))
        if haskey(params, val)
            return params[val]
        end
    end
    return (default_center, default_scale)
end

function _insert_prior_event!(graph::EventGraph, priors::Dict{Symbol,Float64}, symbol::Symbol, probability::Float64;
                              description::AbstractString="", tags::AbstractVector{Symbol}=Symbol[])
    event = EventNode(symbol; description=description, scope=(prior = probability, tags = tags))
    if has_event(graph, symbol)
        upsert_event!(graph, event)
    else
        add_event!(graph, event)
    end
    priors[symbol] = probability
end

clamp01(x::Float64) = clamp(x, 0.0, 1.0)
safe_prob(x::Float64) = clamp(x, METRIC_EPS, 1 - METRIC_EPS)

function _fetch_stat_thresholds(db_path::AbstractString; lookback::Period=STATS_LOOKBACK)
    start_ts = now(Dates.UTC) - lookback
    db = DuckDB.DB(db_path)
    try
        df = DataFrame(DuckDB.execute(db, """
            WITH recent AS (
                SELECT *
                FROM mart.fact_rt_5min
                WHERE sced_ts_utc >= ?
            )
            SELECT
                quantile_cont(system_load_forecast_mw, 0.9) AS load_hi,
                quantile_cont(system_load_forecast_mw, 0.5) AS load_med,
                quantile_cont(wind_system_mw, 0.1) AS wind_low,
                quantile_cont(wind_system_mw, 0.5) AS wind_med,
                quantile_cont(solar_system_mw, 0.1) AS solar_low,
                quantile_cont(solar_system_mw, 0.5) AS solar_med,
                quantile_cont(cap_rrs_total / NULLIF(system_load_forecast_mw,0), 0.1) AS cap_ratio_low,
                quantile_cont(cap_rrs_total / NULLIF(system_load_forecast_mw,0), 0.5) AS cap_ratio_med
            FROM recent
        """, [start_ts]))
        if isempty(df)
            return Dict{Symbol,Float64}()
        end
        row = df[1, :]
        return Dict{Symbol,Float64}(
            :load_hi => _float_default(get(row, :load_hi, missing), 45_000.0),
            :load_med => _float_default(get(row, :load_med, missing), 30_000.0),
            :wind_low => _float_default(get(row, :wind_low, missing), 5_000.0),
            :wind_med => _float_default(get(row, :wind_med, missing), 10_000.0),
            :solar_low => _float_default(get(row, :solar_low, missing), 2_000.0),
            :solar_med => _float_default(get(row, :solar_med, missing), 4_000.0),
            :cap_ratio_low => _float_default(get(row, :cap_ratio_low, missing), 0.05),
            :cap_ratio_med => _float_default(get(row, :cap_ratio_med, missing), 0.10)
        )
    finally
        close(db)
    end
end

function _node_bucket_probabilities(node::String, predicted::Float64, center::Float64, scale::Float64, actual_price::Union{Missing,Float64})
    center0 = center + (0.0 - 25.0)
    center100 = center + (100.0 - 25.0)
    p_gt0 = clamp01(_logistic(predicted; center=center0, scale=scale))
    p_gt25 = clamp01(_logistic(predicted; center=center, scale=scale))
    p_gt100 = clamp01(_logistic(predicted; center=center100, scale=scale))
    p_le0 = clamp01(1 - p_gt0)
    p_0_25 = clamp01(p_gt0 - p_gt25)
    p_25_100 = clamp01(p_gt25 - p_gt100)
    p_ge100 = clamp01(p_gt100)
    total = p_le0 + p_0_25 + p_25_100 + p_ge100
    if total > 0
        p_le0 /= total
        p_0_25 /= total
        p_25_100 /= total
        p_ge100 /= total
    end
    bucket_events = Dict{Symbol,Float64}(
        sanitize_symbol("node_$(node)_bucket_le0") => p_le0,
        sanitize_symbol("node_$(node)_bucket_0_25") => p_0_25,
        sanitize_symbol("node_$(node)_bucket_25_100") => p_25_100,
        sanitize_symbol("node_$(node)_bucket_ge100") => p_ge100,
    )
    outcomes = Dict{Symbol,Int}()
    if !(actual_price === missing) && isfinite(Float64(actual_price))
        price_val = Float64(actual_price)
        selected = if price_val <= 0
            sanitize_symbol("node_$(node)_bucket_le0")
        elseif price_val < 25
            sanitize_symbol("node_$(node)_bucket_0_25")
        elseif price_val < 100
            sanitize_symbol("node_$(node)_bucket_25_100")
        else
            sanitize_symbol("node_$(node)_bucket_ge100")
        end
        for sym in keys(bucket_events)
            outcomes[sym] = sym == selected ? 1 : 0
        end
    end
    return bucket_events, outcomes
end

function _node_basis_probabilities(node::String, predicted::Float64, hub_predicted::Float64,
                                   actual_price::Union{Missing,Float64}, hub_actual::Union{Missing,Float64},
                                   logistic_params::Dict{Tuple{String,String},Tuple{Float64,Float64}})
    basis_pred = predicted - hub_predicted
    key = string(node, "|", BASIS_HUB)
    (pos_center, pos_scale) = _retrieve_params(logistic_params, "basis_spike_pos", BASIS_POS_THRESHOLD, max(BASIS_POS_THRESHOLD * 0.25, 5.0), [key, "global"])
    (neg_center_raw, neg_scale) = _retrieve_params(logistic_params, "basis_spike_neg", abs(BASIS_NEG_THRESHOLD), max(abs(BASIS_NEG_THRESHOLD) * 0.25, 5.0), [key, "global"])
    pos_prob = clamp01(_logistic(basis_pred; center=pos_center, scale=pos_scale))
    neg_prob = clamp01(_logistic(-basis_pred; center=neg_center_raw, scale=neg_scale))
    basis_events = Dict{Symbol,Float64}(
        sanitize_symbol("basis_spike_pos_$(node)_vs_$(BASIS_HUB)") => pos_prob,
        sanitize_symbol("basis_spike_neg_$(node)_vs_$(BASIS_HUB)") => neg_prob,
    )
    outcomes = Dict{Symbol,Int}()
    if !(actual_price === missing || hub_actual === missing)
        actual_basis = Float64(actual_price) - Float64(hub_actual)
        pos_sym = sanitize_symbol("basis_spike_pos_$(node)_vs_$(BASIS_HUB)")
        neg_sym = sanitize_symbol("basis_spike_neg_$(node)_vs_$(BASIS_HUB)")
        outcomes[pos_sym] = actual_basis >= pos_center ? 1 : 0
        outcomes[neg_sym] = actual_basis <= -neg_center_raw ? 1 : 0
    end
    return basis_events, outcomes
end

function expand_event_vocabulary!(graph::EventGraph,
                                  priors::Dict{Symbol,Float64},
                                  predictions::Dict{String,Float64},
                                  contributions;
                                  logistic_params::Dict{Tuple{String,String},Tuple{Float64,Float64}}=Dict(),
                                  fact_row=nothing,
                                  thresholds::Union{Nothing,Dict{Symbol,Float64}}=nothing,
                                  price_map::Union{Nothing,Dict{String,Float64}}=nothing,
                                  congestion_map::Union{Nothing,Dict{String,Float64}}=nothing,
                                  hub::String=BASIS_HUB,
                                  feature_values::Union{Nothing,Dict{String,Float64}}=nothing)
    threshold_map = thresholds === nothing ? Dict{Symbol,Float64}() : thresholds
    event_outcomes = Dict{Symbol,Int}()

    hub_predicted = get(predictions, hub, NaN)
    hub_actual_val = price_map === nothing ? NaN : get(price_map, hub, NaN)
    hub_actual = isfinite(hub_actual_val) ? hub_actual_val : missing

    for (node, value) in predictions
        node_symbol = sanitize_symbol("node_$(node)_gt25")
        if congestion_map !== nothing
            actual_cong = get(congestion_map, node, nothing)
            if actual_cong !== nothing
                event_outcomes[node_symbol] = Float64(actual_cong) > 25.0 ? 1 : 0
            end
        end

        if feature_values !== nothing
            contribs = get(contributions, node, NamedTuple[])
            for contrib in contribs
                constraint_symbol = sanitize_symbol("constraint_$(contrib.constraint_name)_pos")
                actual_mu = get(feature_values, contrib.constraint_name, 0.0)
                event_outcomes[constraint_symbol] = actual_mu > 0 ? 1 : 0
            end
        end

        (node_center, node_scale) = _retrieve_params(logistic_params, "node_gt25", 25.0, 8.0, [node, "global"])
        actual_price = price_map === nothing ? missing : get(price_map, node, missing)
        bucket_probs, bucket_outcomes = _node_bucket_probabilities(node, value, node_center, node_scale, actual_price)
        for (sym, prob) in bucket_probs
            _insert_prior_event!(graph, priors, sym, prob; description = "Probability node $(node) settles in bucket", tags = [:bucket, :node])
        end
        for (sym, outcome) in bucket_outcomes
            event_outcomes[sym] = outcome
        end

        if isfinite(hub_predicted)
            basis_probs, basis_outcomes = _node_basis_probabilities(node, value, hub_predicted, actual_price, hub_actual, logistic_params)
            for (sym, prob) in basis_probs
                _insert_prior_event!(graph, priors, sym, prob; description = "Basis spike for $(node) vs $(hub)", tags = [:basis, :node])
            end
            for (sym, outcome) in basis_outcomes
                event_outcomes[sym] = outcome
            end
        end
    end

    if fact_row !== nothing
        load_value = Float64(coalesce(get(fact_row, :system_load_forecast_mw, missing), 0.0))
        wind_value = Float64(coalesce(get(fact_row, :wind_system_mw, missing), 0.0))
        solar_value = Float64(coalesce(get(fact_row, :solar_system_mw, missing), 0.0))
        cap_rrs_total = Float64(coalesce(get(fact_row, :cap_rrs_total, missing), 0.0))

        load_hi_center = get(threshold_map, :load_hi, load_value)
        load_med = get(threshold_map, :load_med, max(load_value, 1.0))
        load_scale_default = max(abs(load_hi_center - load_med) / 2, max(load_hi_center * 0.05, 100.0))
        (load_center, load_scale) = _retrieve_params(logistic_params, "load_hi", load_hi_center, load_scale_default, ["global"])
        load_prob = _logistic(load_value; center = load_center, scale = load_scale)
        _insert_prior_event!(graph, priors, :load_hi, clamp01(load_prob); description = "System load high regime", tags = [:load, :regime])
        event_outcomes[:load_hi] = load_value >= load_center ? 1 : 0

        wind_low = get(threshold_map, :wind_low, wind_value)
        wind_med = get(threshold_map, :wind_med, max(wind_value, 1.0))
        wind_scale_default = max(abs(wind_med - wind_low) / 2, max(abs(wind_low) * 0.1, 50.0))
        (wind_center, wind_scale) = _retrieve_params(logistic_params, "wind_down", wind_low, wind_scale_default, ["global"])
        wind_prob = _logistic(wind_center - wind_value; center = 0.0, scale = wind_scale)
        _insert_prior_event!(graph, priors, :wind_down, clamp01(wind_prob); description = "Wind generation down regime", tags = [:wind, :regime])
        event_outcomes[:wind_down] = wind_value <= wind_center ? 1 : 0

        solar_low = get(threshold_map, :solar_low, solar_value)
        solar_med = get(threshold_map, :solar_med, max(solar_value, 1.0))
        solar_scale_default = max(abs(solar_med - solar_low) / 2, max(abs(solar_low) * 0.1, 20.0))
        (solar_center, solar_scale) = _retrieve_params(logistic_params, "solar_down", solar_low, solar_scale_default, ["global"])
        solar_prob = _logistic(solar_center - solar_value; center = 0.0, scale = solar_scale)
        _insert_prior_event!(graph, priors, :solar_down, clamp01(solar_prob); description = "Solar generation down regime", tags = [:solar, :regime])
        event_outcomes[:solar_down] = solar_value <= solar_center ? 1 : 0

        load_denom = max(load_value, 1.0)
        cap_ratio = load_denom == 0 ? 0.0 : cap_rrs_total / load_denom
        cap_low = get(threshold_map, :cap_ratio_low, 0.05)
        cap_med = get(threshold_map, :cap_ratio_med, 0.1)
        cap_scale_default = max(abs(cap_med - cap_low) / 2, 0.01)
        (cap_center, cap_scale) = _retrieve_params(logistic_params, "tightness_low_cap", cap_low, cap_scale_default, ["global"])
        tight_prob = _logistic(cap_center - cap_ratio; center = 0.0, scale = cap_scale)
        _insert_prior_event!(graph, priors, :tightness_low_cap, clamp01(tight_prob); description = "Reserve tightness low", tags = [:tightness, :regime])
        event_outcomes[:tightness_low_cap] = cap_ratio <= cap_center ? 1 : 0
    end

    return event_outcomes
end

function load_latest_snapshot(db_path::AbstractString)
    db = DuckDB.DB(db_path)
    try
        mu_row = DataFrame(DuckDB.execute(db, """
            SELECT mu.*, lt.scarcity_adder, lt.mcpc_regup, lt.mcpc_rrs, lt.mcpc_ecrs, lt.mcpc_nspin
            FROM features.sced_mu AS mu
            INNER JOIN features.lmp_target AS lt USING (sced_ts_utc_minute)
            ORDER BY mu.sced_ts_utc DESC
            LIMIT 1
        """))
        isempty(mu_row) && error("features.sced_mu has no rows")
        beta_df = DataFrame(DuckDB.execute(db, "SELECT * FROM ref.estimated_ptdf"))
        intercept_df = DataFrame(DuckDB.execute(db, "SELECT * FROM ref.estimated_ptdf_intercepts"))
        metadata = DataFrame(DuckDB.execute(db, "SELECT * FROM ref.estimated_ptdf_metadata ORDER BY run_ts DESC LIMIT 1"))
        isempty(metadata) && error("ref.estimated_ptdf_metadata is empty")
        return mu_row, beta_df, intercept_df, metadata[1, :]
    finally
        close(db)
    end
end

function load_metadata(db_path::AbstractString)
    db = DuckDB.DB(db_path)
    try
        df = DataFrame(DuckDB.execute(db, "SELECT * FROM ref.estimated_ptdf_metadata ORDER BY run_ts DESC LIMIT 1"))
        isempty(df) && return nothing
        return df[1, :]
    finally
        close(db)
    end
end

model_is_fresh(metadata; max_age::Period = Hour(2)) = (now(Dates.UTC) - metadata.run_ts) <= max_age
model_improvement(metadata) = get(metadata, :improvement_rmse, missing)

function build_feature_vector(mu_row::DataFrame)
    feature_values = Dict{String,Float64}()
    for name in names(mu_row)
        name in ("sced_ts_utc", "sced_ts_utc_minute") && continue
        feature_values[name] = Float64(coalesce(mu_row[1, name], 0.0))
    end
    return feature_values
end

function predict_congestion(beta_df::DataFrame, intercept_df::DataFrame, feature_values::Dict{String,Float64};
                            nodes_filter::Union{Nothing,Vector{String}}=nothing,
                            nodes_limit::Union{Nothing,Int}=nothing,
                            top_constraints::Int=3)
    intercept_map = Dict(String(row.node) => Float64(row.intercept) for row in eachrow(intercept_df))
    nodes = unique(beta_df.node)
    if nodes_filter !== nothing
        filter_set = Set(nodes_filter)
        nodes = filter(n -> n in filter_set, nodes)
    end
    if nodes_limit !== nothing
        nodes = nodes[1:min(nodes_limit, length(nodes))]
    end
    predictions = Dict{String,Float64}()
    contribution_map = Dict{String,Vector{NamedTuple{(:constraint_name,:beta,:mu_value,:contribution,:feature_type),Tuple{String,Float64,Float64,Float64,String}}}}()
    for node in nodes
        sdf = filter(:node => ==(node), beta_df)
        value = get(intercept_map, String(node), 0.0)
        node_contribs = NamedTuple{(:constraint_name,:beta,:mu_value,:contribution,:feature_type),Tuple{String,Float64,Float64,Float64,String}}[]
        for row in eachrow(sdf)
            fv = get(feature_values, row.constraint_name, 0.0)
            contrib = row.beta * fv
            value += contrib
            push!(node_contribs, (constraint_name = row.constraint_name,
                                  beta = row.beta,
                                  mu_value = fv,
                                  contribution = contrib,
                                  feature_type = row.feature_type))
        end
        predictions[String(node)] = value
        sorted = sort(node_contribs; by = x -> abs(x.contribution), rev = true)
        contribution_map[String(node)] = sorted[1:min(top_constraints, length(sorted))]
    end
    return predictions, contribution_map
end

function build_event_graph(predictions::Dict{String,Float64}, contributions;
                          logistic_params::Dict{Tuple{String,String},Tuple{Float64,Float64}}=Dict())
    graph = EventGraph()
    base_events = Dict{Symbol,Float64}()
    for (node, value) in predictions
        (node_center, node_scale) = _retrieve_params(logistic_params, "node_gt25", 25.0, 8.0, [node, "global"])
        prob = _logistic(value; center=node_center, scale=node_scale)
        event_symbol = sanitize_symbol("node_$(node)_gt25")
        add_event!(graph, EventNode(event_symbol; description = "Predicted congestion price > 25 for $(node)", scope = (prior = prob, tags = [:ptdf, :node])))
        base_events[event_symbol] = prob
        contribs = get(contributions, node, NamedTuple[])
        parent_symbols = Symbol[]
        for contrib in contribs
            constraint_symbol = sanitize_symbol("constraint_$(contrib.constraint_name)_pos")
            (contrib_center, contrib_scale) = _retrieve_params(logistic_params, "contrib_pos", 0.0, 10.0, ["global"])
            mu_prob = _logistic(contrib.contribution; center=contrib_center, scale=contrib_scale)
            has_event(graph, constraint_symbol) || add_event!(graph, EventNode(constraint_symbol; description = "Constraint $(contrib.constraint_name) positive contribution", scope = (prior = mu_prob, tags = [:constraint, :ptdf])))
            base_events[constraint_symbol] = mu_prob
            push!(parent_symbols, constraint_symbol)
        end
        if !isempty(parent_symbols)
            share_symbol = sanitize_symbol("node_$(node)_driver_share")
            add_event!(graph, EventNode(share_symbol;
                                        parents = parent_symbols,
                                        scope = (aggregator = :weighted_sum,
                                                 weights = ones(length(parent_symbols)) ./ length(parent_symbols),
                                                 tags = [:aggregation, :ptdf])))
        end
    end
    priors = evaluate_event_priors(graph, base_events)
    return graph, priors
end

function upsert_assimilation_events!(graph::EventGraph,
                                     priors::Dict{Symbol,Float64},
                                     model,
                                     Xa,
                                     config;
                                     prc_threshold::Float64=SCARCITY_PRC_THRESHOLD)
    added = Symbol[]
    scarcity_symbol = :scarcity_hi
    if !has_event(graph, scarcity_symbol)
        add_event!(graph, EventNode(scarcity_symbol;
                                    scope=(variable=:prc, relation=:gt, threshold=prc_threshold, tags=[:assim, :scarcity])))
    end
    push!(added, scarcity_symbol)

    for (label, _) in config.mu_pairs
        label_str = String(label)
        event_symbol = sanitize_symbol("binds_$(label_str)")
        if !has_event(graph, event_symbol)
            add_event!(graph, EventNode(event_symbol;
                                        scope=(variable=label, relation=:gt, threshold=0.0, tags=[:assim, :constraint])))
        end
        push!(added, event_symbol)
    end

    assimilation_priors = ensemble_event_priors(model, Xa, graph)
    new_entries = Dict{Symbol,Float64}()
    for sym in added
        if haskey(assimilation_priors, sym)
            priors[sym] = assimilation_priors[sym]
            new_entries[sym] = assimilation_priors[sym]
        end
    end
    return new_entries
end

function scenario_summary(db_path::AbstractString; nodes_filter::Union{Nothing,Vector{String}}=nothing,
                          nodes_limit::Union{Nothing,Int}=nothing, top_constraints::Int=3, b::Float64=5.0)
    mu_row, beta_df, intercept_df, metadata = load_latest_snapshot(db_path)
    feature_values = build_feature_vector(mu_row)
    predictions, contribution_map = predict_congestion(beta_df, intercept_df, feature_values;
                                                       nodes_filter=nodes_filter,
                                                       nodes_limit=nodes_limit,
                                                       top_constraints=top_constraints)
    logistic_params = _load_logistic_params(db_path)
    graph, priors = build_event_graph(predictions, contribution_map; logistic_params=logistic_params)

    fact_row = _load_latest_fact_row(db_path)
    thresholds = _fetch_stat_thresholds(db_path)
    load_value = Float64(coalesce(get(fact_row, :system_load_forecast_mw, missing), 0.0))
    wind_value = Float64(coalesce(get(fact_row, :wind_system_mw, missing), 0.0))
    solar_value = Float64(coalesce(get(fact_row, :solar_system_mw, missing), 0.0))
    cap_rrs_total = Float64(coalesce(get(fact_row, :cap_rrs_total, missing), 0.0))
    fact_prc = Float64(coalesce(get(fact_row, :prc, missing), 0.0))

    minute_ts = DateTime(mu_row[1, "sced_ts_utc_minute"])
    price_map, congestion_map = _load_actual_node_data(db_path, minute_ts)
    hub_predicted = get(predictions, BASIS_HUB, NaN)
    hub_actual = get(price_map, BASIS_HUB, NaN)

    event_outcomes = expand_event_vocabulary!(graph, priors, predictions, contribution_map;
                                              logistic_params=logistic_params,
                                              fact_row=fact_row,
                                              thresholds=thresholds,
                                              price_map=price_map,
                                              congestion_map=congestion_map,
                                              hub=BASIS_HUB,
                                              feature_values=feature_values)

    assimilation_meta = nothing
    mu_lookup = Dict{Symbol,String}()
    try
        model, Xa, config, meta = analyze_and_forecast!()
        new_entries = upsert_assimilation_events!(graph, priors, model, Xa, config; prc_threshold=SCARCITY_PRC_THRESHOLD)
        for pair in config.mu_pairs
            mu_lookup[pair[1]] = pair[2]
        end
        for (sym, _) in new_entries
            str_sym = String(sym)
            if sym == :scarcity_hi
                event_outcomes[sym] = fact_prc > SCARCITY_PRC_THRESHOLD ? 1 : 0
            elseif startswith(str_sym, "binds_")
                sanitized_label = Symbol(str_sym[length("binds_")+1:end])
                raw = get(mu_lookup, sanitized_label, nothing)
                raw === nothing && continue
                mu_value = get(feature_values, raw, 0.0)
                event_outcomes[sym] = mu_value > 0 ? 1 : 0
            end
        end
        assimilation_meta = Dict{Symbol,Any}(
            :ensemble_size => size(Xa, 2),
            :labels => String.(config.labels),
            :mu_columns => [pair[2] for pair in config.mu_pairs],
            :events => Dict(String(k) => v for (k, v) in new_entries),
            :fact_timestamp => get(meta, :fact_timestamp, missing),
            :fact_minute => get(meta, :fact_minute, missing),
            :mu_timestamp => get(meta, :mu_timestamp, missing),
            :mu_minute => get(meta, :mu_minute, missing)
        )
    catch err
        @warn "Assimilation tick failed" exception=(err, catch_backtrace())
    end
    market = initialize_market(priors; b = b)
    prices = state_prices(market)
    metric_map = MarketScoring.calibration_metrics(prices, event_outcomes; clamp_eps = METRIC_EPS)
    calibration_metrics_dict = Dict(String(k) => v for (k, v) in metric_map)
    node_payload = Dict{String,Any}()
    for (node, value) in predictions
        drivers = [Dict(
            :constraint_name => contrib.constraint_name,
            :beta => contrib.beta,
            :mu_value => contrib.mu_value,
            :contribution => contrib.contribution,
            :feature_type => contrib.feature_type
        ) for contrib in get(contribution_map, node, NamedTuple[])]
        node_payload[node] = Dict(
            :predicted_price => value,
            :drivers => drivers
        )
    end
    metadata_dict = Dict{Symbol,Any}()
    metadata_dict[:run_ts] = string(metadata.run_ts)
    metadata_dict[:improvement_rmse] = get(metadata, :improvement_rmse, missing)
    metadata_dict[:baseline_rmse] = get(metadata, :baseline_rmse, missing)
    if assimilation_meta !== nothing
        metadata_dict[:assimilation] = assimilation_meta
    end

    thresholds_serialized = Dict{String,Float64}()
    for (k, v) in thresholds
        thresholds_serialized[String(k)] = v
    end

    event_outcome_strings = Dict(String(k) => v for (k, v) in event_outcomes)

    return Dict(
        :timestamp => string(mu_row[1, "sced_ts_utc"]),
        :liquidity => b,
        :metadata => metadata_dict,
        :nodes => node_payload,
        :event_prices => Dict(String(k) => v for (k, v) in prices),
        :event_outcomes => event_outcome_strings,
        :thresholds => thresholds_serialized,
        :calibration_metrics => calibration_metrics_dict
    )
end

function persist_event_prices!(db_path::AbstractString, summary::Dict{Symbol,<:Any}; source::AbstractString = "ptdf_scenario")
    haskey(summary, :event_prices) || error("summary missing :event_prices")
    event_prices = summary[:event_prices]
    isempty(event_prices) && return 0

    ts = _parse_timestamp(summary[:timestamp])
    liquidity = get(summary, :liquidity, 5.0)

    events = String.(collect(keys(event_prices)))
    prices = Float64.(collect(values(event_prices)))
    rows = DataFrame(
        sced_ts_utc_minute = fill(ts, length(events)),
        event_id = events,
        price = prices,
        liquidity = fill(Float64(liquidity), length(events)),
        source = fill(source, length(events))
    )

    outcomes = get(summary, :event_outcomes, nothing)
    outcome_map = outcomes === nothing ? Dict{String,Int}() : Dict(outcomes)

    db = DuckDB.DB(db_path)
    try
        DuckDB.execute(db, "CREATE SCHEMA IF NOT EXISTS mart")
        DuckDB.execute(db, """
            CREATE TABLE IF NOT EXISTS mart.event_price_history (
                sced_ts_utc_minute TIMESTAMPTZ,
                event_id TEXT,
                price DOUBLE,
                liquidity DOUBLE,
                source TEXT,
                PRIMARY KEY (sced_ts_utc_minute, event_id, source)
            )
        """)

        DuckDB.register_data_frame(db, rows, "event_prices_tmp")
        try
            DuckDB.execute(db, """
                INSERT OR REPLACE INTO mart.event_price_history
                SELECT * FROM event_prices_tmp;
            """)
        finally
            DuckDB.unregister_data_frame(db, "event_prices_tmp")
        end

        metrics_lookup = get(summary, :calibration_metrics, nothing)
        metrics_dict = if metrics_lookup === nothing
            price_symbol_map = Dict(Symbol(k) => Float64(v) for (k, v) in event_prices)
            outcome_symbol_map = Dict(Symbol(k) => v for (k, v) in outcome_map)
            computed = MarketScoring.calibration_metrics(price_symbol_map, outcome_symbol_map; clamp_eps = METRIC_EPS)
            Dict(String(k) => v for (k, v) in computed)
        else
            Dict(metrics_lookup)
        end

        if !isempty(outcome_map)
            DuckDB.execute(db, """
                CREATE TABLE IF NOT EXISTS mart.event_calibration_history (
                    sced_ts_utc_minute TIMESTAMPTZ,
                    event_id TEXT,
                    price DOUBLE,
                    outcome INTEGER,
                    brier DOUBLE,
                    log_score DOUBLE,
                    source TEXT,
                    PRIMARY KEY (sced_ts_utc_minute, event_id, source)
                )
            """)

            metrics_rows = NamedTuple{(:sced_ts_utc_minute,:event_id,:price,:outcome,:brier,:log_score,:source),Tuple{DateTime,String,Float64,Int,Float64,Float64,String}}[]
            for (idx, event_id) in enumerate(events)
                outcome = get(outcome_map, event_id, nothing)
                outcome === nothing && continue
                metrics = get(metrics_dict, event_id, nothing)
                if metrics === nothing
                    price_val = prices[idx]
                    brier = MarketScoring.brier_score(price_val, outcome; clamp_eps = METRIC_EPS)
                    log_score = MarketScoring.log_score(price_val, outcome; clamp_eps = METRIC_EPS)
                else
                    brier = Float64(metrics.brier)
                    log_score = Float64(metrics.log_score)
                end
                push!(metrics_rows, (ts, event_id, prices[idx], Int(outcome), brier, log_score, source))
            end

            if !isempty(metrics_rows)
                metrics_df = DataFrame(metrics_rows)
                DuckDB.register_data_frame(db, metrics_df, "event_metrics_tmp")
                try
                    DuckDB.execute(db, """
                        INSERT OR REPLACE INTO mart.event_calibration_history
                        SELECT * FROM event_metrics_tmp;
                    """)
                finally
                    DuckDB.unregister_data_frame(db, "event_metrics_tmp")
                end
            end
        end
    finally
        close(db)
    end

    return length(event_prices)
end

function persist_risk_log!(db_path::AbstractString, entries::Vector{<:NamedTuple})
    isempty(entries) && return 0
    df = DataFrame(entries)
    desired_order = String[
        "sced_ts_utc", "node", "hub", "direction", "trade_basis", "quantity",
        "signed_quantity", "scenario_base", "scenario_up", "scenario_down",
        "pnl_base", "pnl_up", "pnl_down", "prob_base", "prob_up", "prob_down",
        "cvar_95", "cvar_per_unit", "expected_pnl", "expected_pnl_per_unit",
        "risk_budget", "max_quantity", "created_at", "policy", "base_quantity",
        "policy_weight", "policy_score"
    ]
    if all(col -> col in names(df), desired_order)
        df = select(df, desired_order)
    else
        present = filter(col -> col in names(df), desired_order)
        remaining = setdiff(names(df), present)
        df = select(df, vcat(present, remaining))
    end

    db = DuckDB.DB(db_path)
    try
        DuckDB.execute(db, "CREATE SCHEMA IF NOT EXISTS mart")
        DuckDB.execute(db, """
            CREATE TABLE IF NOT EXISTS mart.risk_log (
                sced_ts_utc TIMESTAMPTZ,
                node TEXT,
                hub TEXT,
                direction TEXT,
                trade_basis DOUBLE,
                quantity DOUBLE,
                base_quantity DOUBLE,
                policy_weight DOUBLE,
                policy TEXT,
                policy_score DOUBLE,
                signed_quantity DOUBLE,
                scenario_base DOUBLE,
                scenario_up DOUBLE,
                scenario_down DOUBLE,
                pnl_base DOUBLE,
                pnl_up DOUBLE,
                pnl_down DOUBLE,
                prob_base DOUBLE,
                prob_up DOUBLE,
                prob_down DOUBLE,
                cvar_95 DOUBLE,
                cvar_per_unit DOUBLE,
                expected_pnl DOUBLE,
                expected_pnl_per_unit DOUBLE,
                risk_budget DOUBLE,
                max_quantity DOUBLE,
                created_at TIMESTAMPTZ,
                PRIMARY KEY (sced_ts_utc, node, hub)
            )
        """)
        cols = DataFrame(DuckDB.execute(db, "PRAGMA table_info('mart.risk_log')"))
        existing_cols = Set(String.(cols.name))
        alter_map = Dict(
            "base_quantity" => "ALTER TABLE mart.risk_log ADD COLUMN base_quantity DOUBLE",
            "policy_weight" => "ALTER TABLE mart.risk_log ADD COLUMN policy_weight DOUBLE",
            "policy" => "ALTER TABLE mart.risk_log ADD COLUMN policy TEXT",
            "policy_score" => "ALTER TABLE mart.risk_log ADD COLUMN policy_score DOUBLE",
        )
        for (col, stmt) in alter_map
            if !(col in existing_cols)
                try
                    DuckDB.execute(db, stmt)
                catch err
                    @debug "risk_log alter" stmt exception=(err, catch_backtrace())
                end
            end
        end

        DuckDB.register_data_frame(db, df, "risk_log_tmp")
        try
            DuckDB.execute(db, """
                INSERT OR REPLACE INTO mart.risk_log
                SELECT * FROM risk_log_tmp;
            """)
        finally
            DuckDB.unregister_data_frame(db, "risk_log_tmp")
        end
    finally
        close(db)
    end

    return nrow(df)
end

function _ensure_snapshot_table!(conn::DuckDB.DB)
    DuckDB.execute(conn, "CREATE SCHEMA IF NOT EXISTS mart")
    DuckDB.execute(conn, """
        CREATE TABLE IF NOT EXISTS mart.lag_snapshot_log (
            snapshot_date DATE PRIMARY KEY,
            latest_minute TIMESTAMPTZ,
            event_records BIGINT,
            created_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,
            source TEXT
        )
    """)
end

function publish_lag_snapshot!(db_path::AbstractString; source::AbstractString = "lagged_webapp")
    db = DuckDB.DB(db_path)
    try
        _ensure_snapshot_table!(db)
        df = DataFrame(DuckDB.execute(db, """
            SELECT max(sced_ts_utc_minute) AS latest_minute,
                   count(*) AS event_records
            FROM mart.event_price_history
        """))
        if isempty(df) || ismissing(df[1, :latest_minute])
            @warn "No event price history available to publish snapshot"
            return nothing
        end
        latest_minute = df[1, :latest_minute]
        records = df[1, :event_records]
        snapshot_date = Date(latest_minute)
        rows = DataFrame(
            snapshot_date = [snapshot_date],
            latest_minute = [latest_minute],
            event_records = [Int(records)],
            created_at = [Dates.now(Dates.UTC)],
            source = [source]
        )
        DuckDB.register_data_frame(db, rows, "lag_snapshot_tmp")
        try
            DuckDB.execute(db, """
                INSERT OR REPLACE INTO mart.lag_snapshot_log
                SELECT snapshot_date, latest_minute, event_records, created_at, source
                FROM lag_snapshot_tmp
            """)
        finally
            DuckDB.unregister_data_frame(db, "lag_snapshot_tmp")
        end
        return snapshot_date, latest_minute, records
    finally
        close(db)
    end
end

function ensure_training_tables!(db_path::AbstractString)
    db = DuckDB.DB(db_path)
    try
        DuckDB.execute(db, "CREATE SCHEMA IF NOT EXISTS mart")
        DuckDB.execute(db, """
            CREATE TABLE IF NOT EXISTS mart.training_runs (
                run_id TEXT,
                started_at TIMESTAMPTZ,
                finished_at TIMESTAMPTZ,
                snapshot_date DATE,
                policy TEXT,
                status TEXT,
                episodes INTEGER,
                reward_mean DOUBLE,
                reward_std DOUBLE,
                cvar_alpha DOUBLE,
                hyperparams TEXT,
                metrics TEXT,
                created_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,
                PRIMARY KEY (run_id)
            )
        """)
        DuckDB.execute(db, """
            CREATE TABLE IF NOT EXISTS mart.training_notes (
                note_id BIGINT GENERATED ALWAYS AS IDENTITY,
                run_id TEXT,
                created_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,
                author TEXT,
                category TEXT,
                note TEXT,
                PRIMARY KEY (note_id)
            )
        """)
        DuckDB.execute(db, "CREATE INDEX IF NOT EXISTS training_notes_run_idx ON mart.training_notes(run_id)")
    finally
        close(db)
    end
    return nothing
end

function _fetch_top_constraints(db::DuckDB.DB, limit::Int)
    df = DataFrame(DuckDB.execute(db, "SELECT constraint_name FROM ref.top_constraints ORDER BY bind_ct DESC LIMIT $limit"))
    return String.(df.constraint_name)
end

function _mu_unpivot_query(constraints::Vector{String})
    parts = String[]
    for name in constraints
        escaped = replace(name, _DOUBLE_QUOTE => _DOUBLE_QUOTE_ESC)
        push!(parts, "SELECT sced_ts_utc_minute, '$escaped' AS constraint_name, \"$escaped\" AS value FROM base")
    end
    return join(parts, " UNION ALL ")
end

function calibrate_scenario_cone(db_path::AbstractString;
                                 lookback::Dates.Period = Dates.Day(14),
                                 top_constraints::Int = 20,
                                 quantile::Float64 = 0.95)
    cutoff = Dates.now(Dates.UTC) - lookback
    db = DuckDB.DB(db_path)
    try
        constraints = _fetch_top_constraints(db, top_constraints)
        isempty(constraints) && error("ref.top_constraints returned no rows")
        union_sql = _mu_unpivot_query(constraints)
        query = """
            WITH base AS (
                SELECT sced_ts_utc_minute, {columns}
                FROM features.sced_mu
                WHERE sced_ts_utc_minute >= ?
            ),
            series AS (
                {unions}
            ),
            deltas AS (
                SELECT
                    constraint_name,
                    abs(value - lag(value) OVER (PARTITION BY constraint_name ORDER BY sced_ts_utc_minute)) AS abs_delta
                FROM series
            )
            SELECT abs_delta
            FROM deltas
            WHERE abs_delta IS NOT NULL
        """
        columns_sql = join((string('"', replace(name, _DOUBLE_QUOTE => _DOUBLE_QUOTE_ESC), '"') for name in constraints), ", ")
        sql = replace(query, "{columns}" => columns_sql, "{unions}" => union_sql)
        df = DataFrame(DuckDB.execute(db, sql, (cutoff,)))
        values = Float64[]
        for val in df.abs_delta
            isnothing(val) && continue
            push!(values, Float64(val))
        end
        isempty(values) && error("No μ deltas available for calibration window")
        delta = _empirical_quantile(values, quantile)
        tail_prob = mean(v -> v >= delta, values)
        per_tail = min(0.5, tail_prob / 2)
        base_prob = max(0.0, 1 - 2 * per_tail)
        sample_size = length(values)
        mean_abs = mean(values)
        std_abs = length(values) > 1 ? std(values) : 0.0
        lookback_minutes = Int(Dates.value(Dates.convert(Dates.Minute, lookback)))
        return (
            scenario_delta = delta,
            tail_prob = tail_prob,
            per_tail_prob = per_tail,
            base_prob = base_prob,
            quantile = quantile,
            lookback_minutes = lookback_minutes,
            top_constraints = top_constraints,
            sample_size = sample_size,
            mean_abs_delta = mean_abs,
            std_abs_delta = std_abs,
            run_ts = Dates.now(Dates.UTC)
        )
    finally
        isopen(db) && close(db)
    end
end

function persist_scenario_calibration!(db_path::AbstractString, calibration; source::AbstractString = "historical_mu")
    db = DuckDB.DB(db_path)
    try
        DuckDB.execute(db, "CREATE SCHEMA IF NOT EXISTS mart")
        DuckDB.execute(db, """
            CREATE TABLE IF NOT EXISTS mart.scenario_cone_calibration (
                run_ts TIMESTAMPTZ,
                lookback_minutes INTEGER,
                top_constraints INTEGER,
                quantile DOUBLE,
                scenario_delta DOUBLE,
                tail_prob DOUBLE,
                per_tail_prob DOUBLE,
                base_prob DOUBLE,
                sample_size BIGINT,
                mean_abs_delta DOUBLE,
                std_abs_delta DOUBLE,
                source TEXT,
                PRIMARY KEY (run_ts)
            )
        """)
        row = DataFrame(
            run_ts = [calibration.run_ts],
            lookback_minutes = [calibration.lookback_minutes],
            top_constraints = [calibration.top_constraints],
            quantile = [calibration.quantile],
            scenario_delta = [calibration.scenario_delta],
            tail_prob = [calibration.tail_prob],
            per_tail_prob = [calibration.per_tail_prob],
            base_prob = [calibration.base_prob],
            sample_size = [calibration.sample_size],
            mean_abs_delta = [calibration.mean_abs_delta],
            std_abs_delta = [calibration.std_abs_delta],
            source = [source]
        )
        DuckDB.register_data_frame(db, row, "scenario_calibration_tmp")
        try
            DuckDB.execute(db, """
                INSERT INTO mart.scenario_cone_calibration
                SELECT * FROM scenario_calibration_tmp;
            """)
        finally
            DuckDB.unregister_data_frame(db, "scenario_calibration_tmp")
        end
    finally
        close(db)
    end
    return 1
end

function latest_scenario_calibration(db_path::AbstractString)
    db = DuckDB.DB(db_path)
    try
        df = DataFrame(DuckDB.execute(db, "SELECT * FROM mart.scenario_cone_calibration ORDER BY run_ts DESC LIMIT 1"))
        isempty(df) && return nothing
        row = df[1, :]
        return (; run_ts = row.run_ts,
                 lookback_minutes = row.lookback_minutes,
                 top_constraints = row.top_constraints,
                 quantile = row.quantile,
                 scenario_delta = row.scenario_delta,
                 tail_prob = row.tail_prob,
                 per_tail_prob = row.per_tail_prob,
                 base_prob = row.base_prob,
                 sample_size = row.sample_size,
                 mean_abs_delta = row.mean_abs_delta,
                 std_abs_delta = row.std_abs_delta,
                 source = row.source)
    finally
        close(db)
    end
end

function what_if(beta_df::DataFrame, deltas::Dict{String,Float64}; nodes_filter::Union{Nothing,Vector{String}}=nothing)
    nodes = unique(beta_df.node)
    if nodes_filter !== nothing
        filter_set = Set(nodes_filter)
        nodes = filter(n -> n in filter_set, nodes)
    end
    impacts = Dict{String,Float64}()
    for node in nodes
        sdf = filter(:node => ==(node), beta_df)
        delta = 0.0
        for row in eachrow(sdf)
            δμ = get(deltas, row.constraint_name, 0.0)
            δμ == 0.0 && continue
            delta += row.beta * δμ
        end
        impacts[String(node)] = delta
    end
    return impacts
end

end
