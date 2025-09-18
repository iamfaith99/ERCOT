module PTDFUtils

using DataFrames
using DuckDB
using Dates
using Logging
using Statistics

import ..EventAlgebra: EventGraph, EventNode, add_event!, has_event, upsert_event!
import ..AssimilationModel: evaluate_event_priors, ensemble_event_priors
import ..MarketScoring: initialize_market, state_prices
import ..AssimilationRunner: analyze_and_forecast!

const EXTRA_REGRESSORS = [:scarcity_adder, :mcpc_regup, :mcpc_rrs, :mcpc_ecrs, :mcpc_nspin]
const EXTRA_REGRESSORS_STR = String.(EXTRA_REGRESSORS)

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
    function get_params(kind::String, default_center::Float64, default_scale::Float64, keys::AbstractString...)
        for key in keys
            if haskey(logistic_params, (kind, String(key)))
                return logistic_params[(kind, String(key))]
            end
        end
        return (default_center, default_scale)
    end
    for (node, value) in predictions
        (node_center, node_scale) = get_params("node_gt25", 25.0, 8.0, node, "global")
        prob = _logistic(value; center=node_center, scale=node_scale)
        event_symbol = sanitize_symbol("node_$(node)_gt25")
        add_event!(graph, EventNode(event_symbol; description = "Predicted congestion price > 25 for $(node)", scope = (prior = prob, tags = [:ptdf, :node])))
        base_events[event_symbol] = prob
        contribs = get(contributions, node, NamedTuple[])
        parent_symbols = Symbol[]
        for contrib in contribs
            constraint_symbol = sanitize_symbol("constraint_$(contrib.constraint_name)_pos")
            (contrib_center, contrib_scale) = get_params("contrib_pos", 0.0, 10.0, "global")
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
                                     prc_threshold::Float64=2000.0)
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
    assimilation_meta = nothing
    try
        model, Xa, config, meta = analyze_and_forecast!()
        new_entries = upsert_assimilation_events!(graph, priors, model, Xa, config)
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
    return Dict(
        :timestamp => string(mu_row[1, "sced_ts_utc"]),
        :liquidity => b,
        :metadata => metadata_dict,
        :nodes => node_payload,
        :event_prices => Dict(String(k) => v for (k, v) in prices)
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
    finally
        close(db)
    end

    return length(event_prices)
end

function persist_risk_log!(db_path::AbstractString, entries::Vector{<:NamedTuple})
    isempty(entries) && return 0
    df = DataFrame(entries)

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
