#!/usr/bin/env julia
using Pkg
Pkg.activate(abspath(joinpath(@__DIR__, "..")))

using DataFrames
using DuckDB
using Statistics
using Printf
using Dates

push!(LOAD_PATH, abspath(joinpath(@__DIR__, "../src")))

include(joinpath(@__DIR__, "../src/ERCOTPipeline.jl"))
using .ERCOTPipeline

const DB_PATH = abspath(joinpath(@__DIR__, "..", "data", "duckdb", "ercot.duckdb"))
const EXTRA_REGRESSORS = [:scarcity_adder, :mcpc_regup, :mcpc_rrs, :mcpc_ecrs, :mcpc_nspin]

function _sanitize_symbol(label::AbstractString)
    Symbol(replace(label, r"[^A-Za-z0-9_]" => "_"))
end

function _logistic(x; center = 30.0, scale = 10.0)
    return 1 / (1 + exp(-(x - center) / scale))
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

        return mu_row, beta_df, intercept_df
    finally
        close(db)
    end
end

function build_feature_vector(mu_row::DataFrame)
    feature_values = Dict{String,Float64}()
    for col in names(mu_row)
        col in ("sced_ts_utc", "sced_ts_utc_minute") && continue
        feature_values[string(col)] = Float64(coalesce(mu_row[1, col], 0.0))
    end
    return feature_values
end

function predict_congestion(beta_df::DataFrame, intercept_df::DataFrame, feature_values::Dict{String,Float64}; nodes_limit::Int = 5)
    intercept_map = Dict(string(row.node) => Float64(row.intercept) for row in eachrow(intercept_df))
    nodes = unique(beta_df.node)
    isempty(nodes) && error("ref.estimated_ptdf is empty")
    selected_nodes = nodes[1:min(nodes_limit, length(nodes))]
    predictions = Dict{String,Float64}()
    contributions = Dict{String,Vector{NamedTuple{(:constraint_name,:contribution),Tuple{String,Float64}}}}()

    for node in selected_nodes
        sdf = filter(:node => ==(node), beta_df)
        value = get(intercept_map, string(node), 0.0)
        node_contribs = NamedTuple{(:constraint_name,:contribution),Tuple{String,Float64}}[]
        for row in eachrow(sdf)
            fv = get(feature_values, row.constraint_name, 0.0)
            contrib = row.beta * fv
            value += contrib
            push!(node_contribs, (constraint_name = row.constraint_name, contribution = contrib))
        end
        predictions[string(node)] = value
        contributions[string(node)] = sort(node_contribs; by = x -> abs(x.contribution), rev = true)
    end
    return predictions, contributions
end

function build_event_graph(predictions::Dict{String,Float64}, contributions)
    graph = EventGraph()
    base_events = Dict{Symbol,Float64}()

    for (node, value) in predictions
        prob = _logistic(value; center = 25.0, scale = 8.0)
        event_symbol = _sanitize_symbol("node_$(node)_gt25")
        add_event!(graph, EventNode(event_symbol; description = "Predicted congestion price > 25 for $(node)", scope = (prior = prob, tags = [:ptdf, :node])))
        base_events[event_symbol] = prob

        contribs = first(contributions[node], min(length(contributions[node]), 3))
        for contrib in contribs
            constraint_symbol = _sanitize_symbol("constraint_$(contrib.constraint_name)_pos")
            mu_prob = _logistic(contrib.contribution; center = 0.0, scale = 10.0)
            has_event(graph, constraint_symbol) || add_event!(graph, EventNode(constraint_symbol; description = "Constraint $(contrib.constraint_name) positive contribution", scope = (prior = mu_prob, tags = [:constraint, :ptdf])))
            base_events[constraint_symbol] = mu_prob
        end

        parent_symbols = [_sanitize_symbol("constraint_$(c.constraint_name)_pos") for c in contribs]
        if !isempty(parent_symbols)
            share_symbol = _sanitize_symbol("node_$(node)_driver_share")
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

function main()
    mu_row, beta_df, intercept_df = load_latest_snapshot(DB_PATH)
    feature_values = build_feature_vector(mu_row)
    constraint_rows = filter(:feature_type => ==("constraint"), beta_df)
    isempty(constraint_rows) && (constraint_rows = beta_df)

    predictions, contributions = predict_congestion(constraint_rows, intercept_df, feature_values)

    graph, priors = build_event_graph(predictions, contributions)
    market = initialize_market(priors; b = 5.0)
    prices = state_prices(market)

    println("Predicted congestion prices:")
    for (node, value) in predictions
        println(rpad(node, 20), @sprintf("%8.2f", value))
    end

    println("\nEvent price vector:")
    for (event, price) in sort(collect(prices); by = x -> x[2], rev = true)
        println(rpad(String(event), 35), @sprintf("%6.3f", price))
    end
end

main()
