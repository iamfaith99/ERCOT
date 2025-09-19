#!/usr/bin/env julia
using Pkg
Pkg.activate(abspath(joinpath(@__DIR__, "..")))

using DataFrames
using Dates
using Logging

push!(LOAD_PATH, abspath(joinpath(@__DIR__, "../src")))
include(joinpath(@__DIR__, "../src/ERCOTPipeline.jl"))
using .ERCOTPipeline

const PTDF = ERCOTPipeline.PTDFUtils

const DB_PATH = abspath(joinpath(@__DIR__, "..", "data", "duckdb", "ercot.duckdb"))
const SCENARIO_LABELS = (:base, :up, :down)

function parse_env(name::String, default::T, parser::Function) where {T}
    raw = get(ENV, name, nothing)
    raw === nothing && return default
    try
        return parser(raw)
    catch err
        @warn "Failed to parse env; using default" name raw default err
        return default
    end
end

function parse_optional_env(name::String, parser::Function)
    raw = get(ENV, name, nothing)
    raw === nothing && return nothing
    try
        return parser(raw)
    catch err
        @warn "Failed to parse env; ignoring override" name raw err
        return nothing
    end
end

function parse_list(value::Union{Nothing,String})
    if value === nothing || isempty(value)
        return nothing
    end
    return split(value, ',')
end

function parse_probabilities(raw::Union{Nothing,String})
    default_probs = (base = 0.5, up = 0.25, down = 0.25)
    raw === nothing && return default_probs
    isempty(raw) && return default_probs
    parts = split(raw, ',')
    length(parts) == length(SCENARIO_LABELS) || return default_probs
    vals = Float64.(parts)
    total = sum(vals)
    total <= 0 && return default_probs
    normalized = vals ./ total
    return (base = normalized[1], up = normalized[2], down = normalized[3])
end

function scenario_basis_values(beta_df, node::String, hub::String, base_basis::Float64,
                               drivers, shock::Float64, cone_constraints::Int)
    selected = drivers[1:min(length(drivers), cone_constraints)]
    if isempty(selected)
        return (base = base_basis, up = base_basis, down = base_basis)
    end
    up_map = Dict{String,Float64}()
    down_map = Dict{String,Float64}()
    for drv in selected
        up_map[drv.constraint_name] = shock
        down_map[drv.constraint_name] = -shock
    end
    scenarios = Dict{Symbol,Float64}(:base => base_basis)
    for (label, delta_map) in zip((:up, :down), (up_map, down_map))
        impacts = what_if(beta_df, delta_map; nodes_filter = [node, hub])
        delta_basis = (get(impacts, node, 0.0) - get(impacts, hub, 0.0))
        scenarios[label] = base_basis + delta_basis
    end
    return (base = scenarios[:base], up = scenarios[:up], down = scenarios[:down])
end

function pnl_namedtuple(direction_sign::Float64, trade_basis::Float64, scenario_basis)
    base_pnl = direction_sign * (scenario_basis.base - trade_basis)
    up_pnl = direction_sign * (scenario_basis.up - trade_basis)
    down_pnl = direction_sign * (scenario_basis.down - trade_basis)
    return (base = base_pnl, up = up_pnl, down = down_pnl)
end

function compute_cvar(pnls, probs; alpha::Float64 = 0.95)
    tail = 1.0 - alpha
    tail <= 0 && return minimum(pnls[label] for label in SCENARIO_LABELS)
    entries = [(label, pnls[label], probs[label]) for label in SCENARIO_LABELS]
    sorted = sort(entries; by = x -> x[2])
    remaining = tail
    acc = 0.0
    last_pnl = sorted[end][2]
    for (_label, pnl, p) in sorted
        remaining <= 0 && break
        weight = min(p, remaining)
        acc += pnl * weight
        remaining -= weight
        last_pnl = pnl
    end
    remaining > 1e-9 && (acc += last_pnl * remaining)
    return acc / tail
end

function expected_value(pnls, probs)
    total = 0.0
    for label in SCENARIO_LABELS
        total += probs[label] * pnls[label]
    end
    return total
end

function latest_tick_timestamp(mu_row::DataFrame)
    return mu_row[1, "sced_ts_utc"]
end

function data_is_fresh(tick_time; tolerance::Period = Minute(10))
    return (now(Dates.UTC) - tick_time) <= tolerance
end

function suggest_trades(mu_row, beta_df, intercept_df;
                        db_path::AbstractString,
                        hub::String,
                        target_nodes::Vector{String},
                        top_constraints::Int,
                        risk_budget::Float64,
                        scenario_delta::Float64,
                        cone_constraints::Int,
                        scenario_probs,
                        cvar_alpha::Float64,
                        max_quantity::Float64,
                        tick_time,
                        policy::String,
                        temperature::Float64,
                        policy_risk_aversion::Float64)
    feature_values = build_feature_vector(mu_row)
    logistic_params = PTDF._load_logistic_params(db_path)
    thresholds = PTDF._fetch_stat_thresholds(db_path)
    fact_row = PTDF._load_latest_fact_row(db_path)
    minute_ts = DateTime(mu_row[1, "sced_ts_utc_minute"])
    price_map, congestion_map = PTDF._load_actual_node_data(db_path, minute_ts)
    nodes_for_eval = unique([target_nodes...; hub])
    predictions, contributions = predict_congestion(beta_df, intercept_df, feature_values; nodes_filter = nodes_for_eval, top_constraints = top_constraints)
    results = NamedTuple{(:node,:basis,:drivers,:direction,:quantity,:base_quantity,:policy_weight,:policy,:scenario_basis,:scenario_pnl,:cvar_95,:expected_pnl,:probabilities,:signed_quantity,:value_claim_unit,:value_claim_total)}[]
    risk_entries = NamedTuple{(:sced_ts_utc,:node,:hub,:direction,:trade_basis,:quantity,:base_quantity,:policy_weight,:policy,:policy_score,:signed_quantity,
                               :scenario_base,:scenario_up,:scenario_down,
                               :pnl_base,:pnl_up,:pnl_down,
                               :prob_base,:prob_up,:prob_down,
                               :cvar_95,:cvar_per_unit,:expected_pnl,:expected_pnl_per_unit,
                               :risk_budget,:max_quantity,:created_at)}[]
    graph, priors = build_event_graph(predictions, contributions)
    _ = PTDF.expand_event_vocabulary!(graph, priors, predictions, contributions;
                                      logistic_params = logistic_params,
                                      fact_row = fact_row,
                                      thresholds = thresholds,
                                      price_map = price_map,
                                      congestion_map = congestion_map,
                                      hub = hub,
                                      feature_values = feature_values)
    market = initialize_market(priors; b = 5.0)
    event_prices = state_prices(market)

    hub_price = get(predictions, hub, missing)
    if hub_price === missing
        @warn "Hub price missing" hub
        return results, event_prices, risk_entries
    end
    for node in target_nodes
        node_price = get(predictions, node, missing)
        node_price === missing && continue
        basis = node_price - hub_price
        drivers = get(contributions, node, NamedTuple[])
        isempty(drivers) && continue

        direction_sign = basis ≥ 0 ? -1.0 : 1.0
        direction = direction_sign > 0 ? "Buy basis" : "Sell basis"
        scenario_basis = scenario_basis_values(beta_df, node, hub, basis, drivers, scenario_delta, cone_constraints)
        scenario_pnl_unit = pnl_namedtuple(direction_sign, basis, scenario_basis)
        cvar_per_unit = compute_cvar(scenario_pnl_unit, scenario_probs; alpha = cvar_alpha)
        expected_per_unit = expected_value(scenario_pnl_unit, scenario_probs)

        base_quantity = 0.0
        if abs(cvar_per_unit) > 1e-6
            base_quantity = min(max_quantity, risk_budget / abs(cvar_per_unit))
        end
        policy_weight = 1.0
        policy_score = expected_per_unit - policy_risk_aversion * abs(cvar_per_unit)
        if policy == "maxent"
            temp = temperature <= 0 ? 1.0 : temperature
            policy_weight = 1 / (1 + exp(-policy_score / temp))
        end
        quantity = base_quantity * policy_weight
        signed_quantity = direction_sign * quantity
        scenario_pnl_total = (base = scenario_pnl_unit.base * quantity,
                              up = scenario_pnl_unit.up * quantity,
                              down = scenario_pnl_unit.down * quantity)
        cvar_total = cvar_per_unit * quantity
        expected_total = expected_per_unit * quantity

        basis_pos_sym = PTDF.sanitize_symbol("basis_spike_pos_$(node)_vs_$(PTDF.BASIS_HUB)")
        basis_neg_sym = PTDF.sanitize_symbol("basis_spike_neg_$(node)_vs_$(PTDF.BASIS_HUB)")
        claim = Dict{Symbol,Float64}()
        if haskey(event_prices, basis_pos_sym)
            claim[basis_pos_sym] = direction_sign * scenario_delta
        end
        if haskey(event_prices, basis_neg_sym)
            claim[basis_neg_sym] = -direction_sign * scenario_delta
        end
        value_claim_unit = isempty(claim) ? 0.0 : value_claim(event_prices, claim)
        value_claim_total = value_claim_unit * quantity

        push!(results, (node = node,
                        basis = basis,
                        drivers = drivers,
                        direction = direction,
                        quantity = quantity,
                        base_quantity = base_quantity,
                        policy_weight = policy_weight,
                        policy = policy,
                        scenario_basis = scenario_basis,
                        scenario_pnl = scenario_pnl_total,
                        cvar_95 = cvar_total,
                        expected_pnl = expected_total,
                        probabilities = scenario_probs,
                        signed_quantity = signed_quantity,
                        value_claim_unit = value_claim_unit,
                        value_claim_total = value_claim_total))

        push!(risk_entries, (
            sced_ts_utc = tick_time,
            node = node,
            hub = hub,
            direction = direction,
            trade_basis = basis,
            quantity = quantity,
            base_quantity = base_quantity,
            policy_weight = policy_weight,
            policy = policy,
            policy_score = policy_score,
            signed_quantity = signed_quantity,
            scenario_base = scenario_basis.base,
            scenario_up = scenario_basis.up,
            scenario_down = scenario_basis.down,
            pnl_base = scenario_pnl_total.base,
            pnl_up = scenario_pnl_total.up,
            pnl_down = scenario_pnl_total.down,
            prob_base = scenario_probs.base,
            prob_up = scenario_probs.up,
            prob_down = scenario_probs.down,
            cvar_95 = cvar_total,
            cvar_per_unit = cvar_per_unit,
            expected_pnl = expected_total,
            expected_pnl_per_unit = expected_per_unit,
            risk_budget = risk_budget,
            max_quantity = max_quantity,
            created_at = Dates.now(Dates.UTC)
        ))
    end
    return results, event_prices, risk_entries
end

function main()
    iterations = parse_env("TRADING_ITERATIONS", 1, x -> parse(Int, x))
    sleep_seconds = parse_env("TRADING_SLEEP_SECONDS", 300, x -> parse(Int, x))
    nodes = parse_list(get(ENV, "TRADING_NODES", nothing))
    nodes === nothing && (nodes = ["HB_WEST", "HB_NORTH"])
    hub = get(ENV, "TRADING_HUB", "HB_HOUSTON")
    tolerance_minutes = parse_env("TRADING_TOLERANCE_MINUTES", 10, x -> parse(Int, x))
    risk_budget = parse_env("TRADING_RISK_BUDGET", 1000.0, x -> parse(Float64, x))
    top_constraints = parse_env("TRADING_TOP_CONSTRAINTS", 3, x -> parse(Int, x))
    cone_constraints = parse_env("TRADING_SCENARIO_CONSTRAINTS", 2, x -> parse(Int, x))
    cvar_alpha = parse_env("TRADING_CVAR_ALPHA", 0.95, x -> parse(Float64, x))
    max_quantity = parse_env("TRADING_MAX_QUANTITY", 200.0, x -> parse(Float64, x))
    raw_policy = lowercase(strip(get(ENV, "TRADING_POLICY", "cvar")))
    policy_candidates = if isempty(raw_policy)
        ["cvar"]
    elseif occursin(',', raw_policy)
        filter(!isempty, strip.(split(raw_policy, ',')))
    elseif raw_policy in ("swap", "both")
        ["cvar", "maxent"]
    else
        [raw_policy]
    end
    policy_cycle = String[]
    for mode in policy_candidates
        mode_norm = lowercase(strip(mode))
        if mode_norm in ("cvar", "maxent")
            push!(policy_cycle, mode_norm)
        end
    end
    isempty(policy_cycle) && push!(policy_cycle, "cvar")
    temperature = parse_env("TRADING_TEMPERATURE", 50.0, x -> parse(Float64, x))
    policy_risk_aversion = parse_env("TRADING_POLICY_RISK_AVERSION", 1.0, x -> parse(Float64, x))

    scenario_delta_override = parse_optional_env("TRADING_SCENARIO_DELTA", x -> parse(Float64, x))
    scenario_probs_override = get(ENV, "TRADING_SCENARIO_PROBS", nothing)
    calibration = latest_scenario_calibration(DB_PATH)

    scenario_delta = if scenario_delta_override !== nothing
        scenario_delta_override
    elseif calibration !== nothing
        calibration.scenario_delta
    else
        15.0
    end

    scenario_probs = if scenario_probs_override !== nothing
        parse_probabilities(scenario_probs_override)
    elseif calibration !== nothing
        (base = calibration.base_prob,
         up = calibration.per_tail_prob,
         down = calibration.per_tail_prob)
    else
        parse_probabilities(nothing)
    end

    prob_sum = scenario_probs.base + scenario_probs.up + scenario_probs.down
    if abs(prob_sum - 1.0) > 1e-6 && prob_sum > 0
        scenario_probs = (base = scenario_probs.base / prob_sum,
                          up = scenario_probs.up / prob_sum,
                          down = scenario_probs.down / prob_sum)
    end

    if scenario_delta_override === nothing && scenario_probs_override === nothing && calibration !== nothing
        @info "Loaded scenario calibration" delta = scenario_delta base_prob = scenario_probs.base up_prob = scenario_probs.up down_prob = scenario_probs.down run_ts = calibration.run_ts sample_size = calibration.sample_size
    elseif scenario_delta_override === nothing && scenario_probs_override === nothing
        @info "Using default scenario cone" delta = scenario_delta base_prob = scenario_probs.base up_prob = scenario_probs.up down_prob = scenario_probs.down
    end

    function maybe_sleep(iter)
        iter < iterations && sleep_seconds > 0 && sleep(sleep_seconds)
    end

    for iter in 1:iterations
        metadata = load_metadata(DB_PATH)
        if metadata === nothing
            @warn "No published PTDF metadata found"; maybe_sleep(iter); continue
        end
        improvement = model_improvement(metadata)
        if improvement !== missing && improvement < 0
            @warn "Model improvement below zero; skipping tick" improvement
            maybe_sleep(iter); continue
        end
        if !model_is_fresh(metadata; max_age = Hour(2))
            @warn "Model metadata is stale; skipping tick" last_published = metadata.run_ts
            maybe_sleep(iter); continue
        end

        mu_row, beta_df, intercept_df, _ = load_latest_snapshot(DB_PATH)
        tick_time = latest_tick_timestamp(mu_row)
        if !data_is_fresh(tick_time; tolerance = Minute(tolerance_minutes))
            @warn "Latest SCED data is stale; skipping tick" tick_time
            maybe_sleep(iter); continue
        end

        policy_idx = mod(iter - 1, length(policy_cycle)) + 1
        policy_mode = policy_cycle[policy_idx]

        trades, event_prices, risk_entries = suggest_trades(mu_row, beta_df, intercept_df;
                                                           db_path = DB_PATH,
                                                           hub = hub,
                                                           target_nodes = nodes,
                                                           top_constraints = top_constraints,
                                                           risk_budget = risk_budget,
                                                           scenario_delta = scenario_delta,
                                                           cone_constraints = cone_constraints,
                                                           scenario_probs = scenario_probs,
                                                           cvar_alpha = cvar_alpha,
                                                           max_quantity = max_quantity,
                                                           tick_time = tick_time,
                                                           policy = policy_mode,
                                                           temperature = temperature,
                                                           policy_risk_aversion = policy_risk_aversion)

        if isempty(trades)
            @info "No trade suggestions for this tick" tick_time
        else
            @info "Trade suggestions" tick = tick_time
            for trade in trades
                @info "Basis signal" node = trade.node policy = trade.policy basis = trade.basis direction = trade.direction quantity = trade.quantity cvar_95 = trade.cvar_95 expected_pnl = trade.expected_pnl option_value = trade.value_claim_total
            end
            written = persist_risk_log!(DB_PATH, risk_entries)
            @info "Risk log updated" rows = written
        end

        @debug "Event prices" event_prices

        maybe_sleep(iter)
    end
end

main()
