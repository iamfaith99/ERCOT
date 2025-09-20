using Dates
using Statistics
using UUIDs
using JSON3
using DataFrames
using DuckDB

import ..PTDFUtils
import ..PTDFUtils: load_latest_snapshot, build_feature_vector, predict_congestion,
                     build_event_graph, ensure_training_tables!, sanitize_symbol,
                     expand_event_vocabulary!, _load_logistic_params, _fetch_stat_thresholds,
                     _load_latest_fact_row, _load_actual_node_data, what_if, _ensure_snapshot_table!
import ..MarketScoring: initialize_market, state_prices, value_claim

const TRAINING_START_DATE = Date(2025, 8, 18)

struct NodeScenario
    node::String
    basis::Float64
    drivers::Vector{NamedTuple}
    direction_sign::Float64
    scenario_basis::NamedTuple{(:base,:up,:down),NTuple{3,Float64}}
    scenario_pnl_unit::NamedTuple{(:base,:up,:down),NTuple{3,Float64}}
    scenario_probs::NamedTuple{(:base,:up,:down),NTuple{3,Float64}}
    cvar_per_unit::Float64
    expected_per_unit::Float64
    base_quantity::Float64
    value_claim_unit::Float64
    metadata::Dict{Symbol,Any}
end

mutable struct TradingEnv
    db_path::String
    date::Date
    hub::String
    nodes::Vector{String}
    scenarios::Vector{NodeScenario}
    snapshot_timestamp::DateTime
    step_index::Int
    risk_aversion::Float64
    cvar_alpha::Float64
    scenario_probs::NamedTuple{(:base,:up,:down),NTuple{3,Float64}}
    max_quantity::Float64
    risk_budget::Float64
    event_prices::Dict{Symbol,Float64}
    log::Vector{Dict{Symbol,Any}}
end

lag_boundary() = Dates.today(Dates.UTC) - Day(1)

function clamp_training_date(date::Date)
    upper = lag_boundary()
    clamped = date > upper ? upper : date
    return clamped < TRAINING_START_DATE ? TRAINING_START_DATE : clamped
end

function normalize_probs(probs)
    if probs isa NamedTuple
        base = Float64(getfield(probs, :base))
        up = Float64(getfield(probs, :up))
        down = Float64(getfield(probs, :down))
    elseif probs isa Dict
        base_val = haskey(probs, :base) ? probs[:base] : get(probs, "base", 0.5)
        up_val = haskey(probs, :up) ? probs[:up] : get(probs, "up", 0.25)
        down_val = haskey(probs, :down) ? probs[:down] : get(probs, "down", 0.25)
        base = Float64(base_val)
        up = Float64(up_val)
        down = Float64(down_val)
    else
        base, up, down = 0.5, 0.25, 0.25
    end
    total = base + up + down
    if total <= 0
        return (base = 0.5, up = 0.25, down = 0.25)
    end
    return (base = base / total, up = up / total, down = down / total)
end

function scenario_basis_values(beta_df, node::String, hub::String, base_basis::Float64,
                               drivers, shock::Float64, cone_constraints::Int)
    selected = drivers[1:min(length(drivers), cone_constraints)]
    isempty(selected) && return (base = base_basis, up = base_basis, down = base_basis)
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
    tail <= 0 && return minimum(pnls[label] for label in (:base, :up, :down))
    entries = [(label, pnls[label], probs[label]) for label in (:base, :up, :down)]
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

expected_value(pnls, probs) = sum(probs[label] * pnls[label] for label in (:base, :up, :down))

function prepare_scenarios(db_path::AbstractString, safe_date::Date, hub::String, target_nodes::Vector{String};
                           top_constraints::Int = 3,
                           scenario_delta::Float64 = 15.0,
                           cone_constraints::Int = 2,
                           scenario_probs = (base = 0.5, up = 0.25, down = 0.25),
                           cvar_alpha::Float64 = 0.95,
                           risk_budget::Float64 = 1000.0,
                           max_quantity::Float64 = 200.0)
    mu_row, beta_df, intercept_df, metadata = load_latest_snapshot(db_path)
    isempty(mu_row) && error("load_latest_snapshot returned empty mu_row")
    snapshot_ts = mu_row[1, "sced_ts_utc"]
    Date(snapshot_ts) <= safe_date || error("Snapshot $(Date(snapshot_ts)) exceeds allowed training date $(safe_date)")
    feature_values = build_feature_vector(mu_row)
    logistic_params = _load_logistic_params(db_path)
    thresholds = _fetch_stat_thresholds(db_path)
    fact_row = _load_latest_fact_row(db_path)
    minute_ts = DateTime(mu_row[1, "sced_ts_utc_minute"])
    price_map, congestion_map = _load_actual_node_data(db_path, minute_ts)
    nodes_for_eval = unique([target_nodes...; hub])
    predictions, contributions = predict_congestion(beta_df, intercept_df, feature_values;
                                                    nodes_filter = nodes_for_eval,
                                                    top_constraints = top_constraints)
    graph, priors = build_event_graph(predictions, contributions;
                                      logistic_params = Dict{Tuple{String,String},Tuple{Float64,Float64}}(logistic_params))
    expand_event_vocabulary!(graph, priors, predictions, contributions;
                             logistic_params = logistic_params,
                             fact_row = fact_row,
                             thresholds = thresholds,
                             price_map = price_map,
                             congestion_map = congestion_map,
                             hub = hub,
                             feature_values = feature_values)
    market = initialize_market(priors; b = 5.0)
    event_prices = state_prices(market)

    scenarios = NodeScenario[]
    probs = normalize_probs(scenario_probs)

    hub_price = get(predictions, hub, missing)
    hub_price === missing && error("Hub price missing for $(hub)")

    for node in target_nodes
        node_price = get(predictions, node, missing)
        node_price === missing && continue
        basis = node_price - hub_price
        drivers = get(contributions, node, NamedTuple[])
        isempty(drivers) && continue
        direction_sign = basis ≥ 0 ? -1.0 : 1.0
        scenario_basis = scenario_basis_values(beta_df, node, hub, basis, drivers, scenario_delta, cone_constraints)
        scenario_pnl_unit = pnl_namedtuple(direction_sign, basis, scenario_basis)
        cvar_per_unit = compute_cvar(scenario_pnl_unit, probs; alpha = cvar_alpha)
        expected_per_unit = expected_value(scenario_pnl_unit, probs)
        base_quantity = abs(cvar_per_unit) > 1e-6 ? min(max_quantity, risk_budget / abs(cvar_per_unit)) : 0.0
        basis_pos_sym = sanitize_symbol("basis_spike_pos_$(node)_vs_$(PTDFUtils.BASIS_HUB)")
        basis_neg_sym = sanitize_symbol("basis_spike_neg_$(node)_vs_$(PTDFUtils.BASIS_HUB)")
        claim = Dict{Symbol,Float64}()
        haskey(event_prices, basis_pos_sym) && (claim[basis_pos_sym] = direction_sign * scenario_delta)
        haskey(event_prices, basis_neg_sym) && (claim[basis_neg_sym] = -direction_sign * scenario_delta)
        value_claim_unit = isempty(claim) ? 0.0 : value_claim(event_prices, claim)
        metadata = Dict{Symbol,Any}(
            :drivers => drivers,
            :scenario_basis => scenario_basis,
            :scenario_probs => probs,
            :scenario_pnl_unit => scenario_pnl_unit,
            :value_claim_unit => value_claim_unit
        )
        push!(scenarios, NodeScenario(node, basis, drivers, direction_sign, scenario_basis,
                                      scenario_pnl_unit, probs, cvar_per_unit, expected_per_unit,
                                      base_quantity, value_claim_unit, metadata))
    end

    return scenarios, event_prices, snapshot_ts
end

function RLTradingEnv(db_path::AbstractString;
                      date::Union{Nothing,Date} = nothing,
                      hub::String = "HB_HOUSTON",
                      nodes::Vector{String} = ["HB_WEST", "HB_NORTH"],
                      top_constraints::Int = 3,
                      scenario_delta::Float64 = 15.0,
                      cone_constraints::Int = 2,
                      scenario_probs = (base = 0.5, up = 0.25, down = 0.25),
                      cvar_alpha::Float64 = 0.95,
                      risk_budget::Float64 = 1000.0,
                      max_quantity::Float64 = 200.0,
                      risk_aversion::Float64 = 1.0)
    target_date = date === nothing ? Date(clamp_training_date(Dates.today(Dates.UTC) - Day(1))) : Date(date)
    safe_date = clamp_training_date(target_date)
    scenarios, event_prices, snapshot_ts = prepare_scenarios(db_path, safe_date, hub, nodes;
                                                             top_constraints = top_constraints,
                                                             scenario_delta = scenario_delta,
                                                             cone_constraints = cone_constraints,
                                                             scenario_probs = scenario_probs,
                                                             cvar_alpha = cvar_alpha,
                                                             risk_budget = risk_budget,
                                                             max_quantity = max_quantity)
    probs_norm = normalize_probs(scenario_probs)
    log = Dict{Symbol,Any}[]
    return TradingEnv(String(db_path), safe_date, hub, collect(String.(nodes)), scenarios,
                      snapshot_ts, 1, risk_aversion, cvar_alpha, probs_norm,
                      max_quantity, risk_budget, event_prices, log)
end

function is_done(env::TradingEnv)
    return env.step_index > length(env.scenarios)
end

function state(env::TradingEnv)
    is_done(env) && return nothing
    scenario = env.scenarios[env.step_index]
    return Dict(
        :step => env.step_index,
        :total_steps => length(env.scenarios),
        :node => scenario.node,
        :basis => scenario.basis,
        :direction_sign => scenario.direction_sign,
        :scenario_basis => scenario.scenario_basis,
        :scenario_probs => scenario.scenario_probs,
        :cvar_per_unit => scenario.cvar_per_unit,
        :expected_per_unit => scenario.expected_per_unit,
        :base_quantity => scenario.base_quantity,
        :value_claim_unit => scenario.value_claim_unit,
        :drivers => [Dict(:constraint_name => d.constraint_name,
                           :beta => d.beta,
                           :mu_value => d.mu_value,
                           :contribution => d.contribution,
                           :feature_type => d.feature_type) for d in scenario.drivers]
    )
end

function reset!(env::TradingEnv)
    env.step_index = 1
    empty!(env.log)
    return state(env)
end

function step!(env::TradingEnv, action)
    is_done(env) && error("Environment already finished; call reset! first")
    scenario = env.scenarios[env.step_index]
    quantity = action isa NamedTuple ? Float64(action[:quantity]) : Float64(action)
    quantity = clamp(quantity, -env.max_quantity, env.max_quantity)
    expected_total = scenario.expected_per_unit * quantity
    cvar_total = scenario.cvar_per_unit * quantity
    reward = expected_total - env.risk_aversion * abs(cvar_total)
    option_value = scenario.value_claim_unit * quantity
    log_entry = Dict{Symbol,Any}(
        :step => env.step_index,
        :node => scenario.node,
        :quantity => quantity,
        :reward => reward,
        :expected_pnl => expected_total,
        :cvar_total => cvar_total,
        :option_value => option_value,
        :basis => scenario.basis,
        :scenario_probs => scenario.scenario_probs,
        :timestamp => env.snapshot_timestamp
    )
    push!(env.log, log_entry)
    env.step_index += 1
    done = is_done(env)
    next_state = done ? nothing : state(env)
    info = log_entry
    return next_state, reward, done, info
end

function summary(env::TradingEnv)
    rewards = [entry[:reward] for entry in env.log]
    expected = [entry[:expected_pnl] for entry in env.log]
    cvar_vals = [entry[:cvar_total] for entry in env.log]
    total_reward = sum(rewards)
    return Dict(
        :date => string(env.date),
        :hub => env.hub,
        :nodes => env.nodes,
        :steps => length(env.scenarios),
        :completed_steps => length(env.log),
        :total_reward => total_reward,
        :reward_mean => isempty(rewards) ? 0.0 : mean(rewards),
        :reward_std => length(rewards) > 1 ? std(rewards) : 0.0,
        :expected_pnl_total => sum(expected),
        :cvar_total => sum(cvar_vals),
        :risk_aversion => env.risk_aversion,
        :cvar_alpha => env.cvar_alpha,
        :max_quantity => env.max_quantity,
        :risk_budget => env.risk_budget
    )
end

function log_run!(env::TradingEnv; run_id::Union{Nothing,String}=nothing,
                  status::String = "completed",
                  policy::String = "custom",
                  episodes::Union{Nothing,Int}=nothing,
                  notes::Vector{NamedTuple}=NamedTuple[])
    ensure_training_tables!(env.db_path)
    rewards = [entry[:reward] for entry in env.log]
    reward_mean = isempty(rewards) ? 0.0 : mean(rewards)
    reward_std = length(rewards) > 1 ? std(rewards) : 0.0
    total_reward = sum(rewards)
    run_id_val = run_id === nothing ? string(uuid4()) : String(run_id)
    episodes_val = episodes === nothing ? length(env.log) : episodes
    hyperparams = Dict(
        :date => string(env.date),
        :hub => env.hub,
        :nodes => env.nodes,
        :top_constraints => length(env.scenarios) == 0 ? nothing : length(env.scenarios[1].drivers),
        :risk_budget => env.risk_budget,
        :max_quantity => env.max_quantity,
        :risk_aversion => env.risk_aversion,
        :cvar_alpha => env.cvar_alpha
    )
    metrics = summary(env)
    db = DuckDB.DB(env.db_path)
    try
        try
            _ensure_snapshot_table!(db)
        catch
        end
        DuckDB.execute(db, "CREATE SCHEMA IF NOT EXISTS mart")
        run_row = DataFrame(
            run_id = [run_id_val],
            started_at = [env.snapshot_timestamp],
            finished_at = [Dates.now(Dates.UTC)],
            snapshot_date = [env.date],
            policy = [policy],
            status = [status],
            episodes = [episodes_val],
            reward_mean = [reward_mean],
            reward_std = [reward_std],
            cvar_alpha = [env.cvar_alpha],
            hyperparams = [JSON3.write(hyperparams)],
            metrics = [JSON3.write(metrics)],
            created_at = [Dates.now(Dates.UTC)]
        )
        DuckDB.register_data_frame(db, run_row, "training_run_tmp")
        try
            DuckDB.execute(db, """
                INSERT OR REPLACE INTO mart.training_runs
                SELECT run_id, started_at, finished_at, snapshot_date, policy, status,
                       episodes, reward_mean, reward_std, cvar_alpha, hyperparams, metrics, created_at
                FROM training_run_tmp
            """)
        finally
            DuckDB.unregister_data_frame(db, "training_run_tmp")
        end

        if !isempty(notes)
            note_rows = DataFrame(
                run_id = [run_id_val for _ in notes],
                created_at = [Dates.now(Dates.UTC) for _ in notes],
                author = [String(get(n, :author, "")) for n in notes],
                category = [String(get(n, :category, "general")) for n in notes],
                note = [String(get(n, :note, "")) for n in notes]
            )
            DuckDB.register_data_frame(db, note_rows, "training_notes_tmp")
            try
                DuckDB.execute(db, """
                    INSERT INTO mart.training_notes(run_id, created_at, author, category, note)
                    SELECT run_id, created_at, author, category, note FROM training_notes_tmp
                """)
            finally
                DuckDB.unregister_data_frame(db, "training_notes_tmp")
            end
        end
    finally
        close(db)
    end
    return run_id_val
end
