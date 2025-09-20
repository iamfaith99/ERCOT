module LaggedWebApp

export start, stop, init_db_pool, with_lagged_date

using Genie
using Genie.Router
using Genie.Renderer.Json
using Genie.Requests
using DuckDB
using DataFrames
using Dates
using JSON3
using Logging
using Genie.Renderer.Html

const PROJECT_ROOT = abspath(joinpath(@__DIR__, "..", ".."))
const DEFAULT_DB_PATH = abspath(joinpath(PROJECT_ROOT, "data", "duckdb", "ercot.duckdb"))
const TRAINING_START_DATE = Date(2025, 8, 18)
const PIPELINE_PATH = abspath(joinpath(PROJECT_ROOT, "src"))
const PUBLIC_PATH = abspath(joinpath(PROJECT_ROOT, "webapp", "public"))

PIPELINE_PATH in LOAD_PATH || push!(LOAD_PATH, PIPELINE_PATH)

using ERCOTPipeline
using ERCOTPipeline: scenario_summary, load_latest_snapshot, build_event_graph,
                     build_feature_vector, predict_congestion,
                     initialize_market, state_prices, value_claim,
                     ensure_training_tables!, RLTradingEnv, reset!, step!, state,
                     is_done, log_run!

const PTDF = ERCOTPipeline.PTDFUtils

const DB_PATH = Ref{String}(DEFAULT_DB_PATH)
const DB_LOCK = ReentrantLock()
const DB_POOL = Ref(Vector{DuckDB.DB}())
const ROUTES_INITIALIZED = Ref(false)

function init_db_pool(; pool_size::Integer = 4, db_path::AbstractString = DEFAULT_DB_PATH)
    pool_size <= 0 && error("pool_size must be positive")
    DB_PATH[] = String(db_path)
    @info "Initializing DuckDB connection pool" db_path pool_size
    lock(DB_LOCK) do
        for conn in DB_POOL[]
            try
                close(conn)
            catch err
                @warn "Failed closing DuckDB connection" exception=(err, catch_backtrace())
            end
        end
        empty!(DB_POOL[])
        for _ in 1:pool_size
            push!(DB_POOL[], DuckDB.DB(DB_PATH[]))
        end
    end
    return nothing
end

function checkout_connection()
    lock(DB_LOCK) do
        isempty(DB_POOL[]) && push!(DB_POOL[], DuckDB.DB(DB_PATH[]))
        return pop!(DB_POOL[])
    end
end

function release_connection(conn::DuckDB.DB)
    lock(DB_LOCK) do
        push!(DB_POOL[], conn)
    end
    return nothing
end

function close_db_pool()
    lock(DB_LOCK) do
        while !isempty(DB_POOL[])
            conn = pop!(DB_POOL[])
            try
                close(conn)
            catch err
                @warn "Failed closing DuckDB connection" exception=(err, catch_backtrace())
            end
        end
    end
    return nothing
end

function with_db(f::Function)
    conn = checkout_connection()
    try
        return f(conn)
    finally
        release_connection(conn)
    end
end

function lagged_boundary()
    today_utc = Dates.today(Dates.UTC)
    return today_utc - Day(1)
end

function clamp_to_lag(date::Date)
    boundary = lagged_boundary()
    clamped_upper = date > boundary ? boundary : date
    return clamped_upper < TRAINING_START_DATE ? TRAINING_START_DATE : clamped_upper
end

function parse_lagged_date(day_param)
    if day_param === nothing
        return lagged_boundary()
    end
    str = strip(String(day_param))
    isempty(str) && return lagged_boundary()
    requested = try
        Date(str)
    catch err
        @warn "Failed to parse day parameter" value=str exception=(err, catch_backtrace())
        return lagged_boundary()
    end
    return clamp_to_lag(requested)
end

function with_lagged_date(f::Function, date::Date)
    return f(clamp_to_lag(date))
end

with_lagged_date(f::Function) = f(lagged_boundary())

function to_string_key_dict(dict)
    result = Dict{String,Any}()
    for (k, v) in dict
        key = String(k)
        result[key] = v isa Dict ? to_string_key_dict(v) : v isa Vector{<:Dict} ? [to_string_key_dict(x) for x in v] : v
    end
    return result
end

function get_optional_int(body::Dict, key::AbstractString)
    val = get(body, key, nothing)
    val === nothing && return nothing
    return val isa Number ? Int(val) : Int(parse(Int, String(val)))
end

function get_float(body::Dict, key::AbstractString, default::Float64)
    val = get(body, key, default)
    return val isa Number ? Float64(val) : Float64(parse(Float64, String(val)))
end

function get_int(body::Dict, key::AbstractString, default::Int)
    val = get(body, key, default)
    return val isa Number ? Int(val) : Int(parse(Int, String(val)))
end

function get_string(body::Dict, key::AbstractString, default::AbstractString)
    val = get(body, key, default)
    return String(val)
end

function table_to_dicts(df::DataFrame)
    rows = Vector{Dict{Symbol,Any}}(undef, nrow(df))
    cols = names(df)
    for (i, r) in enumerate(eachrow(df))
        d = Dict{Symbol,Any}()
        for col in cols
            d[Symbol(col)] = r[col]
        end
        rows[i] = d
    end
    return rows
end

function ensure_snapshot_date!(mu_row::DataFrame, safe_date::Date)
    isempty(mu_row) && error("Snapshot query returned no rows")
    ts = mu_row[1, "sced_ts_utc"]
    ts isa Dates.AbstractDateTime || error("sced_ts_utc column missing DateTime")
    Date(ts) <= safe_date || error("Latest snapshot $(Date(ts)) exceeds allowed date $(safe_date)")
    return ts
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
        impacts = PTDF.what_if(beta_df, delta_map; nodes_filter = [node, hub])
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

function expected_value(pnls, probs)
    total = 0.0
    for label in (:base, :up, :down)
        total += probs[label] * pnls[label]
    end
    return total
end

function convert_namedtuple_vector(vec)
    [Dict(Symbol(k) => v for (k, v) in pairs(nt)) for nt in vec]
end

function convert_event_prices(prices::Dict{Symbol,Float64})
    Dict(string(k) => v for (k, v) in prices)
end

function convert_log_entries(entries)
    [Dict(string(k) => v for (k, v) in entry) for entry in entries]
end

function prepare_notes(raw_notes)
    if raw_notes isa Vector
        buf = NamedTuple{(:author,:category,:note),Tuple{String,String,String}}[]
        for entry in raw_notes
            author = String(get(entry, "author", get(entry, :author, "")))
            category = String(get(entry, "category", get(entry, :category, "general")))
            note = String(get(entry, "note", get(entry, :note, "")))
            isempty(strip(note)) && continue
            push!(buf, (author = author, category = category, note = note))
        end
        return buf
    elseif raw_notes isa AbstractString
        note = strip(String(raw_notes))
        return isempty(note) ? NamedTuple{(:author,:category,:note),Tuple{String,String,String}}[] : [(author = "", category = "general", note = note)]
    else
        return NamedTuple{(:author,:category,:note),Tuple{String,String,String}}[]
    end
end

function compute_policy_quantity(state_dict::Dict, policy::String, risk_aversion::Float64,
                                 temperature::Float64, max_quantity::Float64)
    dir = Float64(get(state_dict, :direction_sign, 1.0))
    base_quantity = Float64(get(state_dict, :base_quantity, 0.0))
    expected_per_unit = Float64(get(state_dict, :expected_per_unit, 0.0))
    cvar_per_unit = Float64(get(state_dict, :cvar_per_unit, 0.0))
    policy_score = expected_per_unit - risk_aversion * abs(cvar_per_unit)
    unsigned_quantity = if lowercase(policy) == "maxent"
        temp = temperature <= 0 ? 1.0 : temperature
        weight = 1 / (1 + exp(-policy_score / temp))
        base_quantity * weight
    else
        base_quantity
    end
    quantity = dir * unsigned_quantity
    return clamp(quantity, -max_quantity, max_quantity)
end

function run_training_job(date::Date; hub::String = "HB_HOUSTON",
                          target_nodes::Vector{String} = ["HB_WEST", "HB_NORTH"],
                          top_constraints::Int = 3,
                          scenario_delta::Float64 = 15.0,
                          cone_constraints::Int = 2,
                          scenario_probs = (base = 0.5, up = 0.25, down = 0.25),
                          cvar_alpha::Float64 = 0.95,
                          risk_budget::Float64 = 1000.0,
                          max_quantity::Float64 = 200.0,
                          risk_aversion::Float64 = 1.0,
                          policy::String = "cvar",
                          temperature::Float64 = 50.0,
                          notes = NamedTuple[])
    ensure_training_tables!(DB_PATH[])
    safe_date = clamp_to_lag(date)
    env = RLTradingEnv(DB_PATH[];
        date = safe_date,
        hub = hub,
        nodes = target_nodes,
        top_constraints = top_constraints,
        scenario_delta = scenario_delta,
        cone_constraints = cone_constraints,
        scenario_probs = scenario_probs,
        cvar_alpha = cvar_alpha,
        risk_budget = risk_budget,
        max_quantity = max_quantity,
        risk_aversion = risk_aversion)

    current_state = reset!(env)
    while current_state !== nothing
        qty = compute_policy_quantity(current_state, policy, risk_aversion, temperature, max_quantity)
        current_state, _, done, _ = step!(env, qty)
        done && break
    end

    notes_vec = prepare_notes(notes)
    run_id = log_run!(env; policy = policy, status = "completed", notes = notes_vec)
    run_summary = to_string_key_dict(ERCOTPipeline.summary(env))
    probs_norm = env.scenario_probs
    scenario_probs_dict = Dict(
        :base => probs_norm.base,
        :up => probs_norm.up,
        :down => probs_norm.down
    )
    hyperparams = Dict(
        :date => string(safe_date),
        :hub => hub,
        :nodes => target_nodes,
        :top_constraints => top_constraints,
        :scenario_delta => scenario_delta,
        :cone_constraints => cone_constraints,
        :scenario_probs => scenario_probs_dict,
        :cvar_alpha => cvar_alpha,
        :risk_budget => risk_budget,
        :max_quantity => max_quantity,
        :risk_aversion => risk_aversion,
        :policy => policy,
        :temperature => temperature
    )
    return Dict(
        :run_id => run_id,
        :summary => run_summary,
        :log => convert_log_entries(env.log),
        :hyperparams => to_string_key_dict(hyperparams)
    )
end

function fetch_event_price_snapshot(date::Date)
    with_db() do conn
        sql = """
            SELECT sced_ts_utc_minute, event_id, price, liquidity, source
            FROM mart.event_price_history
            WHERE DATE(sced_ts_utc_minute) = ?
            ORDER BY sced_ts_utc_minute, event_id
            LIMIT 200
        """
        try
            df = with_lagged_date(date) do safe_date
                DataFrame(DuckDB.execute(conn, sql, (safe_date,)))
            end
            return (; records = table_to_dicts(df), count = nrow(df))
        catch err
            @warn "Snapshot query failed" date exception=(err, catch_backtrace())
            return (; records = Any[], count = 0, error = string(err))
        end
    end
end

function fetch_snapshot_metadata()
    with_db() do conn
        sql = """
            SELECT max(sced_ts_utc_minute) AS latest_minute,
                   count(*) AS records
            FROM mart.event_price_history
        """
        try
            df = DataFrame(DuckDB.execute(conn, sql))
            if isempty(df)
                return Dict{Symbol,Any}()
            else
                row = first(eachrow(df))
                return Dict(Symbol(col) => row[col] for col in names(df))
            end
        catch err
            @warn "Metadata query failed" exception=(err, catch_backtrace())
            return Dict{Symbol,Any}(:error => string(err))
        end
    end
end

function latest_snapshot_for(date::Date)
    safe_date = clamp_to_lag(date)
    mu_row, beta_df, intercept_df, metadata = load_latest_snapshot(DB_PATH[])
    snapshot_ts = ensure_snapshot_date!(mu_row, safe_date)
    return (; mu_row, beta_df, intercept_df, metadata, snapshot_ts)
end

function ensure_probs(probs)
    if probs isa Dict
        return (base = Float64(get(probs, "base", get(probs, :base, 0.5))),
                up = Float64(get(probs, "up", get(probs, :up, 0.25))),
                down = Float64(get(probs, "down", get(probs, :down, 0.25))))
    elseif probs isa NamedTuple
        return (base = Float64(probs.base), up = Float64(probs.up), down = Float64(probs.down))
    else
        return (base = 0.5, up = 0.25, down = 0.25)
    end
end

function compute_trade_suggestions(date::Date; hub::String = "HB_HOUSTON",
                                   target_nodes::Vector{String} = ["HB_WEST", "HB_NORTH"],
                                   top_constraints::Int = 3,
                                   risk_budget::Float64 = 1000.0,
                                   scenario_delta::Float64 = 15.0,
                                   cone_constraints::Int = 2,
                                   scenario_probs = (base = 0.5, up = 0.25, down = 0.25),
                                   cvar_alpha::Float64 = 0.95,
                                   max_quantity::Float64 = 200.0,
                                   policy::String = "cvar",
                                   temperature::Float64 = 50.0,
                                   policy_risk_aversion::Float64 = 1.0)
    snapshot = latest_snapshot_for(date)
    mu_row = snapshot.mu_row
    beta_df = snapshot.beta_df
    intercept_df = snapshot.intercept_df
    snapshot_ts = snapshot.snapshot_ts

    probs = ensure_probs(scenario_probs)
    feature_values = build_feature_vector(mu_row)
    logistic_params = PTDF._load_logistic_params(DB_PATH[])
    thresholds = PTDF._fetch_stat_thresholds(DB_PATH[])
    fact_row = PTDF._load_latest_fact_row(DB_PATH[])
    minute_ts = DateTime(mu_row[1, "sced_ts_utc_minute"])
    price_map, congestion_map = PTDF._load_actual_node_data(DB_PATH[], minute_ts)
    nodes_for_eval = unique([target_nodes...; hub])
    predictions, contributions = predict_congestion(beta_df, intercept_df, feature_values;
                                                    nodes_filter = nodes_for_eval,
                                                    top_constraints = top_constraints)

    results = NamedTuple{(:node,:basis,:drivers,:direction,:quantity,:base_quantity,:policy_weight,:policy,
                          :scenario_basis,:scenario_pnl,:cvar_95,:expected_pnl,:probabilities,:signed_quantity,
                          :value_claim_unit,:value_claim_total)}[]

    graph, priors = build_event_graph(predictions, contributions;
                                      logistic_params = Dict{Tuple{String,String},Tuple{Float64,Float64}}(logistic_params))
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
    hub_price === missing && error("Hub price missing for $(hub)")

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
        cvar_per_unit = compute_cvar(scenario_pnl_unit, probs; alpha = cvar_alpha)
        expected_per_unit = expected_value(scenario_pnl_unit, probs)

        base_quantity = abs(cvar_per_unit) > 1e-6 ? min(max_quantity, risk_budget / abs(cvar_per_unit)) : 0.0
        policy_weight = 1.0
        policy_score = expected_per_unit - policy_risk_aversion * abs(cvar_per_unit)
        if lowercase(policy) == "maxent"
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
        haskey(event_prices, basis_pos_sym) && (claim[basis_pos_sym] = direction_sign * scenario_delta)
        haskey(event_prices, basis_neg_sym) && (claim[basis_neg_sym] = -direction_sign * scenario_delta)
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
                        probabilities = probs,
                        signed_quantity = signed_quantity,
                        value_claim_unit = value_claim_unit,
                        value_claim_total = value_claim_total))
    end

    return (; trades = results,
            event_prices = event_prices,
            scenario_probs = probs,
            snapshot_timestamp = snapshot_ts,
            hub = hub)
end

function setup_routes()
    route("/", method = GET) do
        json((status = "ok", message = "Lagged training service running"))
    end

    route("/dashboard", method = GET) do
        htmlfile(joinpath(PUBLIC_PATH, "index.html"))
    end

    route("/api/ping", method = GET) do
        json((status = "ok", server_time = Dates.now(Dates.UTC)))
    end

    route("/api/snapshots", method = GET) do
        params_dict = params()
        day_param = get(params_dict, "day", nothing)
        date = parse_lagged_date(day_param)
        snapshot = fetch_event_price_snapshot(date)
        meta = fetch_snapshot_metadata()
        payload = Dict{Symbol,Any}(:status => "ok",
                                   :date => string(date),
                                   :count => snapshot.count,
                                   :records => snapshot.records,
                                   :metadata => meta)
        if hasproperty(snapshot, :error)
            payload[:status] = "error"
            payload[:error] = snapshot.error
        end
        json(payload)
    end

    route("/api/status", method = GET) do
        pool_size = lock(DB_LOCK) do
            length(DB_POOL[])
        end
        json((status = "ok",
              db_path = DB_PATH[],
              pool_size = pool_size,
              server_time = Dates.now(Dates.UTC)))
    end

    route("/api/simulate", method = POST) do
        try
            body = try
                payload = jsonpayload()
                payload isa Dict ? payload : Dict{String,Any}()
            catch
                Dict{String,Any}()
            end
            date = parse_lagged_date(get(body, "date", nothing))
            nodes = get(body, "nodes", nothing)
            nodes_filter = nodes isa Vector ? String.(nodes) : nothing
            nodes_limit = get_optional_int(body, "nodes_limit")
            top_constraints = get_int(body, "top_constraints", 3)
            liquidity = get_float(body, "liquidity", 5.0)
            safe_date = clamp_to_lag(date)
            summary = scenario_summary(DB_PATH[]; nodes_filter = nodes_filter,
                                       nodes_limit = nodes_limit,
                                       top_constraints = top_constraints,
                                       b = liquidity)
            ts_str = get(summary, :timestamp, nothing)
            ts_str === nothing && error("Scenario summary missing timestamp")
            ts = try
                DateTime(ts_str)
            catch
                DateTime(String(ts_str))
            end
            Date(ts) <= safe_date || error("Scenario data $(Date(ts)) exceeds allowed date $(safe_date)")
            json((status = "ok",
                  date = string(safe_date),
                  summary = summary))
        catch err
            @error "Scenario simulation failed" exception=(err, catch_backtrace())
            json((status = "error", error = string(err)), status = 500)
        end
    end

    route("/api/trades", method = POST) do
        try
            body = try
                payload = jsonpayload()
                payload isa Dict ? payload : Dict{String,Any}()
            catch
                Dict{String,Any}()
            end
            date = parse_lagged_date(get(body, "date", nothing))
            raw_nodes = get(body, "nodes", nothing)
            default_nodes = ["HB_WEST", "HB_NORTH"]
            target_nodes = raw_nodes === nothing ? default_nodes : raw_nodes isa Vector ? String.(raw_nodes) : [String(raw_nodes)]
            hub = get_string(body, "hub", "HB_HOUSTON")
            top_constraints = get_int(body, "top_constraints", 3)
            risk_budget = get_float(body, "risk_budget", 1000.0)
            scenario_delta = get_float(body, "scenario_delta", 15.0)
            cone_constraints = get_int(body, "cone_constraints", 2)
            probs = get(body, "scenario_probs", (base = 0.5, up = 0.25, down = 0.25))
            cvar_alpha = get_float(body, "cvar_alpha", 0.95)
            max_quantity = get_float(body, "max_quantity", 200.0)
            policy = lowercase(get_string(body, "policy", "cvar"))
            temperature = get_float(body, "temperature", 50.0)
            policy_risk_aversion = get_float(body, "policy_risk_aversion", 1.0)
            result = compute_trade_suggestions(date;
                hub = hub,
                target_nodes = target_nodes,
                top_constraints = top_constraints,
                risk_budget = risk_budget,
                scenario_delta = scenario_delta,
                cone_constraints = cone_constraints,
                scenario_probs = probs,
                cvar_alpha = cvar_alpha,
                max_quantity = max_quantity,
                policy = policy,
                temperature = temperature,
                policy_risk_aversion = policy_risk_aversion)
            response = Dict{Symbol,Any}(
                :status => "ok",
                :date => string(clamp_to_lag(date)),
                :snapshot_timestamp => string(result.snapshot_timestamp),
                :hub => result.hub,
                :trades => convert_namedtuple_vector(result.trades),
                :event_prices => convert_event_prices(result.event_prices),
                :scenario_probs => Dict(:base => result.scenario_probs.base,
                                        :up => result.scenario_probs.up,
                                        :down => result.scenario_probs.down)
            )
            json(response)
        catch err
            @error "Trade suggestion failed" exception=(err, catch_backtrace())
            json((status = "error", error = string(err)), status = 500)
        end
    end

    route("/api/train", method = POST) do
        try
            body = try
                payload = jsonpayload()
                payload isa Dict ? payload : Dict{String,Any}()
            catch
                Dict{String,Any}()
            end
            date = parse_lagged_date(get(body, "date", nothing))
            hub = get_string(body, "hub", "HB_HOUSTON")
            raw_nodes = get(body, "nodes", nothing)
            nodes = raw_nodes === nothing ? ["HB_WEST", "HB_NORTH"] : raw_nodes isa Vector ? String.(raw_nodes) : [String(raw_nodes)]
            top_constraints = get_int(body, "top_constraints", 3)
            scenario_delta = get_float(body, "scenario_delta", 15.0)
            cone_constraints = get_int(body, "cone_constraints", 2)
            scenario_probs = get(body, "scenario_probs", (base = 0.5, up = 0.25, down = 0.25))
            cvar_alpha = get_float(body, "cvar_alpha", 0.95)
            risk_budget = get_float(body, "risk_budget", 1000.0)
            max_quantity = get_float(body, "max_quantity", 200.0)
            risk_aversion = get_float(body, "risk_aversion", 1.0)
            policy = lowercase(get_string(body, "policy", "cvar"))
            temperature = get_float(body, "temperature", 50.0)
            notes = get(body, "notes", NamedTuple[])
            ensure_training_tables!(DB_PATH[])
            result = run_training_job(date;
                hub = hub,
                target_nodes = nodes,
                top_constraints = top_constraints,
                scenario_delta = scenario_delta,
                cone_constraints = cone_constraints,
                scenario_probs = scenario_probs,
                cvar_alpha = cvar_alpha,
                risk_budget = risk_budget,
                max_quantity = max_quantity,
                risk_aversion = risk_aversion,
                policy = policy,
                temperature = temperature,
                notes = notes)
            response = Dict{Symbol,Any}(:status => "ok", :date => string(clamp_to_lag(date)))
            for (k, v) in result
                response[k] = v
            end
            json(response)
        catch err
            @error "Training run failed" exception=(err, catch_backtrace())
            json((status = "error", error = string(err)), status = 500)
        end
    end

    return nothing
end

function ensure_routes()
    ROUTES_INITIALIZED[] && return
    setup_routes()
    ROUTES_INITIALIZED[] = true
end

function start(; host::AbstractString = get(ENV, "WEBAPP_HOST", "127.0.0.1"),
                 port::Integer = parse(Int, get(ENV, "WEBAPP_PORT", "9000")),
                 pool_size::Integer = parse(Int, get(ENV, "WEBAPP_DB_POOL", "4")),
                 db_path::AbstractString = get(ENV, "WEBAPP_DUCKDB_PATH", DEFAULT_DB_PATH))
    init_db_pool(; pool_size = pool_size, db_path = db_path)
    ensure_routes()
    Genie.config.run_as_server = true
    Genie.config.server_host = host
    Genie.config.server_port = port
    @info "Starting Genie server" host port db_path pool_size
    Genie.AppServer.startup(; host = host, port = port, open_browser = false, async = false)
end

function stop()
    close_db_pool()
    try
        Genie.AppServer.down!()
    catch err
        @warn "Error stopping Genie server" exception=(err, catch_backtrace())
    end
    return nothing
end

atexit(close_db_pool)

end # module
