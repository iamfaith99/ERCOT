#!/usr/bin/env julia

using Pkg

const ROOT = abspath(joinpath(@__DIR__, ".."))
Pkg.activate(ROOT)

if get(ENV, "TRAIN_POLICY_INSTANTIATE", "true") != "false"
    Pkg.instantiate()
end

push!(LOAD_PATH, abspath(joinpath(ROOT, "src")))

using Dates
using Random
using JSON3
using Statistics
using ERCOTPipeline
import ERCOTPipeline: RLTradingEnv, reset!, state, step!, is_done, log_run!, ensure_training_tables!

const DB_PATH = abspath(joinpath(ROOT, "data", "duckdb", "ercot.duckdb"))
const TRAINING_START_DATE = Date(2025, 8, 18)
const NOTE_NT = NamedTuple{(:author,:category,:note),Tuple{String,String,String}}

function clamp_training_date(date::Date)
    upper = Dates.today(Dates.UTC) - Day(1)
    adjusted = date > upper ? upper : date
    return adjusted < TRAINING_START_DATE ? TRAINING_START_DATE : adjusted
end

struct TrainConfig
    dates::Vector{Date}
    hub::String
    nodes::Vector{String}
    episodes::Int
    policy::String
    epsilon::Float64
    risk_budget::Float64
    risk_aversion::Float64
    scenario_delta::Float64
    cone_constraints::Int
    top_constraints::Int
    cvar_alpha::Float64
    max_quantity::Float64
    temperature::Float64
    scenario_probs::NamedTuple{(:base,:up,:down),NTuple{3,Float64}}
    note::Union{Nothing,String}
    seed::Int
end

function parse_dates(str::String)
    items = [strip(s) for s in split(str, ',') if !isempty(strip(s))]
    isempty(items) && return Date[Dates.today(Dates.UTC) - Day(1)]
    return Date.(items)
end

function parse_args(args)
    cfg = Dict{String,Any}(
        "dates" => "",
        "hub" => "HB_HOUSTON",
        "nodes" => "HB_WEST,HB_NORTH",
        "episodes" => 1,
        "policy" => "cvar",
        "epsilon" => 0.05,
        "risk_budget" => 1000.0,
        "risk_aversion" => 1.0,
        "scenario_delta" => 15.0,
        "cone_constraints" => 2,
        "top_constraints" => 3,
        "cvar_alpha" => 0.95,
        "max_quantity" => 200.0,
        "temperature" => 50.0,
        "prob_base" => NaN,
        "prob_up" => NaN,
        "prob_down" => NaN,
        "note" => nothing,
        "seed" => 0,
        "help" => false
    )

    i = 1
    while i <= length(args)
        arg = args[i]
        if arg == "--help" || arg == "-h"
            cfg["help"] = true
            break
        elseif startswith(arg, "--")
            key = arg[3:end]
            i += 1
            i > length(args) && error("Missing value for $(arg)")
            val = args[i]
            cfg[key] = val
        else
            error("Unknown argument $(arg)")
        end
        i += 1
    end

    if cfg["help"]
        println("""
train_policy.jl --dates YYYY-MM-DD[,YYYY-MM-DD...] [options]

Options:
  --hub HB_HOUSTON                Hub node
  --nodes HB_WEST,HB_NORTH        Comma-separated target nodes
  --episodes 1                    Episodes per date
  --policy cvar|maxent|eps        Policy mode (eps uses epsilon-greedy)
  --epsilon 0.05                  Exploration rate for epsilon policy
  --risk_budget 1000.0            Risk budget used for base quantity sizing
  --risk_aversion 1.0             Penalty weight against CVaR
  --scenario_delta 15.0           Shock size for scenario_basis_values
  --cone_constraints 2            Constraints used in scenario cone
  --top_constraints 3             Top constraints for predictions
  --cvar_alpha 0.95               CVaR tail probability
  --max_quantity 200.0            Maximum absolute trade quantity
  --temperature 50.0              Softmax temperature for maxent policy
  --prob_base 0.5                 Optional scenario probability override
  --prob_up 0.25                  Optional scenario probability override
  --prob_down 0.25                Optional scenario probability override
  --note "text"                   Optional note stored with the run
  --seed 0                        RNG seed (0 means random)
        """)
        exit(0)
    end

    dates = parse_dates(String(cfg["dates"]))
    nodes = [strip(s) for s in split(String(cfg["nodes"]), ',') if !isempty(strip(s))]
    nodes = isempty(nodes) ? ["HB_WEST", "HB_NORTH"] : nodes
    prob_base = tryparse(Float64, String(cfg["prob_base"]))
    prob_up = tryparse(Float64, String(cfg["prob_up"]))
    prob_down = tryparse(Float64, String(cfg["prob_down"]))
    probs = if prob_base !== nothing && prob_up !== nothing && prob_down !== nothing
        total = prob_base + prob_up + prob_down
        total <= 0 && (total = 1.0)
        (base = prob_base / total, up = prob_up / total, down = prob_down / total)
    else
        (base = 0.5, up = 0.25, down = 0.25)
    end
    note_val = cfg["note"]
    note = note_val === nothing ? nothing : String(note_val)
    seed = tryparse(Int, String(cfg["seed"]))
    seed === nothing && (seed = 0)

    return TrainConfig(
        dates,
        String(cfg["hub"]),
        nodes,
        Int(cfg["episodes"]),
        lowercase(String(cfg["policy"])),
        Float64(cfg["epsilon"]),
        Float64(cfg["risk_budget"]),
        Float64(cfg["risk_aversion"]),
        Float64(cfg["scenario_delta"]),
        Int(cfg["cone_constraints"]),
        Int(cfg["top_constraints"]),
        Float64(cfg["cvar_alpha"]),
        Float64(cfg["max_quantity"]),
        Float64(cfg["temperature"]),
        probs,
        note,
        seed
    )
end

function compute_policy_quantity(state_dict::Dict{Symbol,Any}, policy::String,
                                 risk_aversion::Float64, temperature::Float64,
                                 max_quantity::Float64, epsilon::Float64)
    dir = Float64(get(state_dict, :direction_sign, 1.0))
    base_quantity = Float64(get(state_dict, :base_quantity, 0.0))
    expected_per_unit = Float64(get(state_dict, :expected_per_unit, 0.0))
    cvar_per_unit = Float64(get(state_dict, :cvar_per_unit, 0.0))
    policy_score = expected_per_unit - risk_aversion * abs(cvar_per_unit)

    quantity = if policy == "maxent"
        temp = temperature <= 0 ? 1.0 : temperature
        weight = 1 / (1 + exp(-policy_score / temp))
        base_quantity * weight * dir
    elseif policy == "eps"
        base_quantity * dir
    else
        base_quantity * dir
    end

    if policy == "eps" && rand() < epsilon
        quantity = (2rand() - 1) * max_quantity
    end
    return clamp(quantity, -max_quantity, max_quantity)
end

function run_episode(cfg::TrainConfig, date::Date)
    env = RLTradingEnv(DB_PATH;
        date = date,
        hub = cfg.hub,
        nodes = cfg.nodes,
        top_constraints = cfg.top_constraints,
        scenario_delta = cfg.scenario_delta,
        cone_constraints = cfg.cone_constraints,
        scenario_probs = cfg.scenario_probs,
        cvar_alpha = cfg.cvar_alpha,
        risk_budget = cfg.risk_budget,
        max_quantity = cfg.max_quantity,
        risk_aversion = cfg.risk_aversion)

    current_state = reset!(env)
    total_reward = 0.0
    steps = 0
    while current_state !== nothing
        qty = compute_policy_quantity(current_state, cfg.policy,
                                      cfg.risk_aversion, cfg.temperature,
                                      cfg.max_quantity, cfg.epsilon)
        current_state, reward, done, _ = step!(env, qty)
        total_reward += reward
        steps += 1
        done && break
    end

    notes_vec = cfg.note === nothing ? NOTE_NT[] : [NOTE_NT("train_policy", "note", cfg.note)]
    run_id = log_run!(env; policy = cfg.policy, status = "completed", notes = notes_vec)
    summary_dict = ERCOTPipeline.summary(env)
    return run_id, total_reward, steps, summary_dict, env
end

function main()
    cfg = parse_args(ARGS)
    if cfg.seed != 0
        Random.seed!(cfg.seed)
    end
    ensure_training_tables!(DB_PATH)

    runs = NamedTuple[]
    for date in cfg.dates
        safe_date = clamp_training_date(date)
        for ep in 1:cfg.episodes
            run_id, reward, steps, summary_dict, env = run_episode(cfg, safe_date)
            entry = (
                run_id = run_id,
                date = string(safe_date),
                episode = ep,
                steps = steps,
                reward = reward,
                summary = summary_dict
            )
            push!(runs, entry)
            @info "Episode completed" run_id date = safe_date episode = ep reward steps
        end
    end

    println(JSON3.write(runs; indent = 2))
end

main()
