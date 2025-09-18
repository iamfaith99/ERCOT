#!/usr/bin/env julia
using Pkg
Pkg.activate(abspath(joinpath(@__DIR__, "..")))

using Logging
using Printf

push!(LOAD_PATH, abspath(joinpath(@__DIR__, "../src")))

include(joinpath(@__DIR__, "../src/ERCOTPipeline.jl"))
using .ERCOTPipeline

const DB_PATH = abspath(joinpath(@__DIR__, "..", "data", "duckdb", "ercot.duckdb"))

function parse_env(name::String, default::T, parser::Function) where {T}
    raw = get(ENV, name, string(default))
    try
        return parser(raw)
    catch err
        @warn "Failed to parse env; using default" name raw default err
        return default
    end
end

function sleep_if_needed(iteration::Int, iterations::Int, pause::Int)
    (iterations <= 0 || iteration < iterations) && pause > 0 && sleep(pause)
end

function print_summary(summary)
    println("Predicted congestion prices:")
    for (node, payload) in summary[:nodes]
        println(rpad(node, 20), @sprintf("%8.2f", payload[:predicted_price]))
    end
    println("\nEvent price vector:")
    for (event, price) in sort(collect(summary[:event_prices]); by = x -> x[2], rev = true)
        println(rpad(event, 35), @sprintf("%6.3f", price))
    end
    println("\nMetadata:")
    md = summary[:metadata]
    println("run_ts=" * get(md, :run_ts, ""))
    println("improvement_rmse=" * string(get(md, :improvement_rmse, missing)))
    println("baseline_rmse=" * string(get(md, :baseline_rmse, missing)))
end

function main()
    iterations = parse_env("SCENARIO_ITERATIONS", 1, x -> parse(Int, x))
    sleep_seconds = parse_env("SCENARIO_SLEEP_SECONDS", 60, x -> parse(Int, x))
    liquidity = parse_env("SCENARIO_LIQUIDITY", 5.0, x -> parse(Float64, x))
    source = get(ENV, "SCENARIO_EVENT_SOURCE", "ptdf_scenario")
    continue_print = parse_env("SCENARIO_PRINT", 1, x -> parse(Int, x)) != 0

    iteration = 1
    while iterations <= 0 || iteration <= iterations
        summary = scenario_summary(DB_PATH; b = liquidity)
        persisted = persist_event_prices!(DB_PATH, summary; source = source)

        println("[Tick $(iteration)] Stored $(persisted) event prices for $(summary[:timestamp]) (b=$(summary[:liquidity])) from source $(source).")
        continue_print && print_summary(summary)

        sleep_if_needed(iteration, iterations, sleep_seconds)
        iteration += 1
    end
end

main()
