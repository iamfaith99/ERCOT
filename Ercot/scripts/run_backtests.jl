#!/usr/bin/env julia

using Pkg

const ROOT = abspath(joinpath(@__DIR__, ".."))
Pkg.activate(ROOT)

if get(ENV, "BACKTEST_INSTANTIATE", "true") != "false"
    Pkg.instantiate()
end

push!(LOAD_PATH, abspath(joinpath(ROOT, "src")))

using Dates
using JSON3
using ERCOTPipeline

const DB_PATH = abspath(joinpath(ROOT, "data", "duckdb", "ercot.duckdb"))

function run_backtest(; date = get(ENV, "BACKTEST_DATE", ""),
                        top_constraints = parse(Int, get(ENV, "BACKTEST_TOP_CONSTRAINTS", "3")),
                        liquidity = parse(Float64, get(ENV, "BACKTEST_LIQUIDITY", "5.0")))
    base_date = isempty(date) ? Dates.today(Dates.UTC) - Day(1) : Date(date)
    summary = scenario_summary(DB_PATH; top_constraints = top_constraints, b = liquidity)
    result = Dict(
        :requested_date => string(base_date),
        :summary_date => get(summary, :timestamp, ""),
        :liquidity => liquidity,
        :top_constraints => top_constraints,
        :nodes => summary[:nodes],
        :event_prices => summary[:event_prices],
        :metadata => summary[:metadata]
    )
    println(JSON3.write(result; indent = 2))
end

run_backtest()
