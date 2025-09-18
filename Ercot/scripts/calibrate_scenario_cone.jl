#!/usr/bin/env julia
using Pkg
Pkg.activate(abspath(joinpath(@__DIR__, "..")))

using Dates
using Logging

push!(LOAD_PATH, abspath(joinpath(@__DIR__, "../src")))
include(joinpath(@__DIR__, "../src/ERCOTPipeline.jl"))
using .ERCOTPipeline

const DB_PATH = abspath(joinpath(@__DIR__, "..", "data", "duckdb", "ercot.duckdb"))

function parse_override(name::String, default, parser)
    raw = get(ENV, name, nothing)
    raw === nothing && return default
    try
        return parser(raw)
    catch err
        @warn "Failed to parse env; using default" name raw default err
        return default
    end
end

function main()
    lookback_days = parse_override("CALIBRATION_LOOKBACK_DAYS", 14, x -> parse(Int, x))
    top_constraints = parse_override("CALIBRATION_TOP_CONSTRAINTS", 20, x -> parse(Int, x))
    quantile = parse_override("CALIBRATION_QUANTILE", 0.95, x -> parse(Float64, x))
    source = get(ENV, "CALIBRATION_SOURCE", "historical_mu")

    lookback = Dates.Day(lookback_days)
    calibration = calibrate_scenario_cone(DB_PATH;
                                          lookback = lookback,
                                          top_constraints = top_constraints,
                                          quantile = quantile)
    persist_scenario_calibration!(DB_PATH, calibration; source = source)

    @info "Scenario cone calibrated" delta = calibration.scenario_delta tail_prob = calibration.tail_prob per_tail = calibration.per_tail_prob base_prob = calibration.base_prob sample_size = calibration.sample_size lookback_minutes = calibration.lookback_minutes top_constraints = top_constraints quantile = quantile source = source
end

main()
