#!/usr/bin/env julia
using Pkg
Pkg.activate(abspath(joinpath(@__DIR__, "..")))

push!(LOAD_PATH, abspath(joinpath(@__DIR__, "../src")))

include(joinpath(@__DIR__, "../src/ERCOTPipeline.jl"))
using .ERCOTPipeline

function main()
    device = detect_device()
    @info "Selected execution device" device
    model = build_rtc_state_model(device; Δt = 300.0)
    sol = simulate_ensemble(model; ensemble_size = 8)
    sample = sol[1].u[end]
    @info "Sample trajectory terminal state" sample
end

main()
