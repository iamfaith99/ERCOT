#!/usr/bin/env julia
using Pkg
Pkg.activate(abspath(joinpath(@__DIR__, "..")))

push!(LOAD_PATH, abspath(joinpath(@__DIR__, "../src")))

include(joinpath(@__DIR__, "../src/ERCOTPipeline.jl"))
using .ERCOTPipeline
using LinearAlgebra
using Statistics

function main()
    device = detect_device()
    @info "Selected execution device" device
    model = build_rtc_state_model(device; Δt = 300.0)
    X = simulate_ensemble(model; ensemble_size = 16)

    H = Matrix{Float64}(I, 4, 4)
    R = (50.0^2) .* Matrix{Float64}(I, 4, 4)
    enkf = build_enkf(H = H, R = R)

    mean_state = mean(X; dims=2)
    observation = vec(mean_state) .+ randn(enkf.rng, 4) .* 25.0

    X_post = update_ensemble!(enkf, X, observation)
    posterior_mean = vec(mean(X_post; dims=2))

    @info "Observation" observation
    @info "Posterior mean" posterior_mean
end

main()
