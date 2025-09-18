#!/usr/bin/env julia
using Pkg
Pkg.activate(abspath(joinpath(@__DIR__, "..")))

push!(LOAD_PATH, abspath(joinpath(@__DIR__, "../src")))

include(joinpath(@__DIR__, "../src/ERCOTPipeline.jl"))
using .ERCOTPipeline
using LinearAlgebra
using Statistics
using Dates

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

    graph = EventGraph()
    interval = (now(), now() + Minute(5))
    add_event!(graph, EventNode(:load_above_equilibrium;
                                scope = (variable = :load,
                                         relation = :gt,
                                         threshold = model.params.equilibrium[1],
                                         interval = interval,
                                         product = :rt_energy)))
    add_event!(graph, EventNode(:wind_below_equilibrium;
                                scope = (variable = :wind,
                                         relation = :lt,
                                         threshold = model.params.equilibrium[2],
                                         interval = interval,
                                         product = :rt_energy)))
    add_event!(graph, EventNode(:scarcity_combo;
                                parents = [:load_above_equilibrium, :wind_below_equilibrium],
                                scope = (aggregator = :all_true,
                                         product = :rt_energy,
                                         tags = [:scarcity])))

    event_priors = ensemble_event_priors(model, X_post, graph)
    @info "Event priors" event_priors

    market = initialize_market(event_priors; b = 5.0)
    @info "LMSR prices" state_prices(market)
end

main()
