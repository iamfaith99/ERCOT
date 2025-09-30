module HayekNetOptions

using Random
using Statistics
using StochasticDiffEq
using PythonCall

export price_call_option

"""
    price_call_option(S0, K, r, σ, T; steps, trajectories, seed)

Monte Carlo price for a European call option under geometric Brownian motion.
Returns the discounted mean payoff and the vector of simulated terminal prices.
"""
function price_call_option(
    S0::Real,
    K::Real,
    r::Real,
    σ::Real,
    T::Real;
    steps::Int = 64,
    trajectories::Int = 1024,
    seed::Union{Nothing, Int} = nothing,
)
    S0 > 0 || throw(ArgumentError("S0 must be positive"))
    σ >= 0 || throw(ArgumentError("σ must be nonnegative"))
    T > 0 || throw(ArgumentError("T must be positive"))

    if seed !== nothing
        Random.seed!(seed)
    end

    drift(u, p, t) = r * u
    diffusion(u, p, t) = σ * u

    prob = SDEProblem(drift, diffusion, float(S0), (0.0, float(T)))
    sol = solve(
        prob,
        EM(),
        dt = T / steps,
        saveat = T,
        adaptive = false,
        trajectories = trajectories,
        ensemblealg = EnsembleThreads(),
    )

    terminal = reduce(hcat, sol.u)
    payoffs = max.(terminal .- K, 0.0)
    discounted = exp(-r * T) * payoffs

    return mean(discounted), vec(terminal)
end

# Export for Python (handled by juliacall automatically)

end # module
