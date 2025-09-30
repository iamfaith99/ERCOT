module HayekNetEnKF

using EnsembleKalmanProcesses
using LinearAlgebra
using Statistics
using PythonCall

export update_enkf

"""
    update_enkf(prior, observations; H, R, inflation, model)

Carry out a single ensemble Kalman filter analysis step.

# Arguments
- `prior::AbstractMatrix`: State ensemble arranged as `state_dim × ensemble_size`.
- `observations::AbstractMatrix`: Observation ensemble arranged as `obs_dim × ensemble_size`.
- `H::AbstractMatrix`: Linear observation operator (defaults to identity when omitted).
- `R::AbstractMatrix`: Observation-error covariance matrix.
- `inflation::Real`: Multiplicative inflation applied to ensemble anomalies.
- `model`: Callable that propagates each ensemble member forward (defaults to identity).

# Returns
Tuple `(analysis_mean, analysis_ensemble)` where `analysis_mean` is a `state_dim × 1` matrix.
"""
function update_enkf(
    prior::AbstractMatrix,
    observations::AbstractMatrix;
    H::AbstractMatrix = Matrix{eltype(prior)}(I, size(prior, 1), size(prior, 1)),
    R::AbstractMatrix = Matrix{eltype(prior)}(I, size(observations, 1), size(observations, 1)),
    inflation::Real = 1.05,
    model::Function = identity,
)
    state_dim, ensemble_size = size(prior)
    obs_dim = size(observations, 1)

    propagated = similar(prior)
    @inbounds for j in 1:ensemble_size
        propagated[:, j] = model(prior[:, j])
    end

    propagated_mean = mean(propagated, dims = 2)
    anomalies = (propagated .- propagated_mean) .* inflation

    innovation = observations .- H * propagated
    innovation_mean = mean(innovation, dims = 2)

    Pf = (anomalies * anomalies') / (ensemble_size - 1)
    S = H * Pf * H' + R
    K = Pf * H' * inv(S)

    analysis = propagated .+ K * (innovation .- innovation_mean)
    analysis_mean = mean(analysis, dims = 2)

    return analysis_mean, analysis
end

# Export for Python (handled by juliacall automatically)

end # module
