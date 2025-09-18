module EnKF

using LinearAlgebra
using Random
using Statistics

import ..Device: AbstractExecutionDevice, CPUDevice, GPUDevice, to_device_array

export EnKFFilter, build_enkf, update_ensemble!

struct EnKFFilter
    observation_matrix::Matrix{Float64}
    observation_cov::Matrix{Float64}
    rng::Random.AbstractRNG
end

function build_enkf(; H::AbstractMatrix, R::AbstractMatrix, rng::Random.AbstractRNG = Random.default_rng())
    return EnKFFilter(Matrix{Float64}(H), Matrix{Float64}(R), rng)
end

function _subtract_mean!(X)
    μ = mean(X; dims=2)
    X .-= μ
    return μ, X
end

function update_ensemble!(filter::EnKFFilter, ensemble::Matrix{Float64}, observation::AbstractVector)
    m = size(ensemble, 2)
    H = filter.observation_matrix
    R = filter.observation_cov

    Y = H * ensemble
    μx, Xc = _subtract_mean!(copy(ensemble))
    μy, Yc = _subtract_mean!(copy(Y))

    Cxy = (Xc * Yc') / (m - 1)
    Cyy = (Yc * Yc') / (m - 1) + R
    K = Cxy * inv(Cyy)

    obs = Vector{Float64}(observation)
    for j in 1:m
        ε = randn(filter.rng, size(R, 1))
        innovation = (obs + ε) - Y[:, j]
        ensemble[:, j] += K * innovation
    end

    return ensemble
end

end
