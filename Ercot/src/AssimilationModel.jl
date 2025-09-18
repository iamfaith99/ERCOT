module AssimilationModel

using LinearAlgebra
using SciMLBase
using OrdinaryDiffEq
using Random

import ..Device: AbstractExecutionDevice, CPUDevice, GPUDevice, to_device_array, detect_device

export RTCStateModel, build_rtc_state_model, simulate_ensemble, default_state_labels

const default_state_labels = (:load, :wind, :solar, :thermal_outage)

struct RTCParameters
    equilibrium::Vector{Float64}
    decay_rates::Vector{Float64}
    process_scale::Vector{Float64}
end

struct RTCStateModel{Tprob,Tparams}
    device::AbstractExecutionDevice
    prob::Tprob
    params::Tparams
    dt_seconds::Float64
    labels::NTuple{4,Symbol}
end

function _default_parameters()
    equilibrium = [45_000.0, 12_000.0, 8_000.0, 4_500.0]
    decay_rates = [0.02, 0.12, 0.18, 0.08]
    process_scale = [200.0, 250.0, 150.0, 100.0]
    RTCParameters(equilibrium, decay_rates, process_scale)
end

_initial_state(params::RTCParameters) = params.equilibrium

function _rtc_rhs!(du, u, p::RTCParameters, t)
    @inbounds for i in eachindex(u)
        du[i] = -p.decay_rates[i] * (u[i] - p.equilibrium[i])
    end
    return nothing
end

function _device_noise(device::AbstractExecutionDevice, params::RTCParameters)
    vec = to_device_array(device, params.process_scale)
    return vec
end

function build_rtc_state_model(device::AbstractExecutionDevice=detect_device(); Δt = 300.0, params = _default_parameters(), labels = default_state_labels)
    u0_cpu = _initial_state(params)
    u0_dev = to_device_array(device, u0_cpu)
    rhs! = (du,u,p,t) -> _rtc_rhs!(du,u,p,t)
    prob = ODEProblem(rhs!, u0_dev, (0.0, Δt), params)
    RTCStateModel(device, prob, params, Δt, labels)
end

function _sample_noise(device::CPUDevice, rng::AbstractRNG, scale::AbstractVector)
    return scale .* randn(rng, length(scale))
end

function _sample_noise(device::GPUDevice, rng::AbstractRNG, scale::AbstractVector)
    length(scale) == 0 && return scale
    CUDA = Base.require(Base.PkgId(Base.UUID("052768ef-5323-5732-b1bb-66c8b64840ba"), "CUDA"))
    noise = CUDA.randn(eltype(scale), length(scale))
    return noise .* scale
end

function simulate_ensemble(model::RTCStateModel; ensemble_size = 32, dt = nothing, rng = Random.default_rng())
    integrator = Tsit5()
    horizon = isnothing(dt) ? model.dt_seconds : dt
    noise_scale = _device_noise(model.device, model.params)

    function prob_func(prob, i, repeat)
        perturbation = _sample_noise(model.device, rng, noise_scale)
        u0 = prob.u0 .+ perturbation
        remake(prob, u0 = u0)
    end

    ensemble_prob = EnsembleProblem(model.prob; prob_func)
    sol = solve(ensemble_prob, integrator; trajectories = ensemble_size, saveat = horizon)
    return sol
end

end
