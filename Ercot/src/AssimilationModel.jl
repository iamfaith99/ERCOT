module AssimilationModel

using LinearAlgebra
using Random

import ..Device: AbstractExecutionDevice, CPUDevice, GPUDevice, to_device_array, detect_device

export RTCStateModel, build_rtc_state_model, simulate_ensemble, default_state_labels

const default_state_labels = (:load, :wind, :solar, :thermal_outage)

struct RTCParameters
    equilibrium::Vector{Float64}
    decay_rates::Vector{Float64}
    process_scale::Vector{Float64}
end

struct RTCStateModel{Tparams}
    device::AbstractExecutionDevice
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

function build_rtc_state_model(device::AbstractExecutionDevice=detect_device(); Δt = 300.0, params = _default_parameters(), labels = default_state_labels)
    RTCStateModel{RTCParameters}(device, params, Δt, labels)
end

function _sample_noise(device::CPUDevice, rng::AbstractRNG, scale::AbstractVector)
    return scale .* randn(rng, length(scale))
end

function _sample_noise(device::GPUDevice, rng::AbstractRNG, scale::AbstractVector)
    length(scale) == 0 && return scale
    noise_cpu = randn(rng, length(scale))
    noise_dev = to_device_array(device, noise_cpu)
    scale_dev = to_device_array(device, scale)
    return noise_dev .* scale_dev
end

function simulate_ensemble(model::RTCStateModel; ensemble_size = 32, dt = model.dt_seconds, steps = 1, rng = Random.default_rng())
    state_dim = length(model.params.equilibrium)
    equilibrium = to_device_array(model.device, copy(model.params.equilibrium))
    scale = model.params.process_scale
    X = Matrix{Float64}(undef, state_dim, ensemble_size)

    for j in 1:ensemble_size
        u = copy(equilibrium)
        perturbation = _sample_noise(model.device, rng, scale)
        u .+= perturbation
        for _ in 1:steps
            du = similar(u)
            _rtc_rhs!(du, u, model.params, 0.0)
            u .+= dt .* du
        end
        X[:, j] = Array(u)
    end

    return X
end

end
