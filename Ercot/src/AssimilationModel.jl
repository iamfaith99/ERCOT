module AssimilationModel

using LinearAlgebra
using Random
using Statistics: mean
using SciMLBase: ODEProblem, EnsembleProblem, remake, solve
using OrdinaryDiffEq: Tsit5, EnsembleSerial

import ..Device: AbstractExecutionDevice, CPUDevice, GPUDevice, to_device_array, detect_device
import ..EventAlgebra: EventGraph, topological_order, get_event

export RTCStateModel, build_rtc_state_model, simulate_ensemble, default_state_labels,
       ensemble_event_priors, evaluate_event_priors

const default_state_labels = (:load, :wind, :solar, :thermal_outage)

struct RTCParameters{Teq,Tdecay,Tscale}
    equilibrium::Teq
    decay_rates::Tdecay
    process_scale::Tscale
end

struct RTCStateModel{Tparams,Tdeviceparams,Tprob}
    device::AbstractExecutionDevice
    params::Tparams
    device_params::Tdeviceparams
    dt_seconds::Float64
    labels::NTuple{4,Symbol}
    base_problem::Tprob
end

function _default_parameters()
    equilibrium = [45_000.0, 12_000.0, 8_000.0, 4_500.0]
    decay_rates = [0.02, 0.12, 0.18, 0.08]
    process_scale = [200.0, 250.0, 150.0, 100.0]
    RTCParameters(equilibrium, decay_rates, process_scale)
end

_initial_state(params::RTCParameters) = params.equilibrium

function _rtc_rhs!(du, u, p::RTCParameters, t)
    @. du = -p.decay_rates * (u - p.equilibrium)
    return nothing
end

function _device_parameters(device::AbstractExecutionDevice, params::RTCParameters)
    equilibrium = to_device_array(device, copy(params.equilibrium))
    decay_rates = to_device_array(device, copy(params.decay_rates))
    process_scale = to_device_array(device, copy(params.process_scale))
    return RTCParameters(equilibrium, decay_rates, process_scale)
end

function build_rtc_state_model(device::AbstractExecutionDevice = detect_device();
                                Δt = 300.0,
                                params = _default_parameters(),
                                labels = default_state_labels)
    params_device = _device_parameters(device, params)
    u0 = copy(params_device.equilibrium)
    tspan = (0.0, Δt)
    base_problem = ODEProblem(_rtc_rhs!, u0, tspan, params_device)
    return RTCStateModel(device, params, params_device, Δt, labels, base_problem)
end

function _sample_noise(device::CPUDevice, rng::AbstractRNG, scale::AbstractVector)
    return scale .* randn(rng, length(scale))
end

function _sample_noise(device::GPUDevice, rng::AbstractRNG, scale::AbstractVector)
    length(scale) == 0 && return scale
    noise_cpu = randn(rng, length(scale))
    noise_dev = to_device_array(device, noise_cpu)
    scale_dev = scale isa Array ? to_device_array(device, scale) : scale
    return noise_dev .* scale_dev
end

function _ensemble_backend(::CPUDevice)
    return EnsembleSerial()
end

const _diffeqgpu_available = Ref(false)

function _try_load_diffeqgpu!()
    return _diffeqgpu_available[] ? true : begin
        try
            @eval import DiffEqGPU
            _diffeqgpu_available[] = true
        catch
            _diffeqgpu_available[] = false
        end
        _diffeqgpu_available[]
    end
end

function _ensemble_backend(::GPUDevice)
    if _try_load_diffeqgpu!()
        return DiffEqGPU.EnsembleGPUArray()
    else
        return EnsembleSerial()
    end
end

function simulate_ensemble(model::RTCStateModel;
                           ensemble_size = 32,
                           dt = model.dt_seconds,
                           steps = 1,
                           rng = Random.default_rng(),
                           alg = Tsit5())
    total_time = dt * steps
    prob = remake(model.base_problem;
                  u0 = copy(model.device_params.equilibrium),
                  tspan = (0.0, total_time))

    prob_func = function (inner_prob, _i, _repeat)
        noise = _sample_noise(model.device, rng, model.device_params.process_scale)
        perturbed = inner_prob.u0 .+ noise
        return remake(inner_prob; u0 = perturbed)
    end

    ensemble_prob = EnsembleProblem(prob; prob_func = prob_func)
    backend = _ensemble_backend(model.device)
    sol = solve(ensemble_prob, alg, backend;
                trajectories = ensemble_size,
                save_everystep = false)

    state_dim = length(model.params.equilibrium)
    X = Matrix{Float64}(undef, state_dim, ensemble_size)

    for j in 1:ensemble_size
        traj = sol[j]
        X[:, j] = Array(traj.u[end])
    end

    return X
end

const _RELATION_FUNCS = Dict{Symbol,Function}(
    :gt => (x, thr) -> x > thr,
    :ge => (x, thr) -> x >= thr,
    :lt => (x, thr) -> x < thr,
    :le => (x, thr) -> x <= thr,
    :eq => (x, thr) -> x == thr,
    :ne => (x, thr) -> x != thr,
)

const _EPSILON = 1e-9

# Heuristic combiners to derive new event probabilities from parent events.
# Available keys: :all_true, :any_true, :mean, :min, :max, :bounded_sum, :weighted_sum,
# :share (requires :target), :ratio (requires :numerator/:denominator), and
# :conditional (requires :target/:given, optional :joint).
const _BUILTIN_AGGREGATORS = Dict{Symbol,Function}(
    :all_true => (parents, order, scope) -> prod(parents[parent] for parent in order),
    :any_true => (parents, order, scope) -> begin
        isempty(order) && return 0.0
        prod_not = prod(1 - parents[parent] for parent in order)
        return 1 - prod_not
    end,
    :mean => (parents, order, scope) -> isempty(order) ? 0.0 : mean(parents[parent] for parent in order),
    :min => (parents, order, scope) -> isempty(order) ? 0.0 : minimum(parents[parent] for parent in order),
    :max => (parents, order, scope) -> isempty(order) ? 0.0 : maximum(parents[parent] for parent in order),
    :bounded_sum => (parents, order, scope) -> sum(parents[parent] for parent in order),
    :weighted_sum => (parents, order, scope) -> begin
        weights = get(scope, :weights) do
            error(":weighted_sum aggregator requires :weights in scope")
        end
        if weights isa AbstractVector
            length(weights) == length(order) || error("weights vector must align with parents order")
            total = 0.0
            for (idx, parent_id) in enumerate(order)
                total += weights[idx] * parents[parent_id]
            end
            return total
        elseif weights isa NamedTuple || weights isa Dict
            total = 0.0
            for parent_id in order
                haskey(weights, parent_id) || error("weights missing entry for $(parent_id)")
                total += weights[parent_id] * parents[parent_id]
            end
            return total
        else
            error("Unsupported weights container $(typeof(weights))")
        end
    end,
    :share => (parents, order, scope) -> begin
        target = get(scope, :target) do
            error(":share aggregator requires :target in scope")
        end
        haskey(parents, target) || error(":share aggregator missing parent $(target)")
        total = sum(parents[parent] for parent in order)
        total <= _EPSILON && return 0.0
        return parents[target] / total
    end,
    :ratio => (parents, order, scope) -> begin
        numerator = get(scope, :numerator) do
            error(":ratio aggregator requires :numerator in scope")
        end
        denominator = get(scope, :denominator) do
            error(":ratio aggregator requires :denominator in scope")
        end
        haskey(parents, numerator) || error(":ratio aggregator missing parent $(numerator)")
        haskey(parents, denominator) || error(":ratio aggregator missing parent $(denominator)")
        denom = max(parents[denominator], _EPSILON)
        return parents[numerator] / denom
    end,
    :conditional => (parents, order, scope) -> begin
        target = get(scope, :target) do
            error(":conditional aggregator requires :target in scope")
        end
        condition = get(scope, :given) do
            error(":conditional aggregator requires :given in scope")
        end
        haskey(parents, target) || error(":conditional aggregator missing parent $(target)")
        haskey(parents, condition) || error(":conditional aggregator missing parent $(condition)")
        joint_key = get(scope, :joint, nothing)
        joint_prob = if joint_key === nothing
            min(parents[target], parents[condition])
        else
            haskey(parents, joint_key) || error(":conditional aggregator missing joint parent $(joint_key)")
            parents[joint_key]
        end
        denom = max(parents[condition], _EPSILON)
        return joint_prob / denom
    end,
)

_clamp_probability(p) = clamp(Float64(p), 0.0, 1.0)

function _apply_aggregator(agg_spec, parents::Dict{Symbol,Float64}, order::Vector{Symbol}, scope)
    isempty(order) && error(":aggregator requires at least one parent event")
    value = if agg_spec isa Function
        agg_spec(parents, order, scope)
    elseif agg_spec isa Symbol
        fn = get(_BUILTIN_AGGREGATORS, agg_spec) do
            error("Unknown aggregator $(agg_spec)")
        end
        fn(parents, order, scope)
    else
        error("Unsupported aggregator specification $(typeof(agg_spec))")
    end
    return _clamp_probability(value)
end

function evaluate_event_priors(graph::EventGraph, base_priors::Dict{Symbol,Float64}=Dict{Symbol,Float64}())
    priors = Dict{Symbol,Float64}()
    merge!(priors, base_priors)
    for id in topological_order(graph)
        node = get_event(graph, id)
        if haskey(node.scope, :prior)
            priors[id] = node.scope[:prior]
        elseif haskey(node.scope, :aggregator)
            parent_probs = Dict(parent => priors[parent] for parent in node.parents)
            priors[id] = _apply_aggregator(node.scope[:aggregator], parent_probs, node.parents, node.scope)
        elseif haskey(priors, id)
            priors[id] = priors[id]
        else
            error("Event $(id) missing :prior or :aggregator to evaluate")
        end
        priors[id] = _clamp_probability(priors[id])
    end
    return priors
end

function _label_index(labels::NTuple{N,Symbol}, variable::Symbol) where {N}
    idx = findfirst(==(variable), labels)
    idx === nothing && error("Variable $(variable) not found in model labels $(labels)")
    return idx
end

function _probability_from_samples(relation::Symbol, threshold::Real, samples)
    fn = get(_RELATION_FUNCS, relation) do
        error("Unsupported relation $(relation)")
    end
    hits = count(x -> fn(x, threshold), samples)
    return hits / length(samples)
end

function _event_prior_from_scope(model::RTCStateModel,
                                 ensemble::AbstractMatrix,
                                 node,
                                 priors::Dict{Symbol,Float64})
    scope = node.scope

    if haskey(scope, :prior)
        return scope[:prior]
    elseif haskey(scope, :variable)
        idx = _label_index(model.labels, scope[:variable])
        size(ensemble, 1) >= idx || error("Ensemble matrix incompatible with model labels")
        relation = get(scope, :relation) do
            error("Scope for event $(node.id) requires :relation when :variable is provided")
        end
        threshold = get(scope, :threshold) do
            error("Scope for event $(node.id) requires :threshold to evaluate relation $(relation)")
        end
        samples = view(ensemble, idx, :)
        return _clamp_probability(_probability_from_samples(relation, threshold, samples))
    elseif haskey(scope, :aggregator)
        isempty(node.parents) && error(":aggregator requires parents for event $(node.id)")
        parent_probs = Dict{Symbol,Float64}()
        for parent_id in node.parents
            haskey(priors, parent_id) || error("Parent $(parent_id) prior unavailable for event $(node.id)")
            parent_probs[parent_id] = priors[parent_id]
        end
        return _apply_aggregator(scope[:aggregator], parent_probs, node.parents, scope)
    elseif isempty(node.parents)
        error("Cannot derive prior for event $(node.id); add :prior or :variable metadata")
    else
        error("Event $(node.id) requires custom aggregation; supply :prior or :variable in scope")
    end
end

"""
    ensemble_event_priors(model, ensemble, graph)

Map ensemble posterior samples to Boolean event probabilities. Each
`EventNode` must supply either `scope.prior`, a `(variable, relation,
threshold)` triple, or an `:aggregator` that combines parent events.
Returns a `Dict{Symbol,Float64}` keyed by event id.
"""

function ensemble_event_priors(model::RTCStateModel,
                               ensemble::AbstractMatrix,
                               graph::EventGraph)
    size(ensemble, 1) == length(model.labels) || error("Ensemble rows must match model labels")
    size(ensemble, 2) > 0 || error("Ensemble must contain at least one trajectory")

    priors = Dict{Symbol,Float64}()
    for id in topological_order(graph)
        node = get_event(graph, id)
        priors[id] = _event_prior_from_scope(model, ensemble, node, priors)
    end
    return priors
end

end
