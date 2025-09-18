module ERCOTPipeline

include("Device.jl")
include("EventAlgebra.jl")
include("AssimilationModel.jl")
include("EnKF.jl")
include("LMSRMarket.jl")

using .Device: AbstractExecutionDevice, CPUDevice, GPUDevice, detect_device
using .AssimilationModel: RTCStateModel, build_rtc_state_model, simulate_ensemble, default_state_labels,
                           ensemble_event_priors
using .EnKF: EnKFFilter, build_enkf, update_ensemble!
using .EventAlgebra: EventNode, EventGraph, add_event!, upsert_event!, has_event, get_event,
                     parents, children, ancestors, topological_order
using .MarketScoring: LMSRMarket, initialize_market, price, state_prices, trade!, shares

export AbstractExecutionDevice, CPUDevice, GPUDevice,
       detect_device,
       RTCStateModel, build_rtc_state_model, simulate_ensemble, ensemble_event_priors,
       EnKFFilter, build_enkf, update_ensemble!,
       default_state_labels,
       EventNode, EventGraph, add_event!, upsert_event!, has_event, get_event,
       parents, children, ancestors, topological_order,
       LMSRMarket, initialize_market, price, state_prices, trade!, shares

end
