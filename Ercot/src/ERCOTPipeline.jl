module ERCOTPipeline

include("Device.jl")
include("AssimilationModel.jl")
include("EnKF.jl")

using .Device: AbstractExecutionDevice, CPUDevice, GPUDevice, detect_device
using .AssimilationModel: RTCStateModel, build_rtc_state_model, simulate_ensemble, default_state_labels
using .EnKF: EnKFFilter, build_enkf, update_ensemble!

export AbstractExecutionDevice, CPUDevice, GPUDevice,
       detect_device,
       RTCStateModel, build_rtc_state_model, simulate_ensemble,
       EnKFFilter, build_enkf, update_ensemble!,
       default_state_labels

end
