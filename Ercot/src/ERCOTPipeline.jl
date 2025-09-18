module ERCOTPipeline

include("Device.jl")
include("AssimilationModel.jl")

using .Device: AbstractExecutionDevice, CPUDevice, GPUDevice, detect_device
using .AssimilationModel: RTCStateModel, build_rtc_state_model, simulate_ensemble, default_state_labels

export AbstractExecutionDevice, CPUDevice, GPUDevice,
       detect_device,
       RTCStateModel, build_rtc_state_model, simulate_ensemble,
       default_state_labels

end
