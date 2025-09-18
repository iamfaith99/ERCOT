module Device

using Adapt

export AbstractExecutionDevice, CPUDevice, GPUDevice, detect_device, to_device_array

abstract type AbstractExecutionDevice end

struct CPUDevice <: AbstractExecutionDevice end
struct GPUDevice <: AbstractExecutionDevice
    backend::Symbol
end

const _cuda_available  = Ref(false)
const _amd_available   = Ref(false)
const _metal_available = Ref(false)

function _try_load_cuda!()
    return _cuda_available[] ? true : begin
        try
            @eval import CUDA
            _cuda_available[] = CUDA.has_cuda() && CUDA.functional()
        catch
            _cuda_available[] = false
        end
        _cuda_available[]
    end
end

function _try_load_amd!()
    return _amd_available[] ? true : begin
        try
            @eval import AMDGPU
            _amd_available[] = AMDGPU.functional()
        catch
            _amd_available[] = false
        end
        _amd_available[]
    end
end

function _try_load_metal!()
    Sys.isapple() || return false
    return _metal_available[] ? true : begin
        try
            @eval import Metal
            _metal_available[] = Metal.functional()
        catch
            _metal_available[] = false
        end
        _metal_available[]
    end
end

function detect_device()
    if _try_load_cuda!()
        return GPUDevice(:cuda)
    elseif _try_load_amd!()
        return GPUDevice(:amd)
    elseif _try_load_metal!()
        return GPUDevice(:metal)
    else
        return CPUDevice()
    end
end

_ensure_backend_loaded(::CPUDevice) = nothing

function _ensure_backend_loaded(backend::Symbol)
    if backend == :cuda
        _try_load_cuda!() || error("CUDA backend unavailable")
        return CUDA
    elseif backend == :amd
        _try_load_amd!() || error("AMDGPU backend unavailable")
        return AMDGPU
    elseif backend == :metal
        _try_load_metal!() || error("Metal backend unavailable")
        return Metal
    else
        error("Unknown GPU backend: $(backend)")
    end
end

to_device_array(::CPUDevice, x) = x

function to_device_array(device::GPUDevice, x)
    mod = _ensure_backend_loaded(device.backend)
    if device.backend == :cuda
        return Adapt.adapt(mod.CuArray, x)
    elseif device.backend == :amd
        return Adapt.adapt(mod.ROCArray, x)
    elseif device.backend == :metal
        return Adapt.adapt(mod.MtlArray, x)
    else
        error("Unsupported GPU backend: $(device.backend)")
    end
end

end
