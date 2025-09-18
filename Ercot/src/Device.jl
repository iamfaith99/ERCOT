module Device

using Adapt

export AbstractExecutionDevice, CPUDevice, GPUDevice, detect_device, to_device_array

abstract type AbstractExecutionDevice end

struct CPUDevice <: AbstractExecutionDevice end
struct GPUDevice <: AbstractExecutionDevice
    backend::Symbol
end

const _cuda_available = Ref(false)

function _try_load_cuda!()
    return _cuda_available[] ? true : begin
        try
            @eval begin
                import CUDA
            end
            _cuda_available[] = CUDA.has_cuda() && CUDA.functional()
        catch
            _cuda_available[] = false
        end
        _cuda_available[]
    end
end

function detect_device()
    if _try_load_cuda!()
        return GPUDevice(:cuda)
    else
        return CPUDevice()
    end
end

function to_device_array(::CPUDevice, x)
    return x
end

function to_device_array(device::GPUDevice, x)
    if !_cuda_available[]
        error("CUDA backend is not available even though GPUDevice was constructed")
    end
    return Adapt.adapt(CUDA.CuArray, x)
end

end
