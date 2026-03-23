from enum import IntEnum

import numpy as np
import logging
from cuda.bindings import driver, nvrtc
from cuda.bindings import runtime
from cuda.bindings.runtime import cudaMemcpyKind

from cuslines.cuda_python._globals import *

logger = logging.getLogger("GPUStreamlines")


class ModelType(IntEnum):
    OPDT = 0
    CSA = 1
    PROB = 2
    PTT = 3


REAL3_SIZE = 3 * REAL_SIZE
if REAL_SIZE == 4:
    REAL_DTYPE = np.float32
    REAL3_DTYPE = np.dtype(
        [("x", np.float32), ("y", np.float32), ("z", np.float32)], align=True
    )
    REAL_DTYPE_AS_STR = "float"
    REAL3_DTYPE_AS_STR = "float3"
elif REAL_SIZE == 8:
    REAL_DTYPE = np.float64
    REAL3_DTYPE = np.dtype(
        [("x", np.float64), ("y", np.float64), ("z", np.float64)], align=True
    )
    REAL_DTYPE_AS_STR = "double"
    REAL3_DTYPE_AS_STR = "double3"
else:
    raise NotImplementedError(f"Unsupported REAL_SIZE={REAL_SIZE} in globals.h")
BLOCK_Y = THR_X_BL // THR_X_SL
DEV_PTR = object


def _cudaGetErrorEnum(error):
    if isinstance(error, driver.CUresult):
        err, name = driver.cuGetErrorName(error)
        return name if err == driver.CUresult.CUDA_SUCCESS else "<unknown>"
    elif isinstance(error, nvrtc.nvrtcResult):
        return nvrtc.nvrtcGetErrorString(error)[1]
    else:
        raise RuntimeError("Unknown error type: {}".format(error))


def checkCudaErrors(result, hard_error=True):
    if result[0].value:
        if hard_error:
            raise RuntimeError(
                "CUDA error code={}({})".format(
                    result[0].value, _cudaGetErrorEnum(result[0])
                )
            )
        else:
            logger.warning(
                "CUDA error code={}({})".format(
                    result[0].value, _cudaGetErrorEnum(result[0])
                )
            )
    if len(result) == 1:
        return None
    elif len(result) == 2:
        return result[1]
    else:
        return result[1:]


def div_up(a, b):
    return (a + b - 1) // b


def allocate_texture(data, address_mode="clamp"):
    channelDesc = checkCudaErrors(
        runtime.cudaCreateChannelDesc(
            32, 0, 0, 0, runtime.cudaChannelFormatKind.cudaChannelFormatKindFloat
        )
    )

    dim0, dim1, dim2 = data.shape
    extent = runtime.make_cudaExtent(dim2, dim1, dim0)
    dataf_array = checkCudaErrors(runtime.cudaMalloc3DArray(channelDesc, extent, 0))

    copyParams = runtime.cudaMemcpy3DParms()
    copyParams.srcPtr = runtime.make_cudaPitchedPtr(
        data.ctypes.data,
        dim2 * 4,
        dim2,
        dim1,
    )

    copyParams.dstArray = dataf_array
    copyParams.extent = extent
    copyParams.kind = cudaMemcpyKind.cudaMemcpyHostToDevice
    checkCudaErrors(runtime.cudaMemcpy3D(copyParams))

    resDesc = runtime.cudaResourceDesc()
    resDesc.resType = runtime.cudaResourceType.cudaResourceTypeArray
    resDesc.res.array.array = dataf_array

    texDesc = runtime.cudaTextureDesc()
    if address_mode == "clamp":
        address_mode = runtime.cudaTextureAddressMode.cudaAddressModeClamp
    elif address_mode == "border":
        address_mode = runtime.cudaTextureAddressMode.cudaAddressModeBorder
        texDesc.borderColor[0] = -1.0
        texDesc.borderColor[1] = -1.0
        texDesc.borderColor[2] = -1.0
    else:
        raise ValueError(f"Unsupported address_mode: {address_mode}")
    texDesc.addressMode[0] = address_mode
    texDesc.addressMode[1] = address_mode
    texDesc.addressMode[2] = address_mode
    texDesc.filterMode = runtime.cudaTextureFilterMode.cudaFilterModeLinear
    texDesc.readMode = runtime.cudaTextureReadMode.cudaReadModeElementType
    texDesc.normalizedCoords = 0

    texObj = checkCudaErrors(runtime.cudaCreateTextureObject(resDesc, texDesc, None))
    return texObj, dataf_array
