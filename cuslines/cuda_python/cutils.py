from cuda.bindings import driver, nvrtc

import numpy as np
import ctypes

from enum import IntEnum

from cuslines.cuda_python._globals import *


class ModelType(IntEnum):
    OPDT = 0
    CSA = 1
    PROB = 2
    PTT = 3

REAL3_SIZE = 3 * REAL_SIZE
if REAL_SIZE == 4:
    REAL_DTYPE = np.float32
    REAL3_DTYPE = np.dtype([('x', np.float32),
                            ('y', np.float32),
                            ('z', np.float32)], align=True)
    REAL_DTYPE_AS_STR = "float"
    REAL3_DTYPE_AS_STR = "float3"
    REAL_DTYPE_AS_CTYPE = ctypes.c_float
elif REAL_SIZE == 8:
    REAL_DTYPE = np.float64
    REAL3_DTYPE = np.dtype([('x', np.float64),
                            ('y', np.float64),
                            ('z', np.float64)], align=True)
    REAL_DTYPE_AS_STR = "double"
    REAL3_DTYPE_AS_STR = "double3"
    REAL_DTYPE_AS_CTYPE = ctypes.c_double
else:
    raise NotImplementedError(f"Unsupported REAL_SIZE={REAL_SIZE} in globals.h")
BLOCK_Y = THR_X_BL//THR_X_SL
DEV_PTR = object

def _cudaGetErrorEnum(error):
    if isinstance(error, driver.CUresult):
        err, name = driver.cuGetErrorName(error)
        return name if err == driver.CUresult.CUDA_SUCCESS else "<unknown>"
    elif isinstance(error, nvrtc.nvrtcResult):
        return nvrtc.nvrtcGetErrorString(error)[1]
    else:
        raise RuntimeError('Unknown error type: {}'.format(error))

def checkCudaErrors(result):
    if result[0].value:
        raise RuntimeError("CUDA error code={}({})".format(result[0].value, _cudaGetErrorEnum(result[0])))
    if len(result) == 1:
        return None
    elif len(result) == 2:
        return result[1]
    else:
        return result[1:]

def div_up(a, b):
    return (a + b - 1) // b
