from cuda.bindings import driver, nvrtc

import re
import os
import numpy as np

from enum import IntEnum


class ModelType(IntEnum):
    OPDT = 0
    CSA = 1
    PROB = 2
    PTT = 3


# We extract REAL_DTYPE, MAX_SLINE_LEN from globals.h
# Maybe there is a more elegant way of doing this?
dir_path = os.path.dirname(os.path.abspath(__file__))
globals_path = os.path.join(dir_path, "globals.h")
with open(globals_path, 'r') as f:
    content = f.read()

defines = dict(re.findall(r"#define\s+(\w+)\s+([^\s/]+)", content))
REAL_SIZE = int(defines["REAL_SIZE"])
REAL3_SIZE = 3 * REAL_SIZE
if REAL_SIZE == 4:
    REAL_DTYPE = np.float32
    REAL3_DTYPE = np.dtype([('x', np.float32),
                            ('y', np.float32),
                            ('z', np.float32)])
elif REAL_SIZE == 8:
    REAL_DTYPE = np.float64
    REAL3_DTYPE = np.dtype([('x', np.float64),
                            ('y', np.float64),
                            ('z', np.float64)])
else:
    raise NotImplementedError(f"Unsupported REAL_SIZE={REAL_SIZE} in globals.h")
MAX_SLINE_LEN = int(defines["MAX_SLINE_LEN"])
THR_X_SL = int(defines["THR_X_SL"])
THR_X_BL = int(defines["THR_X_BL"])
EXCESS_ALLOC_FACT = int(defines["EXCESS_ALLOC_FACT"])


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
