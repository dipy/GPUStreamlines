import importlib.util
from pathlib import Path
import numpy as np

# Import _globals.py directly (bypasses cuslines.cuda_python.__init__
# which would trigger CUDA imports).
_globals_path = Path(__file__).resolve().parent.parent / "cuda_python" / "_globals.py"
_spec = importlib.util.spec_from_file_location("_globals", str(_globals_path))
_globals_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_globals_mod)

MAX_SLINE_LEN = _globals_mod.MAX_SLINE_LEN
PMF_THRESHOLD_P = _globals_mod.PMF_THRESHOLD_P
if _globals_mod.REAL_SIZE == 4:
    REAL_DTYPE = np.float32
elif _globals_mod.REAL_SIZE == 8:
    REAL_DTYPE = np.float64
else:
    raise NotImplementedError(f"Unsupported REAL_SIZE={_globals_mod.REAL_SIZE}")
