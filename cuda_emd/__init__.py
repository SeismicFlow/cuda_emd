"""cuda_emd -- EMD, EEMD and CEEMDAN on NVIDIA GPUs, bit-for-bit PyEMD 1.10, PyEMD's API.

Drop-in use:   from cuda_emd import EMD, EEMD, CEEMDAN        (instead of: from PyEMD import EMD, EEMD, CEEMDAN)
Batch use:     from cuda_emd import GPUBatchEMD, GPUBatchEEMD, GPUBatchCEEMDAN
"""
from .kernels import EMDKernelConfig, ERR_NOEXT, ERR_SPLINE, ERR_CAP, ERR_NONTERM
from .emd import EMD, GPUBatchEMD, EMDOverflowWarning
from .eemd import EEMD, GPUBatchEEMD
from .ceemdan import CEEMDAN, GPUBatchCEEMDAN

__all__ = ["EMD", "EEMD", "CEEMDAN", "GPUBatchEMD", "GPUBatchEEMD", "GPUBatchCEEMDAN", "EMDKernelConfig",
           "EMDOverflowWarning", "Visualisation"]


def __getattr__(name):
    # PyEMD's plotting helper works unchanged on these classes (it only calls get_imfs_and_residue()).
    if name == "Visualisation":
        from PyEMD import Visualisation
        return Visualisation
    raise AttributeError(f"module 'cuda_emd' has no attribute {name!r}")
