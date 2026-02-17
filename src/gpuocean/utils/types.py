from __future__ import annotations
from typing import TYPE_CHECKING, Union

# Create Simulator types
if TYPE_CHECKING:
    from gpuocean.SWEsimulators import (
        CDKLM16,
        CombinedCDKLM16,
        CTCS,
        FBL,
        GPUOceanSim,
        KP07,
        ModelErrorKL,
        OceanStateNoise
    )

AnySimulator = Union[CDKLM16, CombinedCDKLM16, CTCS, FBL, GPUOceanSim, KP07, ModelErrorKL, OceanStateNoise]
"""
Union of all the runnable simulators based on the base Simulator class.
"""
