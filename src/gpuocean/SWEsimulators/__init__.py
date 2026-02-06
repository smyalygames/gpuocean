from enum import Enum

from .CDKLM16 import CDKLM16
from .CombinedCDKLM16 import CombinedCDKLM16
from .CTCS import CTCS
from .FBL import FBL
from .GPUOceanSim import GPUOceanSim
from .KP07 import KP07
from .ModelErrorKL import ModelErrorKL
from .OceanStateNoise import OceanStateNoise


class SimulatorType(Enum):
    CDKLM16 = CDKLM16
    CTCS = CTCS
    FBL = FBL
    KP07 = KP07
