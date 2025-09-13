from dataclasses import dataclass
from typing import Union, TypeVar

import numpy as np
import numpy.typing as npt
from hip import hip
from hip._util.types import Pointer


class __Location:
    location_options = {'src', 'dst'}

    def __init__(self):
        self._location: str | None = None

    @property
    def location(self) -> str:
        """
        The location of where the memory is regarding the memory transfer.
        Returns:
            ``'src'`` or ``'dst'`` depending on if it is the source or destination of the memory copy.
        """
        if self._location is None:
            raise ValueError("The location is not set, make sure to set it before use.")
        return self._location

    @location.setter
    def location(self, value: str):
        """
        The location the memory is in regarding the memory transfer.
        Args:
            value: Can only be ``'src'`` or ``'dst'``.

                - ``'src'`` represents the source of the memory copy.

                - ``'dst'`` represents the destination of the memory copy.
        """
        if value not in self.location_options:
            raise ValueError("An invalid location was parsed")
        self._location = value

    def copy_args(self, *args, **kwargs) -> dict:
        raise NotImplementedError("This function needs to be implemented in a subclass.")


class Host(__Location):
    """
    Represents the host in memory copies on HIP.
    """
    memory_type = hip.hipMemoryType.hipMemoryTypeHost

    T = TypeVar('T', bound=npt.NDArray)

    def __init__(self, array: npt.NDArray[T]):
        """
        Initialises a class used for transferring to/from the host.
        Args:
            array: A numpy array to represent the host array to copy to/from.
        """
        super().__init__()
        self.array = array
        self.pitch = self.array.strides[0]

    def copy_args(self) -> dict[
        str, Union[npt.NDArray[T], int, hip.hipMemoryType]]:
        """
        Creates the arguments required for copying with the device.
        Returns:
            A dictionary representing partial kwargs for the copy to/from the host.
        """
        return {
            f'{self.location}Host': self.array,
            f'{self.location}Pitch': self.pitch,
            f'{self.location}MemoryType': self.memory_type,
        }


class Device(__Location):
    """
    Represents the device in memory copies on HIP.
    """
    memory_type = hip.hipMemoryType.hipMemoryTypeDevice

    def __init__(self, pointer: Pointer, pitch: int, dtype: np.float32 | np.float64, x=0, y=0):
        """
        Initialises a class used for transferring to/from the device.
        Args:
            pointer: A pointer to the device memory location.
            pitch: The pitch of the memory on the device, including the padding, in bytes.
            dtype: numpy dtype of the array element type.
            x: The element on the x dimension in the array on the device.
            y: The element on the y dimension in the array on the device.
        """
        super().__init__()
        self.pointer = pointer
        self.pitch = pitch
        self.x = x
        self.y = y
        self.dtype = dtype

    def copy_args(self) -> dict[str, Union[Pointer, int]]:
        """
        Creates the arguments required for copying with the device.
        Returns:
            A dictionary representing partial kwargs for the copy to/from the device.
        """
        return {
            f'{self.location}Device': self.pointer,
            f'{self.location}Pitch': self.pitch,
            f'{self.location}XInBytes': self.x * self.dtype.itemsize,
            f'{self.location}Y': self.y,
            f'{self.location}MemoryType': self.memory_type,
        }


@dataclass
class Transfer:
    """
    A helper class for managing transfers.
    Args:
        src: The source of the data for the memory copy.
        dst: The destination of the data for the memory copy.
        width: The width of the array in bytes.
        height: The height of the array in elements.
    """
    src: Host | Device
    dst: Host | Device
    width: int
    height: int

    def __post_init__(self):
        if isinstance(self.src, Host) and isinstance(self.dst, Host):
            raise TypeError("Cannot do a Host to Host memory transfer.")

        self.src.location = 'src'
        self.dst.location = 'dst'

    def get_transfer(self):
        """
        Creates a ``hip_Memcpy2D`` object for completing transfers.
        """
        kwargs = {
            'WidthInBytes': self.width,
            'Height': self.height
        }

        kwargs.update(self.src.copy_args())
        kwargs.update(self.dst.copy_args())

        return hip.hip_Memcpy2D(**kwargs)
