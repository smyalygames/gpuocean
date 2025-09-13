from __future__ import annotations
from typing import TypeVar, Generic, Union, TYPE_CHECKING

import numpy as np
import numpy.typing as npt

from ...utils import convert_to_float32

if TYPE_CHECKING:
    from .. import Array3D, GPUStream

T = TypeVar('T', np.float32, np.float64)
data_t = Union[npt.NDArray[T], np.ma.MaskedArray]


class BaseArray3D(Generic[T]):
    """
    A base class that holds 3D data. To be used depending on the GPGPU language.
    """

    def __init__(self, gpu_stream: GPUStream, nx: int, ny: int, nc: int, data: data_t,
                 double_precision=False, integers=False):
        """
        Uploads initial data to the CL device
        """

        self.double_precision = double_precision
        self._host_data = self._convert_to_precision(data)
        self.dtype = self._host_data.dtype

        self.nx = nx
        self.ny = ny
        self.nc = nc
        self.shape = (self.ny, self.nx, self.nc)

        self.bytes_per_float = self._host_data.itemsize

        # Checking the format of the data
        if self.shape[1] != self.nx:
            raise TypeError(f"{self.shape[1]} vs f{str(self.nx)}")
        if self.shape[0] != self.ny:
            raise TypeError(f"{self.shape[0]} vs {str(self.ny)}")
        if self.shape[2] != self.nc:
            raise TypeError(f"{self.shape[2]} vs {str(self.nc)}")
        if data.shape != self.shape:
            raise TypeError(
                f"The shape of the array ({str(data.shape)} does not match the given conditions ({str((self.ny, self.nx, self.nc))}).")

        if (self.bytes_per_float != 4 and not double_precision) and (self.bytes_per_float != 8 and double_precision):
            raise ValueError("Wrong size of data type. It should an array of 32 or 64 bit floats.")

        self.holds_data = True

        self.mask = None
        if np.ma.is_masked(data):
            self.mask = data.mask

    @property
    def pointer(self):
        """
        Gets the pointer for the array on the GPU, that is being managed by this object.
        Returns:
            A pointer for the array on the GPU
        """
        raise NotImplementedError("This function needs to be implemented in a subclass.")

    def upload(self, gpu_stream: GPUStream, data: data_t) -> None:
        """
        Filling the allocated buffer with new data.
        """
        raise NotImplementedError("This function needs to be implemented in a subclass.")

    def copy_buffer(self, gpu_stream: GPUStream, buffer: Array3D) -> None:
        """
        Copying the given device buffer into the already allocated memory
        """
        raise NotImplementedError("This function needs to be implemented in a subclass.")

    def download(self, gpu_stream: GPUStream) -> data_t:
        """
        Enables downloading data from GPU device to Python
        """
        raise NotImplementedError("This function needs to be implemented in a subclass.")

    def release(self) -> None:
        """
        Frees the allocated memory buffers on the GPU
        """
        raise NotImplementedError("This function needs to be implemented in a subclass.")

    def _convert_to_precision(self, data: data_t) -> data_t:
        """
        Converts the ``data`` given to the specified float precision.
        Args:
            data: The array to be converted.
        Returns:
            Either the same array if double precision was specified by the class or an array converted to a 32-bit float.
        """
        if self.double_precision:
            return data
        else:
            return convert_to_float32(data)

    def _check(self, shape: tuple[int, ...], item_size: int) -> None:
        """
        Checks if the 3D array to copy to GPU memory is of a correct format.
        Args:
            shape: The shape of the array, should be a length of 3.
            item_size: The number of bytes per element in the array.
        Returns:
            Returns nothing if the array is correct. Raises a ``ValueError`` if there is something problematic with the array.
        """
        if len(shape) != 3:
            raise ValueError(f"Provided array is not a 3D array, got a {len(shape)} dimensional array instead.")

        ny, nx, nc = shape

        if self.nc != nc:
            raise ValueError(f"Provided data does not match nx (provided: {nc}, expected: {self.nc}).")
        if self.nx != nx:
            raise ValueError(f"Provided data does not match nx (provided: {nx}, expected: {self.nx}).")
        if self.ny != ny:
            raise ValueError(f"Provided data does not match ny halo (provided: {ny}, expected: {self.ny}).")
        if shape != self.shape:
            raise ValueError(f"Provided data has an incorrect shape (provided: {shape}, expected {self.shape}).")
        if self.bytes_per_float != item_size:
            raise ValueError("Size of each item in the provided array does not match the expected size "
                             f"(provided: {item_size} bytes, expected: {self.bytes_per_float} bytes).")
