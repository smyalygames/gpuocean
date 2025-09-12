from __future__ import annotations
from typing import TypeVar, Generic, Union, TYPE_CHECKING

import numpy as np
import numpy.typing as npt

from ...utils import convert_to_float32

if TYPE_CHECKING:
    from .. import GPUStream, Array2D

T = TypeVar('T', np.float32, np.float64)
data_t = Union[npt.NDArray[T], np.ma.MaskedArray]


class BaseArray2D(Generic[T]):
    """
    A base class that holds 2D data. To be used depending on the GPGPU language.
    """

    def __init__(self, gpu_stream: GPUStream, nx: int, ny: int, halo_x: int, halo_y: int, data: data_t,
                 asym_halo: list[int] = None, double_precision=False, integers=False):
        """
        Uploads initial data to the CUDA device
        """

        self.double_precision = double_precision
        self.integers = integers
        self.__host_data = self.__convert_to_precision(data)
        self.dtype = self.__host_data.dtype

        self.nx = nx
        self.ny = ny
        self.nx_halo = nx + 2 * halo_x
        self.ny_halo = ny + 2 * halo_y
        if asym_halo is not None and len(asym_halo) == 4:
            # asymHalo = [halo_north, halo_east, halo_south, halo_west]
            self.nx_halo = nx + asym_halo[1] + asym_halo[3]
            self.ny_halo = ny + asym_halo[0] + asym_halo[2]

        self.shape = (self.nx_halo, self.ny_halo)

        # Make sure data is in proper format
        if self.__host_data.shape != self.shape[::-1]:
            raise ValueError(
                f"Wrong shape of data {str(self.__host_data.shape)} vs {str((self.ny, self.nx))} / "
                + f"{str((self.ny_halo, self.nx_halo))}")

        self.bytes_per_float = self.__host_data.itemsize

        if (self.bytes_per_float != 4 and not double_precision) and (self.bytes_per_float != 8 and double_precision):
            raise ValueError("Wrong size of data type. It should an array of 32 or 64 bit floats.")

        if np.isfortran(data):
            raise TypeError("Wrong datatype (Fortran, expected C)")

        self.holds_data = True

        self.mask: Union[np.ma.MaskedArray, None] = None
        if np.ma.is_masked(data):
            self.mask = data.mask

    def upload(self, gpu_stream: GPUStream, data: data_t) -> None:
        """
        Filling the allocated buffer with new data.
        """
        raise NotImplementedError("This function needs to be implemented in a subclass.")

    def copy_buffer(self, gpu_stream: GPUStream, buffer: Array2D) -> None:
        """
        Copying the given device buffer into the already allocated memory
        """
        raise NotImplementedError("This function needs to be implemented in a subclass.")

    def download(self, gpu_stream: GPUStream) -> data_t:
        """
        Enables downloading data from CUDA device to Python
        """
        raise NotImplementedError("This function needs to be implemented in a subclass.")

    def release(self) -> None:
        """
        Frees the allocated memory buffers on the GPU
        """
        raise NotImplementedError("This function needs to be implemented in a subclass.")

    def __convert_to_precision(self, data: data_t) -> data_t:
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

    def __check(self, shape: tuple[int, ...], item_size: int) -> None:
        """
        Checks if the 2D array to copy to GPU memory is of a correct format.
        Args:
            shape: The shape of the array, should be a length of 2.
            item_size: The number of bytes per element in the array.
        Returns:
            Returns nothing if the array is correct. Raises a ``ValueError`` if there is something problematic with the array.
        """
        if len(shape) != 2:
            raise ValueError(f"Provided array is not a 2D array, got a {len(shape)} dimensional array instead.")

        ny_halo, nx_halo = shape

        if self.nx_halo != nx_halo:
            raise ValueError(f"Provided data does not match nx halo (provided: {nx_halo}, expected: {self.nx_halo}).")
        if self.ny_halo != ny_halo:
            raise ValueError(f"Provided data does not match ny halo (provided: {ny_halo}, expected: {self.ny_halo}).")
        if shape != self.shape:
            raise ValueError(f"Provided data has an incorrect shape (provided: {shape}, expected {self.shape}).")
        if self.bytes_per_float != item_size:
            raise ValueError("Size of each item in the provided array does not match the expected size "
                             f"(provided: {item_size} bytes, expected: {self.bytes_per_float} bytes).")
