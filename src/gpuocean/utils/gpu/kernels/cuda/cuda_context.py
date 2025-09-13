# -*- coding: utf-8 -*-

"""
This python module implements Cuda context handling

Copyright (C) 2018  SINTEF ICT

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <http://www.gnu.org/licenses/>.
"""

import os
import io
import hashlib
import gc

import pycuda.compiler as cuda_compiler
import pycuda.gpuarray
import pycuda.driver as cuda

from gpuocean.utils.timer import Timer
from ..context import Context, DeviceInfo


class CudaContext(Context):
    """
    Class which keeps track of the CUDA context and some helper functions
    """

    def __init__(self, device=0, context_flags=None, use_cache=True):
        """
        Create a new CUDA context

        Args:
            device: To use a specific GPU, provide either an ``id`` or ``pci_bus_id`` for the GPU.
            context_flags: To set a blocking context, provide ``cuda.ctx_flags.SCHED_BLOCKING_SYNC``.
            use_cache: Caches the kernels after they are compiled.
        """

        super().__init__(self.Architecture.CUDA, device, context_flags, use_cache)

        self.device = device

        # Initialize cuda (must be the first call to PyCUDA)
        cuda.init(flags=0)

        self.logger.info(f"PyCUDA version {str(pycuda.VERSION_TEXT)}")

        # Check that the device id is valid
        if self.device >= cuda.Device.count():
            self.device = self.device % cuda.Device.count()
            self.logger.debug(f"Changing device ID from {str(device)} to {str(self.device)}")

        self.cuda_device = cuda.Device(self.device)
        self.device_info = DeviceInfo(self.device, self.cuda_device.name(), str(cuda.get_version()),
                                      str(cuda.get_driver_version()))

        # Print some info about CUDA
        self.logger.info(f"CUDA version {self.device_info.api_version}")
        self.logger.info(f"Driver version {self.device_info.driver_version}")

        self.logger.info(f"Using {self.device_info.name} GPU")
        self.logger.debug(f" => compute capability: {str(self.cuda_device.compute_capability())}")
        free, total = cuda.mem_get_info()
        self.logger.debug(f" => memory: {int(free / (1024 * 1024))} / {int(total / (1024 * 1024))} MB available")

        # Create the CUDA context
        if context_flags is None:
            context_flags = cuda.ctx_flags.SCHED_AUTO

        self.cuda_context = self.cuda_device.make_context(flags=context_flags)

        self.logger.info(f"Created context handle <{str(self.cuda_context.handle)}>")

    def __del__(self, *args):
        self.logger.info(f"Cleaning up CUDA context handle <{str(self.cuda_context.handle)}>")

        # Loop over all contexts in stack, and remove "this"
        other_contexts = []
        while cuda.Context.get_current() is not None:
            context = cuda.Context.get_current()
            if context.handle != self.cuda_context.handle:
                self.logger.debug(f"<{str(self.cuda_context.handle)}> Popping <{str(context.handle)}> (*not* ours)")
                other_contexts = [context] + other_contexts
                cuda.Context.pop()
            else:
                self.logger.debug(f"<{str(self.cuda_context.handle)}> Popping <{str(context.handle)}> (ours)")
                cuda.Context.pop()

        # Add all the contexts we popped that were not our own
        for context in other_contexts:
            self.logger.debug(f"<{str(self.cuda_context.handle)}> Pushing <{str(context.handle)}>")
            cuda.Context.push(context)

        self.logger.debug(f"<{str(self.cuda_context.handle)}> Detaching")
        self.cuda_context.detach()

    def __str__(self):
        return "CudaContext id " + str(self.cuda_context.handle)

    def get_kernel(self, kernel_filename: str,
                   include_dirs: dict = None,
                   defines: dict[str, dict] = None,
                   compile_args: dict = None,
                   jit_compile_args: dict = None) -> cuda.Module:
        """
        Reads a text file and creates an OpenCL kernel from that.

        Args:
            kernel_filename: The file to use for the kernel.
            function: The main function of the kernel.
            include_dirs: List of directories for the ``#include``s referenced.
            defines: Adds ``#define`` tags to the kernel, such as ``#define key value``.
            compile_args: Adds other compiler options (parameters) for ``pycuda.compiler.compile()``.
            jit_compile_args: Adds other just-in-time compilation options (parameters)
                for ``pycuda.driver.module_from_buffer()``.
        
        Returns:
            The kernel module (pycuda.driver.Module).
        """

        if defines is None:
            defines = {}
        if include_dirs is None:
            include_dirs = [os.path.join(self.module_path), "include"]
        if compile_args is None:
            compile_args = {'cuda': {'no_extern_c': True}}
        if jit_compile_args is None:
            jit_compile_args = {}

        def cuda_compile_message_handler(compile_success_bool, info_str, error_str):
            """
            Helper function to print compilation output
            """

            self.logger.debug(f"Compilation returned {str(compile_success_bool)}")
            if info_str:
                self.logger.debug(f"Info: {info_str}")
            if error_str:
                self.logger.debug(f"Error: {error_str}")

        self.logger.debug(f"Getting {kernel_filename}")

        if not 'arch' in compile_args.keys():
            # HACK: Since CUDA 11.1 does not know about newer compute architectures that 8.6
            if ((cuda.Device(self.device).get_attribute(cuda.device_attribute.COMPUTE_CAPABILITY_MAJOR) > 8) or
                    (cuda.Device(self.device).get_attribute(cuda.device_attribute.COMPUTE_CAPABILITY_MAJOR) == 8 and
                     cuda.Device(self.device).get_attribute(cuda.device_attribute.COMPUTE_CAPABILITY_MINOR) > 6)):
                compile_args['arch'] = "sm_80"

        compile_args = compile_args.get('')

        kernel_filename = os.path.normpath(kernel_filename + ".cu")
        kernel_path = os.path.abspath(os.path.join(self.module_path, kernel_filename))
        # self.logger.debug("Getting %s", kernel_filename)

        # Create a hash of the kernel options
        options_hasher = hashlib.md5()
        options_hasher.update(str(defines).encode('utf-8') + str(compile_args).encode('utf-8'))
        options_hash = options_hasher.hexdigest()

        # Create hash of kernel souce
        source_hash = self.hash_kernel(
            kernel_path,
            include_dirs=[self.module_path] + include_dirs)

        # Create final hash
        root, ext = os.path.splitext(kernel_filename)
        kernel_hash = root \
                      + "_" + source_hash \
                      + "_" + options_hash \
                      + ext
        cached_kernel_filename = os.path.join(self.cache_path, kernel_hash)

        # If we have the kernel in our hashmap, return it
        if kernel_hash in self.modules.keys():
            self.logger.debug(f"Found kernel {kernel_filename} cached in hashmap ({kernel_hash})")
            return self.modules[kernel_hash]

        # If we have it on disk, return it
        elif self.use_cache and os.path.isfile(cached_kernel_filename):
            self.logger.debug(f"Found kernel {kernel_filename} cached on disk ({kernel_hash})")

            with io.open(cached_kernel_filename, "rb") as file:
                file_str = file.read()
                module = cuda.module_from_buffer(file_str, message_handler=cuda_compile_message_handler,
                                                 **jit_compile_args)

            self.modules[kernel_hash] = module
            return module

        # Otherwise, compile it from source
        else:
            self.logger.debug(f"Compiling {kernel_filename} ({kernel_hash})")

            # Create kernel string
            kernel_string = ""
            for key, value in defines.items():
                kernel_string += f"#define {str(key)} {str(value)}\n"
            kernel_string += f"#include \"{os.path.join(self.module_path, kernel_filename)}\""
            if self.use_cache:
                cached_kernel_dir = os.path.dirname(cached_kernel_filename)
                if not os.path.isdir(cached_kernel_dir):
                    os.mkdir(cached_kernel_dir)
                with io.open(cached_kernel_filename + ".txt", "w") as file:
                    file.write(kernel_string)

            with Timer("compiler") as timer:
                import warnings
                with warnings.catch_warnings():
                    warnings.filterwarnings("ignore",
                                            message="The CUDA compiler succeeded, but said the following:\nkernel.cu",
                                            category=UserWarning)
                    cubin = cuda_compiler.compile(kernel_string, include_dirs=include_dirs, cache_dir=False,
                                                  **compile_args)
                module = cuda.module_from_buffer(cubin, message_handler=cuda_compile_message_handler,
                                                 **jit_compile_args)
                if self.use_cache:
                    with io.open(cached_kernel_filename, "wb") as file:
                        file.write(cubin)

            self.modules[kernel_hash] = module
            return module

    def clear_kernel_cache(self) -> None:
        """
        Clears the kernel cache (useful for debugging & development)
        """

        self.logger.debug("Clearing cache")
        self.modules = {}
        gc.collect()

    def synchronize(self) -> None:
        """
        Synchronizes all streams, etc.
        """

        self.cuda_context.synchronize()
