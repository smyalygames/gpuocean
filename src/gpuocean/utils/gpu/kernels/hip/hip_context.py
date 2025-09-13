import hashlib
import io
import os.path

import hip as hip_main
from hip import hip, hiprtc

from ..context import Context, DeviceInfo
from ...hip_utils import hip_check
from gpuocean.utils.timer import Timer


class HIPContext(Context):
    """
    Class that manages the HIP context.
    """

    def __init__(self, device=None, context_flags=None, use_cache=True):
        """
        Creates a new HIP context.

        Args:
            device: Device ID of the GPU to use.
            context_flags: Does nothing.
            use_cache: Uses previously compiled kernel cache.
        """
        super().__init__(self.Architecture.HIP, device, context_flags, use_cache)

        hip_version = hip_main.HIP_VERSION_NAME
        rocm_version = hip_main.ROCM_VERSION_NAME

        # Log information about HIP version
        self.logger.info(f"HIP Python version {hip_version}")
        self.logger.info(f"ROCm version {rocm_version}")

        if device is None:
            device = 0

        hip_check(hip.hipSetDevice(device))

        # Device information
        props = hip.hipDeviceProp_t()
        hip_check(hip.hipGetDeviceProperties(props, device))
        device_count = hip_check(hip.hipGetDeviceCount())
        self.device_info = DeviceInfo(device, props.name.decode(), hip_version, rocm_version)
        self.arch = props.gcnArchName

        self.logger.info(
            f"Using device {device}/{device_count} '{self.device_info.name} ({self.arch.decode()})'"
            + f" ({props.pciBusID})"
        )
        self.logger.debug(f" => total available memory: {int(props.totalGlobalMem / pow(1024, 2))} MiB")

    def __del__(self):
        for module in self.modules.values():
            hip_check(hip.hipModuleUnload(module))

    def __str__(self):
        return f"HIPContext id {self.get_device_id()}"

    @staticmethod
    def get_device_id() -> int:
        return hip_check(hip.hipGetDevice())

    def get_kernel(self, kernel_filename: str,
                   include_dirs: list[str] = None,
                   defines: dict[str, int] = None,
                   compile_args: dict[str, list] = None,
                   jit_compile_args: dict = None) -> hip.ihipModule_t:
        """
        Reads a ``.hip`` file and creates a HIP kernel from that.

        Args:
            kernel_filename: The file to use for the kernel.
            include_dirs: List of directories for the ``#include``s referenced.
            defines: Adds ``#define`` tags to the kernel, such as: ``#define key value``.
            compile_args: Adds other compiler options (parameters) for ``pycuda.compiler.compile()``.
            jit_compile_args: Adds other just-in-time compilation options (parameters)
                for ``pycuda.driver.module_from_buffer()``.

        Returns:
            The kernel module (pycuda.driver.Module).
        """
        if defines is None:
            defines = {}
        if include_dirs is None:
            include_dirs = [os.path.join(self.module_path, "include")]
        if compile_args is None:
            compile_args = {'hip': []}
        if jit_compile_args is None:
            jit_compile_args = {}

        compile_args = compile_args.get('')

        compile_args = [bytes(arg, "utf-8") for arg in compile_args]
        compile_args.append(b"--offload-arch=" + self.arch)

        kernel_filename = os.path.normpath(kernel_filename + ".hip")
        kernel_path = os.path.abspath(os.path.join(self.module_path, kernel_filename))

        # Create a hash of the kernel options
        options_hasher = hashlib.md5()
        options_hasher.update(str(defines).encode('utf-8') + str(compile_args).encode('utf-8'))
        options_hash = options_hasher.hexdigest()

        # Create hash of the kernel source
        source_hash = self.hash_kernel(kernel_path, include_dirs=[self.module_path] + include_dirs)

        # Create the final hash
        root, ext = os.path.splitext(kernel_filename)
        kernel_hash = root + "_" + source_hash + "_" + options_hash + ext
        cached_kernel_filename = os.path.join(self.cache_path, kernel_hash)

        # Checks if the module is already cached in the hash map
        if kernel_hash in self.modules.keys():
            self.logger.debug(f"Found kernel {kernel_filename} cached in hashmap ({kernel_hash}).")
            return self.modules[kernel_hash]
        elif self.use_cache and os.path.isfile(cached_kernel_filename):
            # Check if the cache is on the disk
            self.logger.debug(f"Found kernel {kernel_filename} cached on disk ({kernel_hash}).")

            with io.open(cached_kernel_filename, "rb") as file:
                code = file.read()
                module: hip.ihipModule_t = hip_check(hip.hipModuleLoadData(code))

            self.modules[kernel_hash] = module
            return module
        else:
            # As it was not found in the cache, compile it.
            self.logger.debug(f"Compiling {kernel_filename} ({kernel_hash}) for {self.arch}.")

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
                prog: hiprtc.hiprtcProgram = hip_check(
                    hiprtc.hiprtcCreateProgram(bytes(kernel_string, "utf-8"), bytes(kernel_filename, "utf-8"),
                                               0, [], []))

                err, = hiprtc.hiprtcCompileProgram(prog, len(compile_args), compile_args)
                if err != hiprtc.hiprtcResult.HIPRTC_SUCCESS:
                    log_size = hip_check(hiprtc.hiprtcGetProgramLogSize(prog))
                    log = bytearray(log_size)
                    hip_check(hiprtc.hiprtcGetProgramLog(prog, log))
                    raise RuntimeError(log.decode())

                code_size = hip_check(hiprtc.hiprtcGetCodeSize(prog))
                code = bytearray(code_size)
                hip_check(hiprtc.hiprtcGetCode(prog, code))

                hip_check(hiprtc.hiprtcDestroyProgram(prog))

                module: hip.ihipModule_t = hip_check(hip.hipModuleLoadData(code))

                if self.use_cache:
                    with io.open(cached_kernel_filename, "wb") as file:
                        file.write(code)

            self.modules[kernel_hash] = module
            return module

    def synchronize(self) -> None:
        hip_check(hip.hipDeviceSynchronize())
