import ctypes
import importlib.util
import os
import sys


# libcudnn_graph comes first: the other cuDNN libraries depend on it, and
# loading it explicitly stops a system cuDNN on LD_LIBRARY_PATH from being
# picked up instead.
_CUDNN_LIBS = (
    "libcudnn_graph.so.9",
    "libcudnn.so.9",
    "libcudnn_engines_runtime_compiled.so.9",
    "libcudnn_engines_precompiled.so.9",
    "libcudnn_heuristic.so.9",
    "libcudnn_ops.so.9",
    "libcudnn_adv.so.9",
    "libcudnn_cnn.so.9",
)
_CUDNN_PRELOADED = False


def preload_cudnn_libraries():
    global _CUDNN_PRELOADED
    if _CUDNN_PRELOADED:
        return

    spec = importlib.util.find_spec("nvidia.cudnn")
    if spec is None or spec.submodule_search_locations is None:
        return

    libdir = os.path.join(next(iter(spec.submodule_search_locations)), "lib")
    for name in _CUDNN_LIBS:
        path = os.path.join(libdir, name)
        if os.path.exists(path):
            ctypes.CDLL(path, mode=ctypes.RTLD_GLOBAL)

    _CUDNN_PRELOADED = True


def register_flash_attn_kernel():
    # vortex calls the raw flash-attn ops through `import flash_attn_2_cuda`.
    # Serve that module from the prebuilt Hub kernel, which exposes the same
    # ops, so flash-attn doesn't have to be compiled from source.
    if "flash_attn_2_cuda" in sys.modules or importlib.util.find_spec("flash_attn_2_cuda"):
        return

    from kernels import get_kernel

    sys.modules["flash_attn_2_cuda"] = get_kernel("kernels-community/flash-attn2")


def setup_evo2_runtime():
    """Call before importing evo2 or vortex."""
    preload_cudnn_libraries()
    register_flash_attn_kernel()
