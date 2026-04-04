"""Build TUNI CUDA kernels (sm_89 for L4, CUDA 12.8)."""
from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

NVCC_FLAGS = ["-O3", "--use_fast_math", "-gencode=arch=compute_89,code=sm_89"]
CXX_FLAGS = ["-O3"]

setup(
    name="tuni_cuda_kernels",
    version="0.2.0",
    ext_modules=[
        CUDAExtension(
            "tuni_cuda_kernels",
            ["tuni_ops.cu"],
            extra_compile_args={"cxx": CXX_FLAGS, "nvcc": NVCC_FLAGS},
        ),
        CUDAExtension(
            "tuni_local_rgbt_attn",
            ["local_rgbt_attn.cu"],
            extra_compile_args={"cxx": CXX_FLAGS, "nvcc": NVCC_FLAGS},
        ),
        CUDAExtension(
            "tuni_global_rgbt_attn",
            ["global_rgbt_attn.cu"],
            extra_compile_args={"cxx": CXX_FLAGS, "nvcc": NVCC_FLAGS},
        ),
    ],
    cmdclass={"build_ext": BuildExtension},
)
