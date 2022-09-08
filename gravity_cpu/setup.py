from distutils.core import Extension, setup

import numpy as np

module = Extension(
    "gravity_cpu",
    sources=["gravity_cpu"],
    include_dirs=[np.get_include()],
    extra_compile_args=["-fopenmp"],
    extra_link_args=["-fopenmp"],
)

setup(
    name="gravity_cpu",
    version="1.0",
    description="",
    ext_modules=[module],
)
