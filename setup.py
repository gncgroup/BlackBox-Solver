from distutils.core import setup, Extension
from Cython.Build import cythonize
import numpy

setup(
    ext_modules=[
        Extension("neuronet", ["neuronet.c"],
				  extra_compile_args=["-O3"],	
                  include_dirs=[numpy.get_include()]),
    ],
)
 

setup(
    ext_modules=cythonize("neuronet.pyx"),
    include_dirs=[numpy.get_include()]
)    