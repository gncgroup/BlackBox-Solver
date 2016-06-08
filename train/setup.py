from distutils.core import setup, Extension
from Cython.Build import cythonize
import numpy

setup(
    ext_modules=[
        Extension("bot", ["bot.c"],
				extra_compile_args=["-O3"],	
                include_dirs=[numpy.get_include()]),
    ],
)

# Or, if you use cythonize() to make the ext_modules list,
# include_dirs can be passed to setup()

setup(
    ext_modules=cythonize("bot.pyx"),
    include_dirs=[numpy.get_include()]
)    