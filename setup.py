# -*- encoding: utf8 -*-

from distutils.core import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext

ext_modules = [
    Extension("vjdetect", ["vjdetect.pyx"],
    extra_compile_args = ["-g"],
    ),
]

setup(
    name = "vjdetect",
    author = "Alexis Mignon",
    author_email = "alexis.mignon@gmail.com",
    version = "0.1",
    cmdclass = {'build_ext': build_ext},
    ext_modules = ext_modules
)
