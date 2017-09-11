from distutils.core import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext
import mpi4py as mpi4py

#libplexrt.a  libtenstream.a  libtenstr_rrtmg.a  libtenstr_rrtm_lw.a  libtenstr_rrtm_sw.a
tenstr_lib_path = "../build/lib/" 
static_lib = [] 
static_lib.append(tenstr_lib_path + "libc_tenstream.a") 
static_lib.append(tenstr_lib_path + "libtenstr_rrtmg.a")
#static_lib.append(tenstr_lib_path + "libplexrt.a")
static_lib.append(tenstr_lib_path + "libtenstr_rrtm_lw.a")
static_lib.append(tenstr_lib_path + "libtenstr_rrtm_sw.a")
static_lib.append(tenstr_lib_path + "libtenstream.a")
static_lib.append("/home/pressel/local/petsc/fast_double/lib/libpetsc.so") 
static_lib.append("/home/pressel/local/petsc/fast_double/lib/libflapack.a") 
static_lib.append("/home/pressel/local/petsc/fast_double/lib/libfblas.a") 
static_lib.append("/home/pressel/local/petsc/fast_double/lib/libz.so") 
include_path = [mpi4py.get_include()]
include_path.append("/home/pressel/local/petsc/fast_double/lib/")

ext_modules = [Extension("test",
                     ["tenstream.pyx"],
                     library_dirs = ["/home/pressel/local/petsc/fast_double/lib/"], 
                     include_dirs = include_path, 
                     extra_objects=static_lib, 
                     libraries=["mpi", "gfortran", "netcdf", "netcdff", "X11", "pthread", "mpi_usempif08",
                     "mpi_usempi_ignore_tkr", "mpi_mpifh", "quadmath", "mpi_cxx", "stdc++", "m", "gcc_s", "dl"] #Can add petsc here  
                     )]

setup(
  name = 'test',
  cmdclass = {'build_ext': build_ext},
  ext_modules = ext_modules
)
