# Thunder configuration with custom petsc and hdf5 install in Home
# module setting:
#   module switch openmpi openmpi/1.6.5-static-gcc48
#   module switch gcc gcc/4.8.2
#   

set(CMAKE_C_COMPILER       "mpicc")
set(CMAKE_Fortran_COMPILER "mpif90")

set(USER_C_FLAGS               " -cpp --std=c99 ")
set(USER_Fortran_FLAGS         " -cpp -fbacktrace -finit-real=nan -ffree-line-length-none ")
set(USER_Fortran_FLAGS_RELEASE " -funroll-all-loops -O3 -march=native -mtune=native ")
set(USER_Fortran_FLAGS_DEBUG   " -W -Wall -Wuninitialized -fcheck=all -fbacktrace -O0 -g -ffpe-trap=invalid,zero,overflow ")

set(NETCDF_INCLUDE_DIR "/scratch/mpi/mpiaes/m300362/libs/netcdf-fortran-gcc48/include/")
set(NETCDF_LIB_1       "/scratch/mpi/mpiaes/m300362/libs/netcdf-fortran-gcc48/lib/libnetcdff.a")
set(NETCDF_LIB_2       "/scratch/mpi/mpiaes/m300362/libs/netcdf-gcc48/lib/libnetcdf.a")

set(HDF5_INCLUDE_DIRS       "/scratch/mpi/mpiaes/m300362/libs/hdf5/include")
list(APPEND HDF5_LIBRARIES  "/scratch/mpi/mpiaes/m300362/libs/hdf5/lib/libhdf5hl_fortran.a")
list(APPEND HDF5_LIBRARIES  "/scratch/mpi/mpiaes/m300362/libs/hdf5/lib/libhdf5_fortran.a")

set(SZIP_LIB           "/sw/squeeze-x64/szip-latest-static/lib/libsz.a")

set(LIBS ${NETCDF_LIB_1} ${NETCDF_LIB_2} ${SZIP_LIB} ${HDF5_LIBRARIES} m z curl jpeg)
