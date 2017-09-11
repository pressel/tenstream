# Default GCC
#
# example Config script for a gcc linux environment 08.Jan.2015
#
# You need to provide paths for the NETCDF installation
# and additionally, you need to provide an environment variable so that we can find PETSC
# Make sure you have set PETSC_ARCH and PETSC_DIR
# If unclear, see the PETSC installation instructions and read the README for further hints.
#

message(STATUS "### USING GCC CONFIG ###")

set(CMAKE_Fortran_COMPILER   "/share/apps/software/rhel6/software/OpenMPI/1.10.2-GCC-4.9.3-2.25/bin/mpif90")
set(Fortran_COMPILER_WRAPPER "/share/apps/software/rhel6/software/OpenMPI/1.10.2-GCC-4.9.3-2.25/bin/mpif90")

set(USER_C_FLAGS               "-cpp -W -Wall -Wuninitialized --std=c99 -fPIC") 
set(USER_Fortran_FLAGS         "-cpp -ffree-line-length-none -W -Wall -Wuninitialized -g -fPIC") 
set(USER_Fortran_FLAGS_RELEASE "-fno-backtrace -fno-range-check -O3") 
set(USER_Fortran_FLAGS_DEBUG   "-fbacktrace -finit-real=nan -W -Wall -Wuninitialized -g -pg -fcheck=all -fbounds-check -pedantic -Wsurprising")

# Help Cmake find the netcdf installation
set(NETCDF_ROOT      "/share/apps/software/rhel6/software/netCDF/4.4.0-foss-2016a/")
set(HDF5_DIR  "/share/apps/software/rhel6/software/HDF5/1.8.16-foss-2016a/")

#set(PETSC_DIR "/home/thl/petsc")
#set(PETSC_ARCH "fast_double")
