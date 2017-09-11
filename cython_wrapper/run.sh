#!/bin/bash
#PBS -N PyCLES
#PBS -q default
#PBS -l nodes=256
#PBS -l walltime=24:00:00
#PBS -V
echo “MPI Used:” `which mpirun`

echo Working directory is $PBS_O_WORKDIR
cd $PBS_O_WORKDIR

echo Running on host `hostname`
echo Time is `date`

NPROCS=`wc -l < $PBS_NODEFILE`
echo This job has allocated $NPROCS cpus

echo ” ”
echo This jobs runs on the following processors:
echo `cat $PBS_NODEFILE`
echo ” 


export LD_LIBRARY_PATH=/share/apps/software/rhel6/software/matplotlib/1.5.1-foss-2016a-Python-2.7.11/lib:/share/apps/software/rhel6/software/freetype/2.6.2-foss-2016a/lib:/share/apps/software/rhel6/software/libpng/1.6.21-foss-2016a/lib:/share/apps/software/rhel6/software/netCDF-Fortran/4.4.3-foss-2016a/lib:/share/apps/software/rhel6/software/netCDF/4.4.0-foss-2016a/lib64:/share/apps/software/rhel6/software/cURL/7.47.0-foss-2016a/lib:/share/apps/software/rhel6/software/HDF5/1.8.16-foss-2016a/lib:/share/apps/software/rhel6/software/Szip/2.1-foss-2016a/lib:/share/apps/software/rhel6/software/Python/2.7.11-foss-2016a/lib:/share/apps/software/rhel6/software/GMP/6.1.0-foss-2016a/lib:/share/apps/software/rhel6/software/Tk/8.6.4-foss-2016a-no-X11/lib:/share/apps/software/rhel6/software/SQLite/3.9.2-foss-2016a/lib:/share/apps/software/rhel6/software/Tcl/8.6.4-foss-2016a/lib:/share/apps/software/rhel6/software/libreadline/6.3-foss-2016a/lib:/share/apps/software/rhel6/software/ncurses/6.0-foss-2016a/lib:/share/apps/software/rhel6/software/zlib/1.2.8-foss-2016a/lib:/share/apps/software/rhel6/software/bzip2/1.0.6-foss-2016a/lib:/share/apps/software/rhel6/software/ScaLAPACK/2.0.2-gompi-2016a-OpenBLAS-0.2.15-LAPACK-3.6.0/lib:/share/apps/software/rhel6/software/FFTW/3.3.4-gompi-2016a/lib:/share/apps/software/rhel6/software/OpenBLAS/0.2.15-GCC-4.9.3-2.25-LAPACK-3.6.0/lib:/share/apps/software/rhel6/software/OpenMPI/1.10.2-GCC-4.9.3-2.25/lib:/share/apps/software/rhel6/software/hwloc/1.11.2-GCC-4.9.3-2.25/lib:/share/apps/software/rhel6/software/numactl/2.0.11-GCC-4.9.3-2.25/lib:/share/apps/software/rhel6/software/binutils/2.25-GCCcore-4.9.3/lib:/share/apps/software/rhel6/software/GCCcore/4.9.3/lib/gcc/x86_64-unknown-linux-gnu/4.9.3:/share/apps/software/rhel6/software/GCCcore/4.9.3/lib64:/share/apps/software/rhel6/software/GCCcore/4.9.3/lib:/home/pressel/local/petsc/fast_double/lib/

mpirun -machinefile $PBS_NODEFILE -np $NPROCS python test_test.py 
