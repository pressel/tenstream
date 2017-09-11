cimport mpi4py.libmpi as mpi
from mpi4py import MPI
import numpy as np
import netCDF4 as nc

#cdef extern from "/home/pressel/local/tenstream/C_wrapper/c_tenstream.h": 
cdef extern: 
     void f2c_test(int a, int* Nx, int* Ny, int* Nz, 
                   double *dx, double *dy, 
                   double *phi0, double *theta0,
                   double *albedo_thermal, double *albedo_solar, 
                   int *atm_filename_length, char *atm_filename, 
                   int *lthermal, int *lsolar, 
                   int *NZ_merged,
                   double **edir, double **edn, double **eup, double **abso, 
                   double *d_plev, double *d_tlev, double *d_h2ovmr, double *d_lwc, double *d_reliq, 
                   double *d_iwc, double *d_reice,
                   int *nprocx, int *nxproc, int *nprocy, int *nyproc)
     void f2c_init_petsc() 
     void f2c_finalize_petsc() 

cdef int a = 441



from mpi4py import MPI
cdef mpi.MPI_Comm comm_world = mpi.MPI_COMM_WORLD 
cdef int Nx = 4, Ny = 4, Nz = 75
cdef int Nx_full = Nx, Ny_full = Ny, Nz_full = Nz 
cdef double dx = 100, dy = 100, dz = 40
cdef double phi0 = 0.0, theta0 = 60.0 
cdef double albedo_thermal = 1e-8, albedo_solar = 0.2 
cdef char* atm_filename = "afglus_100m.dat" 
cdef int atm_filename_length = len(atm_filename) 
cdef int lthermal = 1, lsolar = 1
cdef int Nz_merged 
cdef double *edir
cdef double *edn 
cdef double *eup
cdef double *abso 



cdef double [:] d_plev  = np.zeros(Nx * Ny * (Nz + 1) , dtype=np.double, order = 'F') 
cdef double [:] d_tlev  = np.zeros(Nx * Ny * (Nz + 1), dtype=np.double, order = 'F') 
cdef double [:] d_h2ovmr = np.zeros(Nx * Ny * Nz, dtype=np.double, order = 'F')
cdef double [:] d_lwc = np.zeros(Nx * Ny * Nz, dtype=np.double, order = 'F')
cdef double [:] d_reliq = np.zeros(Nx * Ny * Nz, dtype=np.double, order = 'F')
cdef double [:] d_iwc = np.zeros(Nx * Ny * Nz, dtype=np.double, order = 'F')
cdef double [:] d_reice = np.zeros(Nx * Ny * Nz, dtype=np.double, order = 'F')


cdef double [:,:,:] abso_reshape 
cdef double [:,:,:] edir_reshape 
cdef double [:,:,:] eup_reshape 
cdef double [:,:,:] edn_reshape 






cdef int numprocs, rank
mpi.MPI_Comm_size(comm_world, &numprocs);
mpi.MPI_Comm_rank(comm_world, &rank) 

#Check to make sure processors are an even square 


cdef int nprocy
cdef int nprocx 
nprocx = np.int(np.sqrt(numprocs)) 
nprocy= nprocx 
if not np.isclose(np.double(nprocx) * np.double(nprocy), np.double(numprocs),atol=1e-3):
    mpi.Finalize() 
    import sys; sys.exit() 


Ny /= nprocy
Nx /= nprocx
cdef int [:] nyproc = np.zeros(nprocy, dtype=np.intc) + Ny 
cdef int [:] nxproc = np.zeros(nprocx, dtype=np.intc) + Nx



cdef int i_rank = rank % nprocx
cdef int j_rank = rank // nprocy

#print rank, numprocs, i_rank, j_rank
#import sys; sys.exit() 

d = nc.Dataset('./bomex_data/6300.nc', 'r') 
cdef double [:,:,:] t_lev = d['fields']['temperature'][:,:,:] 
cdef double [:,:,:] ql_lev = d['fields']['ql'][:,:,:] 
cdef double [:,:,:] qt_lev = d['fields']['qt'][:,:,:] 
d.close() 

print np.shape(t_lev) , np.max(ql_lev) 

cdef int k, i, j, ind  


print Nx, Ny 
#Initialize the pressure profile 
with nogil: 
    for k in xrange(Nz + 1): 
        for i in xrange(Nx): 
            for j in xrange(Ny): 

                
                ind = j * Nx * (Nz + 1) + i * (Nz + 1) + k 
                d_plev[ind] = 1013.0 - k * 200.0/(Nz+1)
                if k == 0: 
                    d_tlev[ind] = t_lev[i_rank * Nx + i, j_rank * Ny +  j, k]#288.0 - k * 10.0/(Nz+1);
                else: 
                    d_tlev[ind] = t_lev[i_rank * Nx + i, j_rank * Ny +  j, k-1]
                
    for k in xrange(Nz): 
        for i in xrange(Nx): 
            for j in xrange(Ny):  
                ind = j * Nx * (Nz) + i * (Nz) + k         
                d_h2ovmr  [ind] = qt_lev[i_rank * Nx + i, j_rank * Ny +  j, k] * 1000.0;
                d_lwc  [ind] = ql_lev[i_rank * Nx + i, j_rank * Ny +  j, k] * 1000.0;
                d_reliq[ind] = 10; 
                d_iwc  [ind] =  0;
                d_reice[ind] = 10;
f2c_init_petsc() 

import time 
for i in range(2): 
    t1 = time.time() 
    f2c_test(mpi.MPI_Comm_c2f(comm_world), 
        &Nx, &Ny, &Nz, 
        &dx, &dy,
        &phi0, &theta0,
        &albedo_thermal, &albedo_solar,
        &atm_filename_length, atm_filename,
        &lthermal, &lsolar, &Nz_merged, 
        &edir, &edn, &eup, &abso, 
        &d_plev[0], &d_tlev[0], &d_h2ovmr[0], &d_lwc[0], &d_reliq[0], &d_iwc[0], &d_reice[0], 
        &nprocx, &nxproc[0], &nprocy, &nyproc[0])  
    t2 = time.time() 
    if rank == 0: 
       print "run ", i , " took ", t2 - t1, " seconds" 
print 'Reshaping arrays'


#These are for a local array 
abso_reshape = np.empty((Nz_merged, Nx, Ny),dtype=np.double, order='F') 
edir_reshape = np.empty((Nz_merged + 1, Nx, Ny),dtype=np.double, order='F') 
edn_reshape = np.empty((Nz_merged + 1, Nx, Ny),dtype=np.double, order='F') 
eup_reshape = np.empty((Nz_merged + 1, Nx, Ny),dtype=np.double, order='F') 

#These are for a local array
cdef double [:,:,:] abso_reshape_full = np.zeros((Nz_merged, Nx_full, Ny_full),dtype=np.double, order='F') 
cdef double [:,:,:] edir_reshape_full = np.zeros((Nz_merged + 1, Nx_full, Ny_full),dtype=np.double, order='F') 
cdef double [:,:,:] edn_reshape_full = np.zeros((Nz_merged + 1, Nx_full, Ny_full),dtype=np.double, order='F') 
cdef double [:,:,:] eup_reshape_full = np.zeros((Nz_merged + 1, Nx_full, Ny_full),dtype=np.double, order='F') 

cdef double [:,:,:] abso_reshape_full_tmp = np.zeros((Nz_merged, Nx_full, Ny_full),dtype=np.double, order='F') 
cdef double [:,:,:] edir_reshape_full_tmp = np.zeros((Nz_merged + 1, Nx_full, Ny_full),dtype=np.double, order='F') 
cdef double [:,:,:] edn_reshape_full_tmp = np.zeros((Nz_merged + 1, Nx_full, Ny_full),dtype=np.double, order='F') 
cdef double [:,:,:] eup_reshape_full_tmp = np.zeros((Nz_merged + 1, Nx_full, Ny_full),dtype=np.double, order='F') 



with nogil: 

    for k in xrange(Nz_merged): 
        for i in xrange(Nx): 
            for j in xrange(Ny): 
                ind = j * Nx * (Nz_merged) + i * (Nz_merged) + k 
                abso_reshape_full_tmp[k, i_rank * Nx + i,j_rank * Ny + j] = abso[ind]
                
    for k in xrange(Nz_merged+1): 
        for i in xrange(Nx): 
            for j in xrange(Ny): 
                ind = j * Nx * (Nz_merged + 1 ) + i * (Nz_merged + 1 ) + k 
                edir_reshape_full_tmp[k, i_rank * Nx + i,j_rank * Ny + j] = edir[ind]
                edn_reshape_full_tmp[k, i_rank * Nx + i,j_rank * Ny + j] = edn[ind]
                eup_reshape_full_tmp[k, i_rank * Nx + i,j_rank * Ny + j] = eup[ind]
                
                
    for k in xrange(Nz_merged): 
        for i in xrange(Nx): 
            for j in xrange(Ny): 
                ind = j * Nx * (Nz_merged) + i * (Nz_merged) + k 
                abso_reshape[k,i,j] = abso[ind]
      
    for k in xrange(Nz_merged+1): 
        for i in xrange(Nx): 
            for j in xrange(Ny): 
                ind = j * Nx * (Nz_merged + 1 ) + i * (Nz_merged + 1 ) + k 
                edir_reshape[k,i,j] = edir[ind]          
                edn_reshape[k,i,j] = edn[ind]
                eup_reshape[k,i,j] = eup[ind]

comm = MPI.COMM_WORLD
comm.Reduce(abso_reshape_full_tmp, abso_reshape_full, op=MPI.SUM)
comm.Reduce(edir_reshape_full_tmp, edir_reshape_full, op=MPI.SUM)
comm.Reduce(edn_reshape_full_tmp, edn_reshape_full, op=MPI.SUM)
comm.Reduce(eup_reshape_full_tmp, eup_reshape_full, op=MPI.SUM)

if rank == 0:
    import cPickle as pkl
    f = 'full.pkl' 
    f = open(f, 'wb') 
    d = {} 
    d['abso_reshape_full'] = np.array(abso_reshape_full) 
    d['edir_reshape_full'] = np.array(edir_reshape_full)
    d['edn_reshape_full'] = np.array(edn_reshape_full)
    d['eup_reshape_full'] = np.array(eup_reshape_full)


    pkl.dump(d, f) 
    f.close() 
    
if rank == 0: 
   print np.amax(np.array(abso_reshape_full))


import matplotlib as mpl
mpl.use('Agg')        
import pylab as plt
plt.figure(1) 
plt.contourf(np.array(abso_reshape)[-1,:,:],100)
plt.colorbar() 
plt.savefig('abso_fig.png') 

plt.figure(2) 
plt.contourf(np.array(edir_reshape)[-1,:,:],100)
plt.colorbar() 
plt.savefig('edir_fig.png') 

plt.figure(3) 
plt.contourf(np.array(edn_reshape)[-1,:,:],100)
plt.colorbar() 
plt.savefig('en_fig.png') 

plt.figure(4) 
plt.contourf(np.array(eup_reshape)[-1,:,:],100)
plt.colorbar() 
plt.savefig('up_fig.png')

plt.figure(5) 
plt.contourf(np.array(abso_reshape)[::-1,1,:][:75,:],100)
plt.colorbar() 
plt.savefig('abso_fig_z.png') 

plt.figure(6) 
plt.contourf(np.array(edir_reshape)[::-1,1,:][:75,:],100)
plt.colorbar() 
plt.savefig('edir_fig_z.png') 

plt.figure(7) 
plt.contourf(np.array(edn_reshape)[::-1,1,:][:75,:],100)
plt.colorbar() 
plt.savefig('en_fig_z.png') 

plt.figure(8) 
plt.contourf(np.array(eup_reshape)[::-1,1,:][:75,:],100)
plt.colorbar() 
plt.savefig('up_fig_z.png')

plt.figure(9) 
plt.contourf(np.mean(np.array(abso_reshape)[::-1,:,:],axis=1)[:75,:],100)
plt.colorbar() 
plt.savefig('abso_fig_z_mean.png') 

plt.figure(10) 
plt.contourf(np.mean(np.array(edir_reshape)[::-1,:,:],axis=1)[:75,:],100)
plt.colorbar() 
plt.savefig('edir_fig_z_mean.png') 

plt.figure(11) 
plt.contourf(np.mean(np.array(edn_reshape)[::-1,:,:],axis=1)[:75,:],100)
plt.colorbar() 
plt.savefig('en_fig_z_mean.png') 

plt.figure(12) 
plt.contourf(np.mean(np.array(eup_reshape)[::-1,:,:],axis=1)[:75,:],100)
plt.colorbar() 
plt.savefig('up_fig_z_mean.png')

f2c_finalize_petsc()
