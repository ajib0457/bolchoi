import numpy as np
import scipy.optimize as op
import emcee
from emcee.utils import MPIPool
import sys
import h5py

mass_bin=0   # 0 to 5

tot_mass_bins=6
f=h5py.File("/scratch/GAMNSCM2/bolchoi_z0/cat_reconfig/files/output_files/bolchoi_DTFE_rockstar_allhalos_xyz_vxyz_jxyz_m_r.h5", 'r')#xyz vxvyvz jxjyjz & Rmass & Rvir: Halo radius (kpc/h comoving).
data=f['/halo'][:]
f.close()

partcl_500=np.where((data[:,9]/(1.35*10**8))>=500)#filter out halos with <500 particles
data=data[partcl_500]
halo_mass=data[:,9]
log_halo_mass=np.log10(halo_mass)#convert into log(M)
mass_intvl=(np.max(log_halo_mass)-np.min(log_halo_mass))/tot_mass_bins
low_int_mass=np.min(log_halo_mass)+mass_intvl*mass_bin
hi_int_mass=low_int_mass+mass_intvl
del data
del halo_mass
del log_halo_mass
#----
grid_nodes=850
sim_sz=250#Mpc
in_val,fnl_val=-140,140
s=3.92

#Calculate the std deviation in physical units
grid_phys=1.*sim_sz/grid_nodes#Size of each voxel in physical units
val_phys=1.*(2*fnl_val)/grid_nodes#Value in each grid voxel
std_dev_phys=1.*s/val_phys*grid_phys

f=h5py.File('/project/GAMNSCM2/bolchoi_z0/correl/DTFE/files/output_files/dotproduct/spin_lss/DTFE_grid%d_spin_store_fil_Log%s-%s_smth%sMpc_%sbins.h5'%(grid_nodes,round(low_int_mass,2),round(hi_int_mass,2),round(std_dev_phys,3),tot_mass_bins),'r')     
costheta=f['/dp'][:]
f.close()
costheta=abs(costheta)#To change the dot products to be between 0-1 since we only care about alignment, not specific direction

#MCMC
def lnlike(c,costheta):
    loglike=np.zeros((1))
    loglike[0]=sum(np.log((1-c)*np.sqrt(1+(c/2))*(1-c*(1-3*(costheta*costheta/2)))**(-1.5)))#log-likelihood 

    return loglike
    
def lnprior(c):
    
    if (-1.5 < c < 0.99):#Assumes a flat prior, uninformative prior
        return 0.0
    return -np.inf
    
def lnprob(c,costheta):
    lp = lnprior(c)
    if not np.isfinite(lp):
        return -np.inf
    return lp + lnlike(c,costheta)
#Parallel MCMC - initiallizes pool object; if process isn't running as master, wait for instr. and exit
pool=MPIPool()
if not pool.is_master():
    pool.wait()
    sys.exit(0)

#Initial conditions
ndim, nwalkers = 1, 600
initial_c=0.4

pos = [initial_c+1e-2*np.random.randn(ndim) for i in range(nwalkers)]#initial positions for walkers "Gaussian ball"
 
#MCMC Running
sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob,args=[costheta],pool=pool)
#Burn-in
#print("Running burn-in...") #get rid of burn-in bit if you want to see walekrs chain from start to finish
burn_in=500
pos, _, _=sampler.run_mcmc(pos,burn_in)#running of emcee burn-in period
sampler.reset()
#MCMC Running
#print("Running production...")
steps_wlk=5000
sampler.run_mcmc(pos, steps_wlk)#running of emcee for steps specified, using pos as initial walker positions

pool.close()

c_samples=sampler.flatchain[:,0]

f=h5py.File("/scratch/GAMNSCM2/bolchoi_z0/correl/DTFE/files/output_files/mcmc/spin_lss/flt_chain_%sburnin_%ssteps_%snwalkrs_grid%d_fil_Log%s-%s_smth%sMpc_spin_lss.pkl"%(burn_in,steps_wlk,nwalkers,grid_nodes,round(low_int_mass,2),round(hi_int_mass,2),round(std_dev_phys,3)),'w')     
f.create_dataset('/mcmc',data=c_samples)
f.close()
