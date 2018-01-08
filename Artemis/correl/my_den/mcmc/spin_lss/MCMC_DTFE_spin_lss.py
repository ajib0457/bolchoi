import numpy as np
import scipy.optimize as op
import pickle
import math as mth
import emcee
import matplotlib.pyplot as plt
np.set_printoptions(threshold=np.inf)#does not truncate arrays in console
import sklearn.preprocessing as skl
from emcee.utils import MPIPool
import sys
import pandas as pd
n_dot_prod=0
mass_bin=5   # 0 to 5
tot_mass_bins=6
data = pd.read_csv('/scratch/GAMNSCM2/bolchoi_z0/cat_reconfig/files/output_files/bolchoi_DTFE_rockstar_halos_z0_xyz_m_j',sep=r"\s+",lineterminator='\n', header = None)
data=data.as_matrix()
partcl_500=np.where((data[:,3]/(1.35*10**8))>=500)#filter out halos with <500 particles
data=data[partcl_500]
halo_mass=data[:,3]
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

inputfile=open("/project/GAMNSCM2/bolchoi_z0/correl/my_den/files/output_files/dotproduct/spin_lss/DTFE_grid%d_spin_store_%dbins_fil_Log%s-%s_smth%sMpc_%sbins.pkl"%(grid_nodes,n_dot_prod,round(low_int_mass,2),round(hi_int_mass,2),round(std_dev_phys,3),tot_mass_bins),'rb')
costheta=pickle.load(inputfile)
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
ndim, nwalkers = 1, 800
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
#Plotting
c_samples=sampler.flatchain[:,0]
output=open("/scratch/GAMNSCM2/bolchoi_z0/correl/my_den/files/output_files/mcmc/spin_lss/flt_chain_%sburnin_%ssteps_%snwalkrs_grid%d_fil_Log%s-%s_smth%sMpc_spin_lss.pkl"%(burn_in,steps_wlk,nwalkers,grid_nodes,round(low_int_mass,2),round(hi_int_mass,2),round(std_dev_phys,3)), 'wb')
pickle.dump(c_samples,output)
output.close() 
#inputfile=open("c_samples.pkl",'rb')
#c_samples=pickle.load(inputfile)
#X,Y,_=plt.hist(c_samples,bins=1000,normed=True)
#z=np.linspace(np.max(c_samples),np.min(c_samples),1000)
#plt.plot(z,x)#this is for the pdf
#print("--- %s seconds ---" %(time.time()-start_time))
'''
plt.xlabel("c values")
plt.title("posterior distribution")
#Gaussian plotting
mu=0.0007
sigma=0.0013
x=np.linspace(np.min(c_samples),np.max(c_samples),1000) 
#normstdis=np.zeros((1000,1))
normstdis=1/(np.sqrt(2*(sigma**2)*mth.pi))*np.exp(-((x-mu)**2)/(2*sigma**2))
plt.plot(x,normstdis,label='normal distribution fitted')
'''
'''
for i in range(n_dist_bins):
    
    plt.suptitle('Chain for each walker')
    p=plt.subplot(5,2,i+1)
    
    
    plt.title('Walker %i'%(i+1),fontsize=10)
    plt.rc('font', **{'size':'10'})
    plt.plot(sampler.chain[i,:,:])
'''
    
'''
# Choose the "true" parameters.
m_true = -0.9594
b_true = 4.294
f_true = 0.534

# Generate some synthetic data from the model.
N = 50
x = np.sort(10*np.random.rand(N))
yerr = 0.1+0.5*np.random.rand(N)
y = m_true*x+b_true
y += np.abs(f_true*y) * np.random.randn(N)
y += yerr * np.random.randn(N)

plt.scatter(x,y)
plt.plot(x,m_true*x+b_true)
'''
