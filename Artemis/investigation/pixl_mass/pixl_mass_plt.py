import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import h5py

grid_nodes=850
f1=h5py.File("/scratch/GAMNSCM2/bolchoi_z0/my_den/den_grid%s_halo_bin_bolchoi" %grid_nodes, 'r')
a=f1['/halo'][:]
f1.close()
a=a.flatten()
min_mass=np.min(a)
max_mass=np.max(a)
no_mass_bins=100
mass_intvl=(max_mass-min_mass)/no_mass_bins

pxl_qty=np.zeros(no_mass_bins)
sum_pxls=np.zeros(no_mass_bins)
for i in range(no_mass_bins):
    intvl_low=min_mass+mass_intvl*i
    intvl_hi=intvl_low+mass_intvl
    mask=np.zeros(len(a))
    filt_low=np.where(a>=intvl_low)
    filt_hi=np.where(a<intvl_hi)
    mask[filt_low]=1
    mask[filt_hi]+=1
    pixls=np.where(mask==2)
    x,y=np.shape(pixls)
    sum_pxls[i]=sum(a[pixls])
    pxl_qty[i]=y
    
#normalize data
pxl_qty_norm=(pxl_qty-np.min(pxl_qty))/(np.max(pxl_qty)-np.min(pxl_qty))
sum_pxls_norm=(sum_pxls-np.min(sum_pxls))/(np.max(sum_pxls)-np.min(sum_pxls))
mass_bins=np.arange(min_mass,max_mass,mass_intvl)
#plot
plt.figure(figsize=(10,7),dpi=100)
ax1=plt.subplot2grid((1,1), (0,0)) 
mass_bins=np.arange(min_mass,max_mass,mass_intvl)
plt.plot(mass_bins,pxl_qty_norm,label='pixels qty')
plt.plot(mass_bins,sum_pxls_norm,label='pixels total')
plt.title('pixels containing mass')    
plt.xlabel('M_solar masses') 
plt.ylabel('pixels qty')    
plt.legend()
plt.title('pxl_qty:%s sum_pxls:%s'%(pxl_qty[0],sum_pxls[0]))#perhaps do this for not just the first bin
plt.savefig('/scratch/GAMNSCM2/bolchoi_z0/investigation/bolchoi_pxl_mass_plt.png')
