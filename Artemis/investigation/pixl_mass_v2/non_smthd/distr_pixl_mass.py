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

bins=500
data=np.histogram(a,bins=bins,density=True)
bin_vals=np.delete(data[1],len(data[1])-1,0)
plt.plot(np.log10(bin_vals),data[0])

plt.xlabel('Mass') 
plt.ylabel( 'p(mass)')    
plt.savefig('/scratch/GAMNSCM2/bolchoi_z0/investigation/mass_distribution_pxl.png')
