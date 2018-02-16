import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import h5py
import matplotlib.ticker as ticker

grid_nodes=850
f1=h5py.File("/scratch/GAMNSCM2/bolchoi_z0/my_den/den_grid%s_halo_bin_bolchoi" %grid_nodes, 'r')
a=f1['/halo'][:]
f1.close()
a=a.flatten()
a[::-1].sort()#sort in descending order

pxl_bunch=3070625
data_points=len(a)/pxl_bunch
cumltv_mass=np.zeros(data_points)
pixl_bins=np.linspace(pxl_bunch,len(a),data_points)

for i in range(data_points):
    cumltv_mass[i]=sum(a[0:int(pixl_bins[i])])
  
cumltv_mass=cumltv_mass/sum(a)*100 
pixl_bins=pixl_bins/len(a)*100

#plot
plt.figure(figsize=(10,7),dpi=100)
ax1=plt.subplot2grid((1,1), (0,0)) 
plt.loglog(pixl_bins,cumltv_mass) 
start, end = ax1.get_ylim()
ax1.yaxis.set_ticks(np.arange(start, end, 5))
ax1.yaxis.set_major_formatter(ticker.FormatStrFormatter('%0.1f'))
plt.xlabel('pixels (%)') 
plt.ylabel( 'Mass (%)')    
plt.savefig('/scratch/GAMNSCM2/bolchoi_z0/investigation/bolchoi_mass_perc_pixl_perc_loglog_2.png')
