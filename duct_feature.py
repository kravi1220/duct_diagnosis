import numpy as np
import netCDF4
from datetime import datetime
import sys
import os
import xarray as xr
import pandas as pd
import glob
import seawater as sw
import scipy
from scipy.signal import find_peaks
import bottleneck as bn
from ismember import ismember
from scipy.signal import argrelextrema
from scipy import interpolate
from dask.diagnostics import ProgressBar

def interpolate_iD_var(x,y,x_new):
    f = interpolate.interp1d(x, y)
    var=f(x_new)
    return var
    
def duct_feature(svel,temp,salt,u,v,z):
    svel_new=np.array(svel)
    svel   = np.array(svel)
    nan_index = np.where(np.isnan(svel))

    if (len(nan_index[0])>0):
        if(nan_index[0][0]>2):
            #print(nan_index[0][0])
            svel[nan_index[0][0]] = svel[nan_index[0][0]-2]
    else:
       svel[-1] = svel[-2]-2.0
    depth_thresh=500
    duct1_0 = np.nan
    duct1_1 = np.nan
    duct1_2 = np.nan
    duct1_3 = np.nan
    duct1_4 = np.nan
    duct1_5 = np.nan
    aduct1_0 = np.nan
    aduct1_1 = np.nan
    aduct1_2 = np.nan
    aduct1_3 = np.nan
    aduct1_4 = np.nan
    aduct1_5 = np.nan
    duct_d_out= np.nan
    duct_c_out=np.nan
    duct_w_out=np.nan
    duct_p_out=np.nan
    duct_s_out=np.nan
    duct_0 = np.nan
    aduct_0 = np.nan
    duct_d_out=np.nan
    duct_c_out=np.nan
    duct_w_out=np.nan
    duct_p_out=np.nan
    duct_s_out=np.nan
    duct_pt_out=np.nan
    duct_sal_out=np.nan
    duct_u_out=np.nan
    duct_v_out=np.nan
    aduct_d_out=np.nan
    aduct_c_out=np.nan
    aduct_w_out=np.nan
    aduct_p_out=np.nan
    aduct_s_out=np.nan
    aduct_pt_out=np.nan
    aduct_sal_out=np.nan
    aduct_u_out=np.nan
    aduct_v_out=np.nan
    aduct_svel_out=np.nan
    duct_svel_out=np.nan
    depth_above_duct_have_eq_cvel_toaduct=np.nan
      
    [index,peaks]=find_peaks(-1*svel)
    a    = z[index]
    b    = svel[index]#peaks['peak_heights']

    if len(a)>0:
        duct_0 = a
        duct_1 = b
    ikeep = np.where(np.logical_and(duct_0<depth_thresh,duct_0>10));
    if len(ikeep[0])>0:
        duct1_0 = duct_0[ikeep[0]]
        duct1_1 = duct_1[ikeep[0]]
        index_new = index[ikeep[0]]
         
    [index_aduc,peaks]=find_peaks(svel)
    ad    = z[index_aduc]
    bd    = svel[index_aduc]
    if len(a)>0:
        aduct_0 = ad
        aduct_1 = bd
        
        
 ##########################
 #if len(ikeep[0])>0:   
#        if any(np.logical_and(np.array(aduct_0)>np.array(duct1_0[0]),np.array(aduct_0)<=depth_thresh)):
#            ican = np.where(np.logical_and(np.array(aduct_0)<=depth_thresh, np.array(aduct_0)>np.array(duct1_0[0])))
#        else:
            #print('cb',len(np.array(aduct_0)[np.where(np.array(aduct_0)>depth_thresh)]))
#            if(len(np.array(aduct_0)[np.where(np.array(aduct_0)>depth_thresh)])>0):
 #               threshold = min(1000,np.min(np.array(aduct_0)[np.where(np.array(aduct_0)>depth_thresh)]))
                #print(threshold)
 #           else:
 #               threshold=depth_thresh
            #print(aduct_0,np.array(aduct_0)[np.where(np.array(aduct_0)>depth_thresh)])
  #          ican = np.where(np.logical_and(np.array(aduct_0)<= threshold, np.array(aduct_0)>depth_thresh))        
 ################
    if len(ikeep[0])>0:    
        ican = np.where(np.logical_and(np.array(aduct_0)<depth_thresh, np.array(aduct_0)>np.array(duct1_0[0])));
    
        if len(ican[0])>0:
            aduct1_0 = aduct_0[ican[0]]
            aduct1_1 = aduct_1[ican[0]]
            index_aduc_new = index_aduc[ican[0]]

        ikeep = np.where(np.logical_and(duct_0<depth_thresh,duct_0>10));
            
        if (len(ican[0])>0 and len(ikeep[0])>0): 
            dum_ad= np.zeros((len(duct1_0),len(aduct1_0)))
            for ii in range(len(duct1_0)):
                for jj in range(len(aduct1_0)):
                    dum_ad[ii,jj] = min(0,(duct1_1[ii]-aduct1_1[jj]))*min(0,(duct1_0[ii]-aduct1_0[jj]))
            if np.any(dum_ad)>0:
                [idx_d,idx_ad]=np.where(np.array(dum_ad)==np.amax(dum_ad))
                if len(idx_ad)>0:
                    if len(idx_ad)>1:
                        id=[idx_ad[0]]
                    else:
                        id = np.array(idx_ad)
  
                        aduct_d_out = aduct1_0[id][0]
                        aduct_c_out = aduct1_1[id][0]
                        ad    = scipy.signal.peak_prominences(svel_new,index_aduc_new[id])
                        aw    = scipy.signal.peak_widths(svel_new,index_aduc_new[id],1.0,ad)
                        al    = np.linspace(0,len(z)-1,len(z))
                        af    = interpolate.interp1d(al, z)
                        ac   = af(aw[2:])
                
                        aduct_p_out = ad[0][0]
                        aduct_w_out = abs(ac[0]-ac[1])[0]
                        aduct_s_out= (ad[0]*abs(ac[0]-ac[1])[0])[0]
                        aduct_pt_out = interpolate_iD_var(z,temp,aduct1_0[id][0]);
                        aduct_sal_out= interpolate_iD_var(z,salt,aduct1_0[id][0]);
                        aduct_u_out = interpolate_iD_var(z,u,aduct1_0[id][0]);
                        aduct_v_out = interpolate_iD_var(z,v,aduct1_0[id][0]);
                        aduct_svel_out =  interpolate_iD_var(z,svel,aduct1_0[id][0]);
                if len(idx_d)>0:
                    if len(idx_d)>1:
                        idd=[idx_d[0]]
                    else:
                        idd = np.array(idx_d)
                
                        duct_d_out = duct1_0[idd][0]
                        duct_c_out = duct1_1[idd][0]
                        #index_new = np.where(svel==np.array(duct1_1[idd][0]))[0]
                        #svel_new[np.where(svel_new==np.array(aduct1_1[idd][0]))[0][0]:]=aduct1_1[idd][0]
                        svel_new[index_aduc_new[id][0]:] = aduct1_1[id][0]
                        d    = scipy.signal.peak_prominences(-1*svel_new,index_new[idd])
                        w    = scipy.signal.peak_widths(-1*svel_new,index_new[idd],1.0,d)
                        l    = np.linspace(0,len(z)-1,len(z))
                        f    = interpolate.interp1d(l, z)
                        c   = f(w[2:])
                
                        duct_p_out = d[0][0]
                        duct_w_out = abs(c[0]-c[1])[0]
                        duct_s_out= (d[0]*abs(c[0]-c[1])[0])[0]

                        duct_pt_out = interpolate_iD_var(z,temp,duct1_0[idd][0]);
                        duct_sal_out= interpolate_iD_var(z,salt,duct1_0[idd][0]);
                        duct_u_out = interpolate_iD_var(z,u,duct1_0[idd][0]);
                        duct_v_out = interpolate_iD_var(z,v,duct1_0[idd][0]);
                        duct_svel_out =  interpolate_iD_var(z,svel,duct1_0[idd][0]);
            if (len(ican)>0 and len(ikeep)>0):
                depth_idx_above_duct = np.where(z<duct_d_out)
                #c_index_above_duct = np.where((svel[depth_idx_above_duct]-aduct_svel_out)<=0.10)
                    
                if len(depth_idx_above_duct[0])>0:
            
                    depth_above_duct_have_eq_cvel_toaduct=z[max(depth_idx_above_duct[0])]
    out=[duct_d_out,duct_c_out,duct_w_out,duct_p_out,duct_s_out,duct_pt_out,duct_sal_out,duct_u_out,duct_v_out,duct_svel_out,aduct_d_out,aduct_c_out,aduct_w_out,aduct_p_out,aduct_s_out,aduct_pt_out,aduct_sal_out,aduct_u_out,aduct_v_out,duct_svel_out,depth_above_duct_have_eq_cvel_toaduct]
    
    return np.array(out)
