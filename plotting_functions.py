# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import pylab as plt 

import seaborn as sns
from sklearn.model_selection import train_test_split

from scipy.stats import binned_statistic, binned_statistic_2d
from matplotlib.colors import BoundaryNorm, LogNorm
from statsmodels.stats.proportion import proportion_confint
import sys

    
  
  # for validation
  # transcript=transcript.loc[transcript.time<600,:]
  # print('transcript.shape:',transcript.shape)

def plot_transactions(df,point_color='k',point_alpha=0.1, bin_color='r',bin_alpha=0.5, day_bins=True, week_bins=True, bin_week_color='g'):
  plt.plot(df['time'],df['payments'],'.',color=point_color,alpha=point_alpha)

  if day_bins:
    bin_meansD, bin_edgesD, binnumber = binned_statistic(df['time'],df['payments'], bins=range(0,31*24,24))
    plt.hlines(bin_meansD, bin_edgesD[:-1], bin_edgesD[1:], colors=bin_color, lw=3,alpha=bin_alpha)

  if week_bins:
    bin_means, bin_edges, binnumber = binned_statistic(df['time'],df['payments'], bins=range(0,31*24,24*7))
    plt.hlines(bin_means, bin_edges[:-1], bin_edges[1:], colors=bin_week_color, lw=5,alpha=bin_alpha);


def bin_transactions(df,xcol,ycol,xedges,yedges,count_bins,cmap='gnuplot',ylog=True,
                     xlabel='hours',ylabel_left='log10 purchase',ylabel_right='purchase [$]'):
  # plots 2d histogram of counts by time bin and transaction amount bin (log)
  
  fig, ax= plt.subplots()
  if ylog:
    yedges_bin=np.log10(yedges)
    yvalues=np.log10( df[ycol] )
  else:      
    yedges_bin=yedges
    yvalues=df[ycol]
    
  H, xedgesH, yedgesH = np.histogram2d(df[xcol],yvalues,bins=(xedges, yedges_bin))
  H=H.T

  norm = BoundaryNorm(count_bins,256) 
  X, Y = np.meshgrid(xedgesH, yedges_bin)
  cf=ax.pcolormesh(X, Y, H,cmap=plt.get_cmap(cmap) , norm=norm,zorder=1)
  
  plt.ylabel(ylabel_left)
  
#   ax1=ax.twinx()
#   ax1.set_yticks(yedgesH)
#   yticks=['%g' % x for x in yedges]
#   ax1.set_yticklabels(yticks)

  ax.set_yticks(yedgesH)
  yticks=['%g' % x for x in yedges]
  ax.set_yticklabels(yticks)
    
  plt.xlabel(xlabel)
  plt.ylabel(ylabel_right)

  cbaxes = fig.add_axes([0.99, 0.1, 0.03, 0.8]) 

  cticks=['%d' % x for x in count_bins]


  cbar=fig.colorbar(cf,cax = cbaxes, spacing='uniform',norm=norm)
  cbar.set_ticks(count_bins)
  cbar.set_ticklabels(cticks)
  cbar.set_label('counts');

  #ax.vlines(xedges,Y.min(),Y.max(),lw=1,color='w',zorder=2);
  
#   return fig,ax,ax1,Y.min(),Y.max() 
  return fig,ax,Y.min(),Y.max() 


def plot_bn_offer(offer_rvc,num,den,alpha=0.05,xleg=1.1,**kwargs):
  # plots confidence intervals for offers type (a,b,c) vs date 
  
  cl,cu=proportion_confint(offer_rvc[num],offer_rvc[den],alpha=alpha)
  cmean=offer_rvc[num]/offer_rvc[den]

  cmean=cmean.unstack().T
  cl=cl.unstack().T
  cu=cu.unstack().T

  x=cmean.columns.values
  for off in cmean.index:
    ym=cmean.loc[off,:].values
    lb=cl.loc[off,:].values
    ub=cu.loc[off,:].values


    yerr_l = np.abs(lb - ym)
    yerr_u = np.abs(ub - ym)
    yerr = np.vstack((yerr_l, yerr_u))  
    plt.errorbar(x, ym, yerr=yerr,label=off,**kwargs)


  plt.legend(bbox_to_anchor=(xleg, 0.8)); 

  

def rvc_factor_plot(df,factor,xleg=1.15,**kwargs):
  rvc=df.groupby(by=[factor,'offer'])['received','viewed','completed'].agg('sum')

  plt.figure(figsize=(15,4));

  plt.subplot(121)
  plot_bn_offer(rvc,'viewed','received',alpha=0.05,xleg=xleg,**kwargs);
  rvc['vr']=rvc['viewed']/rvc['received']
  plt.xlabel(factor);
  plt.ylabel('fraction');
  plt.title('viewed/received @ alpha=0.05 ');


  plt.subplot(122)
  plot_bn_offer(rvc,'completed','viewed',alpha=0.05,xleg=xleg,**kwargs);
  plt.xlabel(factor);
  plt.title('completed/viewed @ alpha=0.05 ');
  

def plot_offers_of_customer(piv,customer,yc=0,step=0.1,plot_transactions=True,plot_offer_name=True):
  # plots the offers and the transactions for a customer
  
  gg=piv.loc[(piv.person==customer),'offer_received' ].drop_duplicates()
  ind=gg.index.values

  viewed_offers=[]

  v_off_times=np.zeros((ind.shape[0],2))
  v_off_set=set([])

  c=0

  for n in ind:
    hh=plt.plot([piv.loc[n,'offer_received'],piv.loc[n,'offer_end']  ],[yc,yc],'o-',alpha=0.6)
    color=hh[0].get_color()
    
    if plot_offer_name:
      plt.text(  (piv.loc[n,'offer_received']+piv.loc[n,'offer_end'])/2,yc-.2, piv.loc[n,'offer'] );
    
    if not np.isnan(piv.loc[n,'offer_viewed']):
      plt.plot([piv.loc[n,'offer_viewed'],piv.loc[n,'offer_end']  ],[yc,yc],'.-',alpha=1.0,color=color)
      viewed_offers.append(piv.loc[n,'offer'] )
      v_off_times[c,0]=piv.loc[n,'offer_viewed']
      v_off_times[c,1]=piv.loc[n,'offer_end']
      v_off_set=v_off_set.union(set( range(int(piv.loc[n,'offer_viewed']), int(piv.loc[n,'offer_end'])+0) ))
      c+=1


    yc+=step
  v_off_times=v_off_times[0:c,:]

  diff=np.diff(np.array(list(v_off_set)) )
  diff[diff>1]=0
  overlapping_offers_time=diff.sum()

  if plot_transactions:
    df=piv.loc[(piv.person==customer)]
    plt.plot(df['time'],df['payments'],'o',color='k',alpha=0.3)
  
  plt.xlabel('hours');
  plt.ylabel('purchases');
  
  
  return(viewed_offers,v_off_times,overlapping_offers_time)
