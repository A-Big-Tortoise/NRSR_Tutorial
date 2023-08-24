import numpy as np
from PyEMD import EEMD, EMD
# from vmdpy import VMD
from statsmodels.tsa.seasonal import seasonal_decompose

"""
VMD每个参数究竟表示什么含义
VMD的三个输出究竟都代表什么
"""

def emd_decomposition(signal):
   emd = EMD()
   imfs = emd(signal)
   return imfs

def eemd_decomposition(signal, noise_width=0.05, ensemble_size=100):
   eemd = EEMD(trails=ensemble_size, noise_width=noise_width)
   imfs = eemd.eemd(signal)
   return imfs

def vmd_decomposition(signal, K, alpha=2000, tau=0, DC=0, init=1, tol=1e-7):
   """
   K: how many modes 
   alpha: moderate bandwidth constraint  
   tau: noise-tolerance (no strict fidelity enforcement) 
   DC: whether have DC part imposed  
   init: initialize omegas uniformly 
   tol:

   Reference:
    link: https://vmd.robinbetz.com/
   """
   u, u_hat, omega = VMD(signal, alpha, tau, K, DC, init, tol)
   return u, u_hat, omega

def seasonal_decomposition(signal, period, model="addative"):
   """
   Parameters:
    model : "addative" or "multiplicative"
    period : Period of the series.
   returns : 
    results: get values of results by
    result.seasonal, result.trend, result.resid
   """
   result = seasonal_decompose(signal, model=model, period=period)
   return result
