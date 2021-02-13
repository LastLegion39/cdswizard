#!/usr/bin/env python
# coding: utf-8

# # Q2

# In[1]:


import numpy as np 
import math 
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import minimize
import scipy.integrate as integrate


# In[2]:


def getCumulLambdas(r, t, rec, dSpread, payment):
    
    """
    returns a list (or list of lists) of cumulative hazard rates
    
    r: interest rate
    t: tenor of CDS
    rec: recovery rate
    dSpread: spread of CDS (please enter it by dividing it by 10000 (bps))
    payment: 4 if quarterly, 1 if annually etc
    
    """
    
    vPaymentTimes = np.linspace(0,t,(t*payment)+1)
    
    dInitialLambda = dSpread / (1-rec) 
    
    def calcDiff(h, r, t, vPaymentTimes, rec, dSpread, payment):
        
        vPremiumPayments = np.ones(len(vPaymentTimes))
        for j,i in enumerate(vPaymentTimes):
            vPremiumPayments[j] = np.exp(-r*i) * (1/payment) * np.exp(-h*i) # h = cumulative_lambda (dInitialLambda at first but it converts to its optimal value which is --> cumulative lambda)
        dPremiumPayment = np.sum(vPremiumPayments) # summation of premium payments times survival prob (discounted)
        
        def calcIntegrandAccrued(u, r, h):
            dIntegrandAccrued = np.exp(-r*u) * ((1/payment)/2) * h * np.exp(-h*u)
            return dIntegrandAccrued
        
        vIaccrued = integrate.quad(calcIntegrandAccrued, 0, t, args=(r, h))
        dAccruedPayment = vIaccrued[0]  
        
        def calcIntegrandProtection(u, r, h):
            dIntegrandProtection = np.exp(-r*u) * h * np.exp(-h*u)
            return dIntegrandProtection
        
        vIprotection = integrate.quad(calcIntegrandProtection, 0, t, args=(r, h))
        dProtectionPayment = vIprotection[0]
        
        dPremiumLeg = dSpread * (dPremiumPayment + dAccruedPayment)
        dProtectionLeg = (1 - rec) * dProtectionPayment 
        
        return dPremiumLeg - dProtectionLeg # difference function. this should be apprx equal to 0
    
    diff = lambda x: calcDiff(x, r, t, vPaymentTimes, rec, dSpread, payment)
    cumulative_lambda = minimize(diff, dInitialLambda, method='Nelder-Mead', tol = 0.1).x
    
    return cumulative_lambda # lambda between year 0 and 1 OR 0-3 OR 0-5 etc

