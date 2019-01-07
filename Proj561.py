#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 28 21:07:26 2018

Description: Within this project, we explored Markowitz Portfolio Theory. 
Particularly, we we use mean-variance analysis to obtain an optimal trade-off between expectedd risk and expected return

@author: Jingrong Tian, Zhengqian Xu, Ling Yang, Zikai Zhu

"""

#matplotlib inline
import numpy as np
import matplotlib.pyplot as plt

#import cvxopt as opt
#from cvxopt import blas, solvers

import pandas as pd

#read in file
import cvxopt as opt
from cvxopt import blas, solvers

#############################################################
####       Define function to calculate parameters       ####
#############################################################

## Produce the weight vector for each stock
def spweights(n):
    ''' Produces n random weights that sum to 1 '''
    k = np.random.rand(n)
    return k / sum(k)

## Get expected return and risk for the combination of stock 
def sp_portfolio(returns):
    ''' 
    Returns the mean and standard deviation of returns for a random portfolio
    '''
    ## p is the mean returns for each stock 
    ## w is the weight vector of the portfolio
    ## C is the  covariance matrix of the returns
    p = np.asmatrix(np.mean(returns, axis=1))
    w = np.asmatrix(spweights(returns.shape[0]))
    C = np.asmatrix(np.cov(returns))
    
    ## mu is the expected portfolio returns 
    mu = w * p.T
    ## sigma represents the expected portfolio risk.
    sigma = np.sqrt(w * C * w.T)
    '''
    # This recursion reduces outliers to keep plots pretty
    if sigma > 2:
        return sp_portfolio(returns)
    '''
    return mu, sigma


def optimal_portfolio(returns):
    ## the number of stock
    n = len(returns)
    returns = np.asmatrix(returns)
    
    
    ## different weights
    N = 100
    mus = [10**(5.0 * t/N - 1.0) for t in range(N)]
    
    # Convert to cvxopt matrices
    S = opt.matrix(np.cov(returns))
    pbar = opt.matrix(np.mean(returns, axis=1))
    
    # Create constraint matrices
    G = -opt.matrix(np.eye(n))   # negative n x n identity matrix
    h = opt.matrix(0.0, (n ,1))
    A = opt.matrix(1.0, (1, n))
    b = opt.matrix(1.0)
    
    # Calculate efficient frontier weights using quadratic programming
    portfolios = [solvers.qp(mu*S, -pbar, G, h, A, b)['x'] 
                  for mu in mus]
    ## calculate risks and returns for efficient frontier
    returns = [blas.dot(pbar, x) for x in portfolios]
    risks = [np.sqrt(blas.dot(x, S*x)) for x in portfolios]
   
    
    m1 = np.polyfit(returns, risks, 2)
    x1 = np.sqrt(m1[2] / m1[0])
    
    ## calculate the optimal portfolio
    wt = solvers.qp(opt.matrix(x1 * S), -pbar, G, h, A, b)['x']
    return np.asarray(wt), returns, risks



#############################################################
####        Apply MPT to each possible combination       ####
#############################################################

## combination of all risk levels
def get_total():
    df = pd.read_csv('risk.csv' , sep=',', encoding='latin1')
    return_vec = np.array([pd.Series(df["low"]).values,
                       pd.Series(df["median"]).values,
                       pd.Series(df["high"]).values])
    means, stds = np.column_stack([
        sp_portfolio(return_vec) 
        for _ in range(n_portfolios)
    ])
    
    weights, returns, risks = optimal_portfolio(return_vec)
        
    plt.plot(stds, means, 'o', markersize=5)
    plt.xlabel('std')
    plt.ylabel('mean')
    plt.plot(risks, returns, 'y-o')
    plt.title('Mean and standard deviation of returns')
    fig = plt.gcf()
    fig.set_size_inches(8, 5, forward=True)
    fig.savefig('mix_total.png', dpi=100)
    
    return weights


##  two low-risk companies vs. two median-risk vs. two high-risk as a general analysis
def get_mixed(df):
    return_vec = np.array([pd.Series(df["AGG"]).values,
                       pd.Series(df["LQD"]).values,
                       pd.Series(df["KRE"]).values,
                       pd.Series(df["USO"]).values,
                       pd.Series(df["DFE"]).values,
                       pd.Series(df["EFA"]).values])
    means, stds = np.column_stack([
        sp_portfolio(return_vec) 
        for _ in range(n_portfolios)
    ])
    
    weights, returns, risks = optimal_portfolio(return_vec)
        
    plt.plot(stds, means, 'o', markersize=5)
    plt.xlim(0, 0.025) 
    plt.ylim(0, 0.0006)
    plt.xlabel('std')
    plt.ylabel('mean')
    plt.plot(risks, returns, 'y-o')
    plt.title('Mean and standard deviation of returns')
    fig = plt.gcf()
    fig.set_size_inches(8, 5, forward=True)
    fig.savefig('mix_low+high+mid.png', dpi=100)
    plt.clf()

## three low-risk vs. three median-risk as analysis for low-risk and median-risk companies
def lowmean(df):
    return_vec1 = np.array([pd.Series(df["SPY"]).values,
                       pd.Series(df["DBC"]).values,
                       pd.Series(df["XLP"]).values,
                       pd.Series(df["IBB"]).values,
                       pd.Series(df["XLB"]).values,
                       pd.Series(df["EEM"]).values])
    means1, stds1 = np.column_stack([
        sp_portfolio(return_vec1) 
        for _ in range(n_portfolios)
    ])

    weights1, returns1, risks1 = optimal_portfolio(return_vec1)
    plt.xlabel('std')
    plt.ylabel('mean')
    #low-median
    plt.plot(stds1, means1, 'o', markersize=5)
    plt.plot(risks1, returns1, 'y-o')
    plt.title('Mean and standard deviation of returns of randomly generated portfolios')
    fig = plt.gcf()
    fig.set_size_inches(8, 5, forward=True)
    fig.savefig('low+median.png', dpi=100)
    plt.clf()
 
## three low-risk vs. three high-risk as analysis for low-risk and high-risk companies.
def lowhigh(df):
    #combine low/high
    return_vec2 = np.array([pd.Series(df["SPY"]).values,
                           pd.Series(df["DBC"]).values,
                           pd.Series(df["XLP"]).values,
                           pd.Series(df["OIL"]).values,
                           pd.Series(df["SH"]).values,
                           pd.Series(df["SDS"]).values])
    means2, stds2 = np.column_stack([
        sp_portfolio(return_vec2) 
        for _ in range(n_portfolios)
    ])
    weights2, returns2, risks2 = optimal_portfolio(return_vec2)  
    #low-high
    plt.plot(stds2, means2, 'o', markersize=5)
    plt.plot(risks2, returns2, 'y-o')
    plt.title('Mean and standard deviation of returns of randomly generated portfolios')
    fig = plt.gcf()
    fig.set_size_inches(8, 5, forward=True)
    fig.savefig('low+high.png', dpi=100)
    plt.clf()
    
    
##  three median-risk vs. three high-risk as analysis for median-risk and high-risk companies
def medhigh(df):
    return_vec3 = np.array([pd.Series(df["IBB"]).values,
                       pd.Series(df["XLB"]).values,
                       pd.Series(df["EEM"]).values,
                       pd.Series(df["OIL"]).values,
                       pd.Series(df["SH"]).values,
                       pd.Series(df["SDS"]).values])
    means3, stds3 = np.column_stack([
        sp_portfolio(return_vec3) 
        for _ in range(n_portfolios)
    ])
    weights3, returns3, risks3 = optimal_portfolio(return_vec3) 
    #median-high
    plt.plot(stds3, means3, 'o', markersize=5)
    plt.plot(risks3, returns3, 'y-o')
    plt.title('Mean and standard deviation of returns of randomly generated portfolios')
    fig = plt.gcf()
    fig.set_size_inches(8, 5, forward=True)
    fig.savefig('high+median.png', dpi=100)
    

## six low-risk companies as analysis for medium-risk companies.
def medium(df):
    return_vec = np.array([pd.Series(df["DFE"]).values,
                       pd.Series(df["IBB"]).values,
                       pd.Series(df["EFA"]).values,
                       pd.Series(df["IWM"]).values,
                       pd.Series(df["XLB"]).values,
                       pd.Series(df["EWU"]).values
                       ])
   

    means, stds = np.column_stack([
            sp_portfolio(return_vec) 
            for _ in range(n_portfolios)
            ])

    weights, returns, risks = optimal_portfolio(return_vec)
        
    plt.plot(stds, means, 'o', markersize=3)
    plt.xlabel('std')
    plt.ylabel('mean')
    #plt.xlim(0,0.025)
    #plt.ylim(-0.00025,0.0006)
    plt.grid()
    plt.title('Mean and standard deviation of returns of randomly generated portfolio of median risk')
    plt.plot(risks, returns, 'y-o')
    fig = plt.gcf()
    fig.set_size_inches(8, 5, forward=True)
    fig.savefig('median.png', dpi=100)


## six low-risk companies as analysis for high-risk companies.
def high(df):
    return_vec1 = np.array([pd.Series(df["KRE"]).values,
                       pd.Series(df["XHB"]).values,
                       pd.Series(df["USO"]).values,
                       pd.Series(df["VNQ"]).values,
                       pd.Series(df["XLF"]).values,
                       pd.Series(df["OIL"]).values])
   

    means1, stds1 = np.column_stack([
            sp_portfolio(return_vec1) 
            for _ in range(n_portfolios)
    ])

    weights1, returns1, risks1 = optimal_portfolio(return_vec1)

    ### different portfolio     
    plt.plot(stds1, means1, 'o', markersize=3,color = 'red')
    #plt.xlim(0,0.025)
    #plt.ylim(-0.00025,0.0006)
    plt.xlabel('std')
    plt.ylabel('mean')
    plt.grid()
    plt.title('Mean and standard deviation of returns of randomly generated portfolio pf high risk')
    ### efficient portfolio
    plt.plot(risks1, returns1, 'y-o')
    fig = plt.gcf()
    fig.set_size_inches(8, 5, forward=True)
    fig.savefig('high.png', dpi=100)


## six low-risk companies as analysis for low-risk companies.
def low(df):
    return_vec2 = np.array([pd.Series(df["AGG"]).values,
                           pd.Series(df["LQD"]).values,
                           pd.Series(df["XLP"]).values,
                           pd.Series(df["XLV"]).values,
                           pd.Series(df["XLU"]).values,
                           pd.Series(df["DIA"]).values
                           ])
       
    
    means2, stds2 = np.column_stack([
        sp_portfolio(return_vec2) 
        for _ in range(n_portfolios)
    ])
    
    weights2, returns2, risks2 = optimal_portfolio(return_vec2)
        
    plt.plot(stds2, means2, 'o', markersize=3,color = 'green')
    #plt.xlim(0,0.025)
    #plt.ylim(-0.00025,0.0006)
    plt.xlabel('std')
    plt.ylabel('mean')
    plt.plot(risks2, returns2, 'y-o')
    plt.grid()
    plt.title('Mean and standard deviation of returns of randomly generated portfolio of low risk')
    fig = plt.gcf()
    fig.set_size_inches(8, 5, forward=True)
    fig.savefig('low.png', dpi=100)

## main function
if __name__ == "__main__":
    n_portfolios = 502
    df = pd.read_csv('returns.csv' , sep=',', encoding='latin1')
    get_mixed(df)
    weights=get_total()
    lowmean(df)
    lowhigh(df)
    medhigh(df)
    medium(df)
    high(df)
    low(df)
    
    
       
    
    