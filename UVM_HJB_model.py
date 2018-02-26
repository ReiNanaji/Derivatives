# -*- coding: utf-8 -*-
"""
Created on Mon Feb 26 12:25:44 2018

@author: f619871
"""

import numpy as np

def get_regular_grid(Tstart, Tend, Tstep, Xstart, Xend, Xstep, Ystart, Yend, Ystep):
    
    ''' 
        Assumption: Xstep = Ystep    
    '''    
    
    tgrid = np.linspace(Tstart, Tend, Tstep)
    xgrid = np.linspace(Xstart, Xend, Xstep)
    ygrid = np.linspace(Ystart, Yend, Ystep)
    
    t = np.tile( tgrid, ( Xstep, Ystep, Tstep ) )   
    x = np.tile( xgrid, ( Xstep, Ystep, Tstep ) ) 
    y = np.tile( ygrid, ( Xstep, Ystep, Tstep ) ) 
    
    grid = {
    'T': {'start':Tstart, 'end':Tend, 'step':Tstep, 'incr':tgrid[1] - tgrid[0], 'matrix': t},
    'X': {'start':Xstart, 'end':Xend, 'step':Xstep, 'incr':xgrid[1] - xgrid[0], 'matrix': x},
    'Y': {'start':Ystart, 'end':Yend, 'step':Ystep, 'incr':ygrid[1] - ygrid[0], 'matrix': y}     
    }    
    return grid
    
def zero_curve(t):
    return 0.01*np.sqrt(t) - 0.001*t + 0.05
    
def volatility_matrix(X, smin, smax, upper=True):
    if upper:
        return smin * (X < 0) + smax * (X >= 0)
    else:
        return smax * (X < 0) + smin * (X >= 0)
    
    
def UVMHJB_pde_pricer(smin, smax, beta, zero_curve, payoff, grid, upper=True):
    # Define the constant 
    A = np.diag(np.ones(grid['X']['step'] - 1), k=1)
    B = np.diag(np.ones(grid['X']['step'] - 1), k=-1)
    
    dt = grid['T']['incr']
    dx = grid['X']['incr']
    dy = grid['Y']['incr']
    
    N = grid['T']['step']
    
    F = zero_curve(grid['T']['matrix'])            
    X = grid['X']['matrix']
    Y = grid['Y']['matrix']   
    
    # Initialize Pricing Matrix
    U = np.zeros((grid['X']['step'], grid['Y']['step'], grid['T']['step']))
    U[:,:,-1] = payoff(X[:,:,-1], Y[:,:,-1])    
    
    # Solve the PDE
    for t in range(N-2, -1, -1):
        print(t)
        M = dt / dx**2 * (np.dot(A, U[:,:,t+1]) - 2 * U[:,:,t+1] + np.dot(B, U[:,:,t+1])) + 2 * dt / dy * (np.dot(U[:,:,t+1], B) - U[:,:,t+1])   
        V = volatility_matrix(M, smin, smax, upper)
        U[:,:,t] = U[:,:,t+1] - 2 * beta * dt / dy * Y[:,:,t+1] * (np.dot(U[:,:,t+1], B) - U[:,:,t+1]) + \
        dt / dx * (Y[:,:,t+1] - beta * X[:,:,t+1]) * (np.dot(A, U[:,:,t+1]) - U[:,:,t+1]) + \
        0.5 * V * M - (F[:,:,t+1] - X[:,:,t+1]) * U[:,:,t+1]
    
    return {'grid':grid, 'zero':F, 'price':U}

def bond(X, Y):
    return np.ones(X.shape)
 

Tstart, Tend, Tstep = 0, 1, 11
Xstart, Xend, Xstep = 0, 1, 11
Ystart, Yend, Ystep = 0, 1, 11

smin, smax = 0.1, 0.2
beta = 0.05
    
grid = get_regular_grid(Tstart, Tend, Tstep, Xstart, Xend, Xstep, Ystart, Yend, Ystep)

res = UVMHJB_pde_pricer(smin, smax, beta, zero_curve, bond, grid, upper=True)
