# -*- coding: utf-8 -*-
"""
Created on Mon Feb 26 12:25:44 2018

@author: f619871
"""

import numpy as np
import matplotlib.pyplot as plt

def get_regular_grid(Tstart, Tend, Tstep, Xstart, Xend, Xstep, Ystart, Yend, Ystep):
    
    ''' 
        Assumption: Xstep = Ystep    
    '''    
    
    tgrid = np.linspace(Tstart, Tend, Tstep)
    xgrid = np.linspace(Xstart, Xend, Xstep)
    ygrid = np.linspace(Ystart, Yend, Ystep)
    
    t = np.tile( tgrid[np.newaxis, np.newaxis, :], ( Xstep, Ystep, 1) )   
    x = np.tile( xgrid[:, np.newaxis, np.newaxis], ( 1, Ystep, Tstep ) ) 
    y = np.tile( ygrid[np.newaxis,:,np.newaxis], ( Xstep, 1, Tstep ) ) 
    
    grid = {
    'T': {'start':Tstart, 'end':Tend, 'step':Tstep, 'incr':tgrid[1] - tgrid[0], 'matrix': t},
    'X': {'start':Xstart, 'end':Xend, 'step':Xstep, 'incr':xgrid[1] - xgrid[0], 'matrix': x},
    'Y': {'start':Ystart, 'end':Yend, 'step':Ystep, 'incr':ygrid[1] - ygrid[0], 'matrix': y}     
    }    
    return grid
    
def zero_curve(t):
    return 0.01*np.sqrt(t) - 0.001*t + 0.05
    #return np.zeros(t.shape)
    
def volatility_matrix(X, smin, smax, upper=True):
    if upper:
        return smin * (X < 0) + smax * (X >= 0)
    else:
        return smax * (X < 0) + smin * (X >= 0)
    
def UVMHJB_pde_pricer(smin, smax, beta, zero_curve, payoff, grid, upper=True):
    # Define the multiplicative
    A = np.diag(np.ones(grid['X']['step'] - 1), k=1) + np.diag(-np.ones(grid['X']['step']), k=0)
    A[-1, -1] = 1 
    A[-1, -2] = -1
    
    B = np.diag(np.ones(grid['X']['step'] - 1), k=-1) + np.diag(-np.ones(grid['X']['step']), k=0)
    B[-1, -1] = 1
    B[-2, -1] = -1
    
    C = np.diag(np.ones(grid['X']['step'] - 1), k=-1) + np.diag(-2*np.ones(grid['X']['step']), k=0) + \
        np.diag(np.ones(grid['X']['step'] - 1), k=1)
    
    '''    
    Approximation with the first order derivative:
    ----------------------------------------------
    C[0, 0] = -1
    C[0, 1] = 1
    C[-1, -1] = 1
    C[-1, -2] = -1
    '''
    
    '''
    Approximation with the next second order derivative:
    ----------------------------------------------------
    '''
    C[0,0] = 1
    C[0,1] = -2
    C[0,2] = 1
    
    C[-1,-1] = 1
    C[-1,-2] = -2
    C[-1,-3] = 1
    
    dt = grid['T']['incr']
    dx = grid['X']['incr']
    dy = grid['Y']['incr']
    
    N = grid['T']['step']
    
    F = zero_curve(grid['T']['matrix'])            
    X = grid['X']['matrix']
    Y = grid['Y']['matrix']   
    
    # Initialize Pricing Matrix
    U = np.zeros((grid['X']['step'], grid['Y']['step'], grid['T']['step']))
    U[:,:,-1] = payoff(X[:,:,-1], Y[:,:,-1], grid['T']['step'], 0.125, beta, zero_curve, 0.05)    
    
    # Solve the PDE
    for t in range(N - 2, -1, -1):
        M = dt / dx**2 * np.dot(C, U[:,:,t+1]) + 2 * dt / dy * np.dot(U[:,:,t+1], B) 
            
        V = volatility_matrix(M, smin, smax, upper)
        
        U[:,:,t] = U[:,:,t+1] - 2 * beta * dt / dy * Y[:,:,t+1] * np.dot(U[:,:,t+1], B) + \
        dt / dx * (Y[:,:,t+1] - beta * X[:,:,t+1]) * np.dot(A, U[:,:,t+1]) + \
        0.5 * V * V * M - (F[:,:,t+1] + X[:,:,t+1]) * U[:,:,t+1] * dt
    return {'grid':grid, 'zero':F, 'price':U}

def bond(X, Y, T=None, delta=None, beta=None, zero_curve=None, K=None):
    return np.ones(X.shape)
    
def forward(X, Y, T, delta, beta, zero_curve, K=None):
    time = np.linspace(T, T + delta, 100)
    dt = time[1] - time[0]
    b = np.exp( dt * np.sum( zero_curve( time[:-1] ) ) )
    k = np.exp(-beta*delta)
    g = ( 1 - k ) / beta    
    Libor = (b * np.exp( - ( 0.5 * Y * (1 - k * k) - X * g ) ) - 1 ) / delta 
    return Libor
    
def caplet(X, Y, T, delta, beta, zero_curve, K):
    Libor = forward( X, Y, T, delta, beta, zero_curve )
    P = Libor - K
    P[P < 0] = 0
    return P
    
def floorlet(X, Y, T, delta, beta, zero_curve, K):
    Libor = forward( X, Y, T, delta, beta, zero_curve )
    P = K - Libor
    P[P < 0]
    return P 

def bond_price(beta, grid, zero_curve):
    dt = grid['T']['incr']
    F = zero_curve(grid['T']['matrix'][0,0,:])
    
    B0T = np.exp(-dt * np.cumsum(F))
    B0T[1:] = B0T[:-1]
    B0T[0] = 1
    
    BtT = B0T[-1] / B0T    
    
    k = np.exp(- beta * grid['T']['matrix'][0,0,:][::-1])
    g = (1 - k) / beta
    
    Y = grid['Y']['matrix'][:,:,0]
    X = grid['X']['matrix'][:,:,0]
    
    bond = BtT[np.newaxis, np.newaxis, :] * np.exp(-(0.5*(k**2)[np.newaxis, np.newaxis,:]*Y[:,:,np.newaxis] + g[np.newaxis, np.newaxis,:]*X[:,:,np.newaxis]))   
    return bond


Tstart, Tend, Tstep = 0, 1, 101
Xstart, Xend, Xstep = -0.25, 0.25, 21
Ystart, Yend, Ystep = 0, 0.005, 21

smin, smax = 0.05, 0.2
beta = 0.05

grid = get_regular_grid(Tstart, Tend, Tstep, Xstart, Xend, Xstep, Ystart, Yend, Ystep)
    
upper_res = UVMHJB_pde_pricer(smin, smax, beta, zero_curve, bond, grid, upper=True)
lower_res = UVMHJB_pde_pricer(smin, smax, beta, zero_curve, bond, grid, upper=False)


close = bond_price(beta, grid, zero_curve)



'''
Comparing the bond price from the zero curve using close form formula or the PDE solver


i = 3
j = 3

plt.figure()
plt.plot(grid['T']['matrix'][0,0,:], upper_res['price'][i,j,:], label='PDE')
plt.plot(grid['T']['matrix'][0,0,:], close[i,j,:], label='Exact')
plt.grid()
plt.legend(loc=0)
plt.show()
'''
'''
3D PLOT
'''

pos = 0

fig = plt.figure()
ax = fig.gca(projection='3d')
ax.set_xlabel('Gamma (X)')
ax.set_ylabel('Phi (Y)')
ax.set_zlabel('Difference')
#ax.plot_surface(grid['X']['matrix'][:,:,pos], grid['Y']['matrix'][:,:,pos], close[:,:,pos] - upper_res['price'][:,:,pos], linewidth=2)
ax.plot_surface(grid['X']['matrix'][:,:,pos], grid['Y']['matrix'][:,:,pos], upper_res['price'][:,:,pos], linewidth=2)
ax.plot_surface(grid['X']['matrix'][:,:,pos], grid['Y']['matrix'][:,:,pos], lower_res['price'][:,:,pos], linewidth=2)
#ax.plot_surface(grid['X']['matrix'][:,:,pos], grid['Y']['matrix'][:,:,pos], upper_res['price'][:,:,pos] - lower_res['price'][:,:,pos], linewidth=2)
plt.show()

#plt.savefig('M:\Vathana\Notes\sanity_check2_t0.pdf', format='pdf')



