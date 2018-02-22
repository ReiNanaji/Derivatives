import numpy as np
from scipy.stats import norm
from scipy.optimize import minimize

def underlyingTree(S0, r, sMax, dt, N):
    S = np.zeros((2*N+1,N+1)) 
    S[:,-1] = np.exp(sMax*np.sqrt(dt) * np.linspace(N, -N, 2*N+1)) * np.exp(r*dt*N)
    for i in range(1,N+1):
        S[i:2*N+1-i:,N-i] = S[i:2*N+1-i,N-i+1] * np.exp(-r*dt)
    return S*S0

def worst_case_proba(L, sMin, sMax):
    if L >= 0:
        return 0.5
    else:
        return 0.5*(sMin / sMax)**2
        
def best_case_proba(L, sMin, sMax):
    if L < 0:
        return 0.5
    else:
        return 0.5*(sMin / sMax)**2
    
def worst_case_value(S, payoff, r, dt, sMin, sMax, N):
    V = np.zeros((2*N+1,N+1))
    V[:,-1] = payoff(S)
    for i in range(1,N+1):
        for j in range(i,2*N+1-i):
            L = (1 - 0.5*sMax*np.sqrt(dt))*V[j-1,N+1-i] + (1 + 0.5*sMax*np.sqrt(dt))*V[j+1,N+1-i] - 2*V[j, N+1-i]
            p = worst_case_proba(L, sMin, sMax)
            V[j, N-i] = np.exp(-r*dt)*(V[j,N+1-i] + p * L)
    return V[N,0]
            
def best_case_value(S, payoff, r, dt, sMin, sMax, N):
    V = np.zeros((2*N+1,N+1))
    V[:,-1] = payoff(S)
    for i in range(1,N+1):
        for j in range(i,2*N+1-i):
            L = (1 - 0.5*sMax*np.sqrt(dt))*V[j-1,N+1-i] + (1 + 0.5*sMax*np.sqrt(dt))*V[j+1,N+1-i] - 2*V[j, N+1-i]
            p = best_case_proba(L, sMin, sMax)
            
            V[j, N-i] = np.exp(-r*dt)*(V[j,N+1-i] + p * L)
    return V[N,0]
    
# moneyness M = K / S0
    
def BS_call(S0, M, r, T, vol):
    d1 = (np.log(1/M) + (r + 0.5*vol**2)*T) / (vol*np.sqrt(T))
    d2 = d1  - vol*np.sqrt(T)
    P = S0*(norm.cdf(d1) - M*np.exp(-r*T)*norm.cdf(d2))
    return P    
    
def objective(lambda1):
    def payoff(S, M=1, M_hedge=1):
        value1 = S[:,-1] - M*np.sum(S[:,0])
        value1[value1<0] = 0
        
        value2 = S[:,-1] - M_hedge*np.sum(S[:,0])
        value2[value2<0] = 0
        return value1 - lambda1*value2
    # market price 
    sImp = 0.16
    C = BS_call(S0, M_hedge, r, 183/365., sImp)
    val = worst_case_value(S, payoff, r, dt, sMin, sMax, N)
    return (val + lambda1*C) / S0 
    
''' Payoff functions '''

def europ_call(S,M=1):
    value = S[:,-1] - M*np.sum(S[:,0])
    value[value<0] = 0
    return value
    
def one(S):
    value = np.ones(len(S[:,-1]))
    return value 
    
def forward(S):
    value = S[:,-1]
    return value
    
''' Hedging Instrument '''         
    
# Parameters (contract to value)
S0 = 100
M = 1
r = 0.07
T = 183. / 365
N = 10 #Number of timestep
dt = T / N

# Volatility Band
sMax = 0.32
sMin = 0.08

# Hedging Instrument
M_hedge = 1
sImp = 0.16
p = BS_call(S0, M_hedge, r, T, sImp)



S = underlyingTree(S0, r, sMax, dt, N)
V_worst = worst_case_value(S, europ_call, r, dt, sMin, sMax, N)
V_best = best_case_value(S, europ_call, r, dt, sMin, sMax, N)
print "Super-replication price"
print V_worst
print "Sub-replication price"
print V_best
print "Black-Scholes price"
print BS_call(S0, M_hedge, r, 183/365., sImp)

# to get the value 
res = minimize(objective, 0.5)
print "Optimal lambda"
print res.x[0]
print "Lagrangian UVM Price"
print res.fun
 
