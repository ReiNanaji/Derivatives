import numpy as np
from scipy.stats import norm
from scipy.optimize import minimize


'''
    Pricing and Hedging Derivative Securities in Markets with Uncertain Volatilities, Avellaneda-Levy-Paras
    Managing the volatility risk of portfolios of derivative securities: the lagrangian uncertain volatility model

    One-parameter family of probabilities (It is a choice!):

        pU = p.( 1 - 0.5 * sMax.sqrt(dt) )
        pM = 1 - 2p
        pD = p.( 1 + 0.5 * sMax.sqrt(dt) )

    with 0.5 * ( sMin / sMax )^2 <= p < 0.5

    The upper bound on p ensures that pM < 1. The lower bound?

    The boundary on the volatility must have been translated on the parameter p.

'''

def get_blackscholes_call_price(S0, K, r, T, vol):
    M = K / S0 # moneyness
    d1 = (np.log(1/M) + (r + 0.5*vol**2)*T) / (vol*np.sqrt(T))
    d2 = d1  - vol*np.sqrt(T)
    P = S0*(norm.cdf(d1) - M*np.exp(-r*T)*norm.cdf(d2))
    return P 


def build_underlying_Tree(S0, r, sMax, dt, N):
    '''
    Build the binomial tree representing the stock process as follow:

        S.exp(sMax.sqrt(dt) + r.dt)
      / 
    S-  S.exp(r.dt)
      \
        S.exp(-sMax.sqrt(dt) + r.dt)  

    Note: actually, only the last column matters.  

    '''
    #S = np.zeros( ( 2 * N + 1, N + 1 ) ) 
    #S[:,-1] = np.exp( sMax * np.sqrt( dt ) * np.linspace( N, - N, 2 * N +1  ) ) * np.exp( r * dt * N )
    #for i in range( 1, N + 1 ):
    #    S[i:2 * N + 1 - i:, N - i] = S[i:2 * N + 1 - i, N - i + 1] * np.exp( - r * dt )
    #return S*S0

    return S0 * np.exp( sMax * np.sqrt( dt ) * np.linspace( N, - N, 2 * N +1  ) ) * np.exp( r * dt * N )

    
def get_lower_bound(S, payoff, r, dt, sMin, sMax, N):

    '''
        Get the inf price for the parameter p in the range defined above
    '''

    V = np.zeros( ( 2 * N + 1, N + 1 ) )

    V[:,-1] = payoff( S )

    for i in range( 1, N + 1 ):
        for j in range( i, 2 * N + 1 - i ):

            L = ( 1 - 0.5 * sMax * np.sqrt( dt ) ) * V[j - 1,N + 1 - i] + (1 + 0.5 * sMax * np.sqrt( dt ) ) * V [j + 1, N + 1 - i] - 2 * V[j, N + 1 - i]
            
            if L >=0:
                p = 0.5
            else:
                p = 0.5*( sMin / sMax )**2

            V[j, N - i] = np.exp( - r * dt ) * ( V[j, N + 1 - i] + p * L )

    return V[N,0]
            
def get_upper_bound(S, payoff, r, dt, sMin, sMax, N):

    '''
        Get the sup price for the parameter p in the range defined above
    '''

    V = np.zeros( ( 2 * N + 1, N + 1 ) )

    V[:,-1] = payoff( S )

    for i in range( 1, N + 1 ):
        for j in range( i, 2 * N + 1 - i ):

            L = (1 - 0.5 * sMax * np.sqrt( dt ) ) * V[j - 1,N + 1 - i] + (1 + 0.5 * sMax * np.sqrt( dt ) ) * V[j + 1, N + 1 - i] - 2 * V[j, N + 1 - i]

            if L < 0:
                p = 0.5
            else:
                p = 0.5*( sMin / sMax )**2
            
            V[j, N-i] = np.exp( - r * dt ) * ( V[j, N + 1 - i] + p * L )

    return V[N, 0]
    
def get_payoff_call(S, K):
    value = S - K
    value[value<0] = 0
    return value
    
def get_payoff_forward(S):
    value = S
    return value


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


    
# Parameters (contract to value)
S0 = 100
K = S0
r = 0.07
T = 183. / 365
N = 10 #Number of timestep
dt = T / N

# Volatility Band
sMax = 0.32
sMin = 0.08

# Hedging Instrument
sImp = 0.16
p = get_blackscholes_call_price(S0, K, r, T, sImp)

payoff = get_payoff_call

S = build_underlying_Tree(S0, r, sMax, dt, N)
V_worst = get_lower_bound(S, payoff, r, dt, sMin, sMax, N)
V_best = get_upper_bound(S, payoff, r, dt, sMin, sMax, N)


print "Super-replication price"
print V_worst
print "Sub-replication price"
print V_best
print "Black-Scholes price"
print p

# to get the value 
#res = minimize(objective, 0.5)
#print "Optimal lambda"
#print res.x[0]
#print "Lagrangian UVM Price"
#print res.fun
 
