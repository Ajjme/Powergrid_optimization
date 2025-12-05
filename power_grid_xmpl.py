#! /usr/bin/python3 -i

import numpy as np

from scipy.optimize import linprog
from collections import namedtuple

# git clone https://github.com/hpgavin/multivarious
from multivarious.opt import sqp 
from multivarious.opt import LP_analysis 
from multivarious.rvs import lognormal
from multivarious.rvs import beta
from multivarious.rvs import plot_CDF_ci 

N = 1     # for tasks 1 - 7
N = 100   # for tasks 8 and 9

# --------------------------
# Power Grid data 
# --------------------------
GA = 35.0
GB = 20.0
G = np.array([ GA , GB ])    # generation capacity, MW
cA = 20.0
cB = 12.0
c = np.array( [ cA , cA , cB , cB , 0 ] )

D = np.array( [ 30. , 20. ] )

T_cap = 35.0             # transmission line capacities and bounds

A = np.array( [ [ -1 ,  0 , -1 ,  0 ,  1 ] , 
                [  0 , -1 ,  0 , -1 , -1 ] , 
                [  1 ,  1 ,  0 ,  0 ,  0 ] ,
                [  0 ,  0 ,  1 ,  1 ,  0 ] ] )


# Sanity check
sumA = A.sum(axis=0)

# n: number of design variables ... 5 transmission lines 
# m: number of constrains ... 2 generator + 2 demand constraints
m , n = A.shape

## randomize power demands for parts 8 and 9
nd = len(D)              # number of demand nodes
medD = D                 # median demand
covD = 0.20* np.ones(nd) # coefficient of variation of demand
RD = np.eye( nd )                       # uncorrelated demands (task 8)
RD = 0.8*np.ones(nd) + 0.2*np.eye(nd);  #  correlated demands (task 9)
Drand = lognormal.rnd( medD, covD, N, RD )
Drand[:,0] = D;

## randomize line capacities for parts 8 and 9
aT = 25.0* np.ones(n)  # lower transmission capacity, MW
bT = 40.0* np.ones(n)  # upper transmission capacity, MW
qT =  4.0* np.ones(n)  # lower exponent of Beta distribution
pT =  2.0* np.ones(n)  # upper exponent of Beta distribution
RT = np.eye(n)                          # uncorrelated line capacity (task 8) 
RT = 0.7* np.ones(n) + 0.3*np.eye(n);   #  correlated line capacity (task 9)
Trand = beta.rnd ( aT , bT , qT , pT , N, RT ) # correlated beta sample

v_lb = np.zeros([m,N])
v_ub = np.zeros([m,N])

v_lb[:,0] = -T_cap * np.ones(m)
v_ub[:,0] = +T_cap * np.ones(m)
v_lb[0:4,0] = 0                  # power can not go into the power-plants

shortfall = np.nan * np.ones(N)

for ii in range(N):
    # --------------------------
    # Solve the LP with the SQP method 
    # --------------------------
    print("\n optimize using the sqp solver ----------------------")
    
    v_lb = -Trand[:,ii] 
    v_ub = +Trand[:,ii] 
    v_lb[0:4] = 0                # power can not go into the power-plants

    b = np.array( [ -Drand[0,ii] , -Drand[1,ii] , GA , GB ] )
    # various constants used within the optimization analysis ... in a named tuple
    Constant = namedtuple('Constant', [ 'A', 'b', 'c' ])
    C = Constant( A , b , c )
    
    # initial guess: midpoint of bounds
    v_init = ( v_lb + v_ub ) / 2.0 
    
    # optimization options vector  
    #        msglev   tolX   tolF   tolG   MaxEvals 
    options = [ 1   , 1e-3 , 1e-3 , 1e-3 , 100 ]
    
    # solve the LP with the SQP method 
    v_opt, f_opt, g_opt, cvg_hst, _,_ = sqp(LP_analysis, v_init, v_lb, v_ub, options, C )
    
    print("sum of each column in A should be zero (sanity check):", sumA)
    
    net_supply = np.sum ( v_opt[0:4] )    
    net_demand = np.sum ( Drand[:,ii] )
    shortfall[ii] = net_demand - net_supply
    if shortfall[ii] > 10*sum(D):    # catch numerical overflow
        shortfall[ii] = sum(D)
    print(f'\n *** shortfall # {ii:3d} = {shortfall[ii]:8.1f}        D1 = {Drand[0,ii]:6.2f} D2 = {Drand[1,ii]:6.2f} \n')
    
 
if N > 10:
    # plot the CDF of the shortfall
    plot_CDF_ci(shortfall, 90, 101, 'shortfall, MW', False )
    shortfall_probability = sum(shortfall > 0.001)/N
    print(f' *** max shortfall = {np.max(shortfall):5.2f} MW')
    print(f' *** shortfall probability = {100*shortfall_probability:5.2f} %\n')

if N == 1:
    # --------------------------
    # Solve the LP with SciPy's linprog (HiGHS / dual-simplex preference)
    # --------------------------
    print("\n optimize using the SciPi linprog solver ----------------------")
    v_lb_ub = [ ( float(v_lb[i]) , float(v_ub[i] ) ) for i in range(n) ]

    method = 'highs-ds'  #  highs-ds , highs  

    res = linprog(c=c, A_ub=A, b_ub=b, bounds=v_lb_ub, method=method )

    v_opt = res.x
    f_opt = res.fun
    print("linprog v_opt:", v_opt)
    print("linprog f_opt:", f_opt)

    g_opt = A.dot(v_opt) - b
    print("SciPy linprog g_opt (A*x - b):", g_opt)

    y_opt = getattr(res, 'ineqlin', res)  # solver-dependent

    netSupply = np.sum(v_opt[0:4]) 
    netDemand = np.sum(D)
    print("netSupply:", netSupply)
    print("netDemand:", netDemand)
    print("shortfall (netDemand - netSupply):", netDemand - netSupply)

