import numpy as np
from corr_beta_rnd import rnd

n = 1 # number of beta variables


aT = 30.0 * np.ones(n)
bT = 65.0 * np.ones(n)
qT = 4.0 * np.ones(n)
pT = 2.0 * np.ones(n)

RT = np.eye(n)
# 9
# RT = 0.7*np.ones(n) + 0.3*np. eye(n ); % c o r r e l a t i o n ma tr ix f o r t a s k 9 : pos . c o r r e l 

N = 100  # or whatever sample size you want

Trand = rnd(aT, bT, qT, pT, N, R=RT)