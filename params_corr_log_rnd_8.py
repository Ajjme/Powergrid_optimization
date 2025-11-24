import numpy as np
from corr_beta_rnd import rnd

D = np.array([30, 20, 5, 7, 35, 25, 25, 20, 5, 6, 25, 30], dtype=float)
N = 100  # or whatever sample size you want Increase if want better results

medD = D
nd = len(D)
covD = 0.10 * np.ones(nd)
RD = np.eye(nd)

# 9
# RD = 0.8*np.ones( nd ) + 0.2*np.eye( nd ); # c o r r e l a t e d v a r i a b i l i t y be tween node demands


Drand = rnd(medD, covD, N, R=RD)
