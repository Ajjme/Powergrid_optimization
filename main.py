
import numpy as np

from scipy.optimize import linprog
# Assume sqp is defined elsewhere and imported
from sqp import sqp  # User's SQP solver
from LP_analysis import LP_analysis  # User's LP analysis function
from opt_options import opt_options  # User's options function

# --------------------------
# Problem data (direct translation)
# --------------------------
G = np.array([60.0, 225.0])
cA = 20.0
cB = 12.0

var_pairs = [
    ('A', 1), ('A', 2), ('A', 11), ('A', 12),
    ('B', 5), ('B', 6), ('B', 7), ('B', 8),
    (1, 2), (1, 3), (1, 12), (2, 3), (3, 4),
    (4, 5), (4, 6), (5, 6), (6, 7), (7, 8),
    (7, 9), (8, 9), (9, 10), (10, 11), (10, 12), (11, 12)
]
n = len(var_pairs)   # 24
m = 14               # 2 generator + 12 demand constraints

D = np.array([30., 20., 5., 7., 35., 25., 25., 20., 5., 6., 25., 30.])

# Build A (m x n)
A = np.zeros((m, n), dtype=float)
for idx, (i, j) in enumerate(var_pairs):
    if i == 'A':
        A[0, idx] = 1.0
    if i == 'B':
        A[1, idx] = 1.0

for node in range(1, 13):
    row = 2 + (node - 1)
    for var_idx, (i, j) in enumerate(var_pairs):
        if isinstance(i, int) and isinstance(j, int):
            if j == node and i < j:
                A[row, var_idx] = -1.0
            if i == node and i < j:
                A[row, var_idx] = +1.0
        else:
            if j == node and (i == 'A' or i == 'B'):
                A[row, var_idx] = -1.0
            if i == node and isinstance(j, int) and j > i:
                A[row, var_idx] = +1.0

# Build b
b = np.zeros((m,), dtype=float)
b[0] = G[0]
b[1] = G[1]
for node in range(1, 13):
    b[2 + (node - 1)] = -D[node - 1]

# Cost vector c
c = np.zeros((n,), dtype=float)
c[0:4] = cA
c[4:8] = cB

# Line capacities and bounds
T_val = 48.0
T = T_val * np.ones((n,), dtype=float)

x_lb = np.empty(n, dtype=float)
x_ub = np.empty(n, dtype=float)
for idx, (i, j) in enumerate(var_pairs):
    if i == 'A' or i == 'B':
        x_lb[idx] = 0.0
        x_ub[idx] = T[idx]
    else:
        x_lb[idx] = -T[idx]
        x_ub[idx] = +T[idx]

# Sanity check
sumA = A.sum(axis=0)
print("sum of A columns (sanity):", sumA)

# --------------------------
# Solve with linprog (HiGHS / dual-simplex preference)
# --------------------------
print("\nlinprog ----------------------")
A_ub = A.copy()
b_ub = b.copy()
bounds = [(float(x_lb[i]), float(x_ub[i])) for i in range(n)]

try:
    res = linprog(c=c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs-ds')
    if not res.success:
        res = linprog(c=c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
except Exception:
    res = linprog(c=c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')

x_opt = res.x
f_opt = res.fun
print("linprog x_opt:", x_opt)
print("linprog f_opt:", f_opt)

g_opt = A.dot(x_opt) - b
print("linprog g_opt (A*x - b):", g_opt)

y_opt = getattr(res, 'ineqlin', res)  # solver-dependent

netGeneration = np.sum(x_opt[0:4]) + np.sum(x_opt[4:8])
netDemand = np.sum(D)
print("netGeneration:", netGeneration)
print("netDemand:", netDemand)
print("shortfall (netDemand - netGeneration):", netDemand - netGeneration)

# --------------------------
# Solve with user's sqp solver
# --------------------------
print("\ncalling user's sqp solver ----------------------")

# constants as expected by LP_analysis
constants = (A, b, c)

# initial guess: midpoint of bounds
x_init = np.array([(lb + ub) / 2.0 for lb, ub in bounds])

# options vector similar to MATLAB: [msglev tolX tolF tolG MaxEvals]
options = [0, 0.01, 0.01, 0.01, 1e4]

# Directly call the user's sqp function (assumed available in namespace)
# Preferred call: sqp(function_callable, x_init, x_lb, x_ub, options, constants)
try:
    sqp_result = sqp(LP_analysis, x_init, x_lb, x_ub, options, constants)
except TypeError:
    # fallback: maybe sqp expects function name string instead of callable
    sqp_result = sqp('LP_analysis', x_init, x_lb, x_ub, options, constants)

# Unpack results (expecting tuple (x_opt, f_opt, g_opt, cvg_hst, y_opt))
if isinstance(sqp_result, tuple) and len(sqp_result) >= 3:
    x_opt_sqp = sqp_result[0]
    f_opt_sqp = sqp_result[1]
    g_opt_sqp = sqp_result[2]
    cvg_hst = sqp_result[3] if len(sqp_result) > 3 else None
    y_opt_sqp = sqp_result[4] if len(sqp_result) > 4 else None
else:
    # try attribute access if an object was returned
    x_opt_sqp = getattr(sqp_result, 'x', None)
    f_opt_sqp = getattr(sqp_result, 'fun', None)
    g_opt_sqp = getattr(sqp_result, 'g', None)
    cvg_hst = getattr(sqp_result, 'history', None)
    y_opt_sqp = getattr(sqp_result, 'y', None)

print("sqp x_opt:", x_opt_sqp)
print("sqp f_opt:", f_opt_sqp)
print("sqp g_opt (A*x - b):", g_opt_sqp)
print("sqp convergence/history:", cvg_hst)
print("sqp multipliers/y_opt:", y_opt_sqp)

netGeneration_sqp = np.sum(x_opt_sqp[0:4]) + np.sum(x_opt_sqp[4:8])
print("sqp netGeneration:", netGeneration_sqp)
print("sqp shortfall (netDemand - netGeneration):", netDemand - netGeneration_sqp)
