# 13 line version fro PDF
import numpy as np
def LP_analysis(x, constants):
    # [f, g] = LP_analysis(x, constants)
    # analyze a trial solution x to a linear programming problem,
    # minimize f = c' * x such that g = A * x - b <= 0

    A = constants[0]   # constraint coefficient matrix (dimension m by n)
    b = constants[1]   # constraint vector (dimension m by 1)
    c = constants[2]   # cost coefficient vector (dimension n by 1)

    f = np.dot(c.T, x)  # the cost function

    g = A @ x - b       # the constraint inequalities, compared to zero

    return f, g