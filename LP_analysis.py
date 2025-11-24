import numpy as np

def LP_analysis(x, constants):
    """
    [f, g] = LP_analysis(x, constants)
    Analyze a trial solution x to a linear programming problem:
      minimize f = c^T x  such that  g = A x - b <= 0

    Parameters
    ----------
    x : array_like
        Decision vector (n,)
    constants : sequence
        constants[0] -> A : ndarray (m, n)  constraint coefficient matrix
        constants[1] -> b : ndarray (m,)    constraint vector
        constants[2] -> c : ndarray (n,)    cost coefficient vector

    Returns
    -------
    f : float
        The scalar cost c^T x
    g : ndarray
        Constraint inequalities A x - b  (m,)
    """
    # Convert inputs to numpy arrays (defensive)
    x = np.asarray(x).ravel()
    A = np.asarray(constants[0])
    b = np.asarray(constants[1]).ravel()
    c = np.asarray(constants[2]).ravel()

    # Validate shapes (helpful errors if shapes mismatch)
    if A.shape[1] != x.size:
        raise ValueError(f"Incompatible shapes: A has {A.shape[1]} cols but x has length {x.size}")
    if A.shape[0] != b.size:
        raise ValueError(f"Incompatible shapes: A has {A.shape[0]} rows but b has length {b.size}")
    if c.size != x.size:
        raise ValueError(f"Incompatible shapes: c has length {c.size} but x has length {x.size}")

    # compute cost
    f = float(np.dot(c, x))   # scalar

    # compute constraint inequalities
    g = A.dot(x) - b          # vector of length m

    return f, g
