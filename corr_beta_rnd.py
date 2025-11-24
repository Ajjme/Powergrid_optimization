import numpy as np
from scipy.special import erf, erfinv


def rnd(a, b, q, p, N, R=None):
    '''
    beta.rnd
    Generate N observations of correlated (or uncorrelated) beta random variables.

    INPUT:
        a : float or array_like
            Lower bound(s) of the distribution. If array, shape (n,) for n variables.
        b : float or array_like
            Upper bound(s) of the distribution. If array, shape (n,) for n variables.
        q : float or array_like
            First shape parameter(s). If array, shape (n,) for n variables.
        p : float or array_like
            Second shape parameter(s). If array, shape (n,) for n variables.
        N : int
            Number of observations (samples) to generate.
        R : ndarray, optional
            n×n correlation matrix of standard normal deviates.
            If None, defaults to identity matrix (uncorrelated samples).

    OUTPUT:
        x : ndarray
            Shape (n, N) array of correlated beta random samples.
            Each row corresponds to one random variable.
            Each column corresponds to one observation.

    Method:
        1. Perform eigenvalue decomposition of correlation matrix R
        2. Generate uncorrelated standard normal samples
        3. Apply correlation structure: Y = eVec @ sqrt(eVal) @ Z
        4. Transform to uniform via standard normal CDF
        5. Transform to beta via inverse CDF for each variable

    Example (Usage):
        # Single variable, uncorrelated samples
            x = rnd(0, 1, 2, 5, N=1000)
        
        # Multiple correlated variables
            a = np.array([0, 1])
            b = np.array([1, 3])
            q = np.array([2, 3])
            p = np.array([5, 4])
            R = np.array([[1.0, 0.7], [0.7, 1.0]])
            x = rnd(a, b, q, p, N=1000, R=R)
    '''
    
    # Convert inputs to arrays
    a = np.atleast_1d(a).astype(float) # Note: we must convert inputs to arrays. 
    b = np.atleast_1d(b).astype(float) # MATLAB implicitly handles scalars vs arrays. Python does not.
    q = np.atleast_1d(q).astype(float)
    p = np.atleast_1d(p).astype(float)
    
    # Determine number of random variables
    n = len(a)
    

    # -------------------------------------- Input Validations:
    # Validate that all parameter arrays have the same length
    if not (len(b) == n and len(q) == n and len(p) == n):
        raise ValueError(f"All parameter arrays must have the same length. "
                        f"Got a:{len(a)}, b:{len(b)}, q:{len(q)}, p:{len(p)}")
    
    if np.any(b <= a):
        raise ValueError("beta_rnd: all b values must be greater than corresponding a values")
    if np.any(q <= 0):
        raise ValueError("beta_rnd: q must be positive")
    if np.any(p <= 0):
        raise ValueError("beta_rnd: p must be positive")
    
    # If no correlation matrix provided, default to identity matrix
    # Identity matrix R = I means all variables are independent (correlation = 0)
    if R is None:
        R = np.eye(n) # In
    
    # Convert R to array and validate its properties
    R = np.asarray(R)
    if R.shape != (n, n):
        raise ValueError(f"Correlation matrix R must b square {n}×{n}, got {R.shape}")
    
    if not np.allclose(np.diag(R), 1.0): # diagonals must be 1s
        raise ValueError("corr_beta_rnd: diagonal of R must equal 1")
    
    if np.any(np.abs(R) > 1): # all elements must be [-1, 1] i.e valid correlations
        raise ValueError("corr_beta_rnd: R values must be between -1 and 1")
    # # -------------------------------------- End Input Validations
    
    # Eigenvalue decomposition of correlation matrix: R = V @ Λ @ V^T
    #   eVec (V): matrix of eigenvectors (n×n)
    #   eVal (Λ): array of eigenvalues (length n)
    eVal, eVec = np.linalg.eig(R)
    
    if np.any(eVal < 0):
        raise ValueError("corr_beta_rnd: R must be positive definite")
    
    # Generate independent standard normal samples: Z ~ N(0, I)
    Z = np.random.randn(n, N) 
    
    # Apply correlation structure
    # Y = V @ sqrt(Λ) @ Z, so Y ~ N(0, R)
    #   = eVec @ sqrt(eVal) @ Z
    Y = eVec @ np.diag(np.sqrt(eVal)) @ Z
    
    # Transform to uniform [0,1] via standard normal CDF, preserving correlation
    U = norm.cdf(Y)
    
    # Transform each variable to its beta distribution via inverse CDF
    x = np.zeros((n, N))
    for i in range(n):
        x[i, :] = inv(U[i, :], a[i], b[i], q[i], p[i])
    
    return x