import numpy as np
from scipy.special import erf, erfinv

def rnd(medX, covX, N, R=None):
    '''
    lognormal.rnd
 
    Generate N observations of correlated (or uncorrelated) lognormal random variables.
 
    Parameters:
        medX : float or array_like
               Median(s) of the lognormal distribution. If array, shape (n,) for n variables.
        covX : float or array_like
               Coefficient(s) of variation. If array, shape (n,) for n variables.
          N  : int
               Number of observations (samples) to generate.
          R  : ndarray, optional
               If None, defaults to identity matrix (uncorrelated samples).
 
    Output:
          x  : ndarray
               Shape (n, N) array of correlated lognormal random samples.
               Each row corresponds to one random variable.
               Each column corresponds to one observation.
  
    Method: (Gaussian Copula)
        1. Perform eigenvalue decomposition of correlation matrix R = V @ Λ @ V^T
        2. Generate uncorrelated standard normal samples Z ~ N(0, I)
        3. Apply correlation structure: Y = V @ sqrt(Λ) @ Z, so Y ~ N(0, R)
        4. Transform to lognormal: X = exp(log(medX) + Y * sqrt(V))
        where V = log(1 + covX²)
    
    If X is a lognormal random variable, then log(X) is normally 
    distributed with mean log(medX) and variance log(1+covX²)
    
    Examples
    --------
        # Single variable, uncorrelated samples
            x = rnd(1.0, 0.5, N=1000)
        
        # Multiple correlated variables
            medX = np.array([1.0, 2.0])
            covX = np.array([0.5, 0.3])
            R = np.array([[1.0, 0.7], [0.7, 1.0]])
            x = rnd(medX, covX, N=1000, R=R)
    '''
    
    # Convert inputs to arrays # Python needs this to handle both scalars and arrays!
    medX = np.atleast_1d(medX).astype(float)
    covX = np.atleast_1d(covX).astype(float)
    
    # Determine number of random variables
    n = len(medX)
    
    # Validate all parameters --------------------------------------
    if len(covX) != n:
        raise ValueError(f"medX and covX must have the same length. "
                        f"Got medX:{len(medX)}, covX:{len(covX)}")
    
    # Check parameter validity
    if np.any(medX <= 0):
        raise ValueError("lognormal_rnd: medX must be greater than zero")
    if np.any(covX <= 0):
        raise ValueError("lognormal_rnd: covX must be greater than zero")
    
    # Default to identity matrix (uncorrelated samples) if no correlation specified
    if R is None:
        R = np.eye(n) # In
    
    # Validate correlation matrix
    R = np.asarray(R)
    if R.shape != (n, n):
        raise ValueError(f"Correlation matrix R must be square {n}×{n}, got {R.shape}")
    
    if not np.allclose(np.diag(R), 1.0):
        raise ValueError("corr_logn_rnd: diagonal of R must equal 1")
    
    if np.any(np.abs(R) > 1):
        raise ValueError("corr_logn_rnd: R values must be between -1 and 1")
    # ---------------------------------------------------------------------
    
    # Decompose correlation matrix: R = V @ Λ @ V^T
    eVal, eVec = np.linalg.eig(R)
    
    if np.any(eVal < 0):
        raise ValueError("corr_logn_rnd: R must be positive definite")
    
    # Generate independent standard normal samples: Z ~ N(0, I)
    Z = np.random.randn(n, N)
    
    # Apply correlation structure: Y = V @ sqrt(Λ) @ Z, so Y ~ N(0, R)
    Y = eVec @ np.diag(np.sqrt(eVal)) @ Z
    
    # Compute variance of log(X) for each variable
    VlnX = np.log(1 + covX**2)
    
    # Transform to lognormal: x = exp(log(medX) + Y * sqrt(VlnX))
    # Broadcasting: medX and VlnX are (n,), need to reshape for broadcasting with (n, N)
    x = np.exp(np.log(medX[:, np.newaxis]) + Y * np.sqrt(VlnX[:, np.newaxis]))
    
    ''' 
    # current output shape is (n, N). Add this if we want to transpose output:
    if n == 1:
    return x.T  # Return (N, 1) instead of (1, N) for single variable
    return x
    '''
    return x