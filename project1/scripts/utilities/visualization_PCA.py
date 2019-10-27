# VISUALIZATION

import numpy as np
import matplotlib.pyplot as plt

def pca(tX): # Computes principal components of the set of observations.
    """
        This orthogonal transformation leads to the first principal component having the largest possible variance.
        Each succeeding component in turn has the highest variance possible under the constraint that it is orthogonal to the preceding components.
    """
    n, m = tX.shape
    # center data matrix tX
    tX_centered = np.nan_to_num((tX - np.mean(tX, 0)) / np.std(tX, 0)) ## Better solution required!!
    
    assert np.allclose(tX_centered.mean(axis=0), np.zeros(m))
    # Compute covariance matrix
    C = np.dot(tX_centered.T, tX_centered) / (n-1)
    
    # Eigen decomposition
    eigen_vals, eigen_vecs = np.linalg.eig(C)
    
    # Project X onto PC space
    tX_pca = np.dot(tX_centered, eigen_vecs)
    
    return tX_pca