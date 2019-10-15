#DATA PREPROCESSING
import numpy as np

def standardize(x):
    """Standardize the original data set."""
    mean_x = np.mean(x)
    x = x - mean_x
    std_x = np.std(x)
    x = x / std_x
    return x, mean_x, std_x

def build_model_data(height, weight):
    """Form (y,tX) to get regression data in matrix form 
    with x[:,0] = 1 (offset)."""
    y = weight
    x = height
    num_samples = len(y)
    tx = np.c_[np.ones(num_samples), x]
    return y, tx

def adding_offset(tX):
    # adding weight w0
    """ Adding ones to tX in order for the model to include w0."""

    tX_w0 = np.c_[np.ones(tX.shape[0]), tX]
    return tX_w0

def build_poly(x, degree):
    """polynomial basis functions for input data x, for j=0 up to j=degree."""
    # ***************************************************
    n = x.shape[0]
    tx_poly = np.zeros((n,degree))
    for i in range(n):
        for j in range(degree):
            tx_poly[i,j] = x[i]**j
    return tx_poly

def add_interaction_terms(tX):
    # adding interaction terms
    """ The function returns a data matrix complemented with first order interaction terms of the explanatory variables.
        Ex. [ [1,2,3],    =>   [ [ 1,  2,  3,  1,  2,  3,  4,  6,  9],
              [4,5,6] ]          [ 4,  5,  6, 16, 20, 24, 25, 30, 36] ]
    """

    tX_c_T = tX.T

    for col in range(tX.shape[1]):
        tX_c_T = np.vstack((tX_c_T, tX.T[col:]*tX.T[col]))
    
    return tX_c_T.T

def add_square_terms(tX):
    # adding squares of explanatory variables

    """ The function returns a data matrix complemented with first order interaction terms of the explanatory variables.
        Ex. [ [1,2,3],    =>   [ [ 1,  2,  3,  1,  2,  3,  4,  6,  9],
              [4,5,6] ]          [ 4,  5,  6, 16, 20, 24, 25, 30, 36] ]
    """
    
    return np.hstack((tX,np.power(tX,2)))

def add_higher_degree_terms(tX, degree):
    """ The function returns a data matrix complemented with exponential terms of the explanatory 
        variables up to the degree indicated by the parameter 'degree'.
        
        Ex. [ [1,2,3],    => degree = 3 =>   [ [  1   2   3   1   4   9   1   8  27],
              [4,5,6] ]                        [  4   5   6  16  25  36  64 125 216] ]
    """
    tX_c = tX
    for i in range(2,degree+1):
        tX_c = np.hstack((tX_c,np.power(tX,i)))
    return tX_c