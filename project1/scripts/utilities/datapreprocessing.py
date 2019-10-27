#DATA PREPROCESSING
import numpy as np


def standardize(x):
    """Standardize the original data set."""
    x_copy = np.copy(x) 
    mean_x = np.mean(x)
    x_copy = x_copy - mean_x
    std_x = np.std(x)
    x_copy = x_copy / std_x
    return x_copy, mean_x, std_x

def adding_offset(tX):
    # adding weight w0
    """ Adding ones to tX in order for the model to include w0."""
    tX_w0 = np.c_[np.ones(tX.shape[0]), tX]
    return tX_w0

def add_interaction_terms(tX):
    # adding interaction terms
    """ The function returns a data matrix complemented with first order interaction terms of the explanatory variables.
        Ex. [ [1,2,3],    =>   [ [ 1,  2,  3,  1,  2,  3,  4,  6,  9],
              [4,5,6] ]          [ 4,  5,  6, 16, 20, 24, 25, 30, 36] ]
    """
    tX_c_T = np.copy(tX.T)
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
    """ The function returns a data matrix complemented with polynomial terms of the explanatory 
        variables up to the degree indicated by the parameter 'degree'.
        
        Ex. [ [1,2,3],    => degree = 3 =>   [ [  1   2   3   1   4   9   1   8  27],
              [4,5,6] ]                        [  4   5   6  16  25  36  64 125 216] ]
    """
    tX_c = np.copy(tX)
    for i in range(2,degree+1):
        tX_c = np.hstack((tX_c,np.power(tX,i)))
    return tX_c

def add_higher_degree_terms_customized(tX, degrees):
    """ The function returns a data matrix complemented with polynomial terms of the explanatory 
        variables of degrees indicated by the parameter 'degrees'.
        
        Ex. [ [1,2,3],    => degree = (3,4) =>   [ [   1    2    3    1    8   27    1   16   81],
              [4,5,6] ]                        [   4    5    6   64  125  216  256  625 1296] ]
              
    """
    tX_c = np.copy(tX)
    for degree in degrees:
        tX_c = np.hstack((tX_c,np.power(tX,degree)))
    return tX_c


def rescale_y(y): #rescale y to get estimates between 0 and 1
    y_rescaled = np.ones(len(y))
    y_rescaled[np.where(y==-1)] = 0
    return y_rescaled

def rescale_predictions(p): #reverse rescaling
    return 2*p-1