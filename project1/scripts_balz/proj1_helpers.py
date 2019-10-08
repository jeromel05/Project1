# -*- coding: utf-8 -*-
"""some helper functions for project 1."""
import csv
import numpy as np


def load_csv_data(data_path, sub_sample=False):
    """Loads data and returns y (class labels), tX (features) and ids (event ids)"""
    y = np.genfromtxt(data_path, delimiter=",", skip_header=1, dtype=str, usecols=1)
    x = np.genfromtxt(data_path, delimiter=",", skip_header=1)
    ids = x[:, 0].astype(np.int)
    input_data = x[:, 2:]

    # convert class labels from strings to binary (-1,1)
    yb = np.ones(len(y))
    yb[np.where(y=='b')] = -1
    
    # sub-sample
    if sub_sample:
        yb = yb[::50]
        input_data = input_data[::50]
        ids = ids[::50]

    return yb, input_data, ids


def predict_labels(weights, data):
    """Generates class predictions given weights, and a test data matrix"""
    y_pred = np.dot(data, weights)
    y_pred[np.where(y_pred <= 0)] = -1
    y_pred[np.where(y_pred > 0)] = 1
    
    return y_pred


def create_csv_submission(ids, y_pred, name):
    """
    Creates an output file in csv format for submission to kaggle
    Arguments: ids (event ids associated with each prediction)
               y_pred (predicted class labels)
               name (string name of .csv output file to be created)
    """
    with open(name, 'w') as csvfile:
        fieldnames = ['Id', 'Prediction']
        writer = csv.DictWriter(csvfile, delimiter=",", fieldnames=fieldnames)
        writer.writeheader()
        for r1, r2 in zip(ids, y_pred):
            writer.writerow({'Id':int(r1),'Prediction':int(r2)})

 ### Helper functions

## Visualization

# PCA

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

## Models

# Logistic regression

def compute_p(w,tX): 
    """
        Computes probabilities of all observations in tX corresponding to -1 = 'b' or 1 = 's' based on weights w according to the logistc transformation.
        (p = 0 correponds to -1 resp. 'b' and p = 1 to 1 resp. 's' to simplify the computations)
    """
    #odds = np.exp(np.dot(tX,w))
    odds = np.nan_to_num(np.exp(np.dot(tX,w)))
    
    return odds/(1+odds)

def compute_log_likelihood(w,tX, y): # Be careful! y must be binary. (0 or 1) !!
    "Computes the log-likelihood of observing the data tX with the given weights w."
    
    p = compute_p(w,tX)
    return sum(y*np.nan_to_num(np.log(p))+(1-y)*np.nan_to_num(np.log(1-p)))

def compute_gradient_log_likelihood(w,tX, y): # Be careful! y must be binary. (0 or 1) !!
    "Computes the gradient of the log-likelihood the given current weights w and the data tX with respect to w."
    
    p = compute_p(w,tX) 
    odds = np.exp(np.dot(tX,w)) 
    return np.dot(tX.T,(y-p))

def compute_jacobean_log_likelihood(w,tX, y): # Be careful! y must be binary. (0 or 1) !!
    "Computes the jacobean of the log-likelihood the given current weights w and the data tX with respect to w."
    
    jacobean = np.zeros((len(w), len(w)))
    p = compute_p(w,tX) 
    for i in range(len(w)):
        for j in range(len(w)):
            jacobean[i,j] = sum(tX[:,i]*tX[:,j]*(p*(1-p)))
    
    return -jacobean

    #return -np.dot(tx.T,np.dot(np.diag(p*(1-p)),tx)) # A way more elegant solution. Unfortunately, the diagonal matrix is too large.
    
def find_optimal_w(y, tX, w_initial, max_iters):
    """Finds the most likely weights using the iterative Newton-Raphson method."""
    
    # Define parameters to store w and loss
    ws = [w_initial]
    log_likelihoods = [compute_log_likelihood(w_initial,tX, y)]
    w = w_initial
    for n_iter in range(max_iters):
        
        with np.errstate(divide='raise',invalid='raise'):
            try:
                gradient = compute_gradient_log_likelihood(w, tX, y)
                jacobean = compute_jacobean_log_likelihood(w,tX, y)
            except FloatingPointError as e:
                print(e)
                print(tX, n_iter)
                return log_likelihoods, ws
        
        
        try:
            w = w - np.linalg.solve(jacobean,gradient)
        except np.linalg.LinAlgError as e:
            print(e)
            #print(log_likelihood,gradient,jacobean)
            print(tX, n_iter)
            
            return log_likelihoods, ws
            
        
        # store w and loss
        ws.append(w)
        log_likelihood = compute_log_likelihood(w,tX, y)
        log_likelihoods.append(log_likelihood)


    return log_likelihoods, ws

def rescale_y(y): #rescale y to get estimates between -1 and 1
    y_rescaled = np.ones(len(y))
    y_rescaled[np.where(y==-1)] = 0
    return y_rescaled

def rescale_predictions(p): #reverse rescaling
    return 2*p-1

## complementing data matrix tX

# adding weight w0

def adding_offset(tX):
    """ Adding ones to tX in order for the model to include w0."""

    tX_w0 = np.c_[np.ones(tX.shape[0]), tX]
    return tX_w0

# adding interaction terms

def add_interaction_terms(tX):
    """ The function returns a data matrix complemented with first order interaction terms of the explanatory variables.
        Ex. [ [1,2,3],    =>   [ [ 1,  2,  3,  1,  2,  3,  4,  6,  9],
              [4,5,6] ]          [ 4,  5,  6, 16, 20, 24, 25, 30, 36] ]
    """

    tX_c_T = tX.T

    for col in range(tX.shape[1]):
        tX_c_T = np.vstack((tX_c_T, tX.T[col:]*tX.T[col]))
    
    return tX_c_T.T

# adding squares of explanatory variables

def add_square_terms(tX):
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

## handling missing values in data

# replace -999 by average value

def set_missing_explanatory_vars_to_mean(tX): # Missing values appear as -999 in the data. The function sets them to the mean of the successful measurements of the variable.
    tX_corr = tX
    for i in range(tX.shape[1]):
        mean_xi = np.mean(tX[:,i][np.not_equal(tX[:,i],-999*np.ones(len(tX[:,i])))])
        tX_corr[:,i][np.where(tX_corr[:,i] == -999)] = mean_xi
    return tX_corr

# Taking into account their pattern for the handling of missing values

def split_data_according_to_pattern_of_missing_values(tX):
    """
        This function separates the data tX into a list of of arrays containing only instances of the explanatory
        variable with the same values missing. It returns:
            - 'tX_split', a list of arrays of arrays with the same pattern of missing values
            - 'ind_row_groups', a list of 1D arrays containing the subgroup's rows indices (ids) they had
                                in the orginal data matrix tX.
            - 'groups_mv_num', an array containing a numerical, binary representation of each subgroup's pattern of missing
                               values in the columns of tX that lack any value. Can be used to verify if tX and tX_test
                               are divided into the same groups and in the same order.
            (- 'bool_mask_col_mv_groups', an array containing the same information as 'groups_mv_num' but in the form of a bolean matrix.)
    """
    
    # Extracting ensemble of columns that contain missing values.
    ind_col = np.arange(tX.shape[1])
    ind_col_mv = ind_col[np.where(sum(tX == -999) > 0)]

    tX_cols_mv = tX[:,ind_col_mv]
    
    # Simplifying by taking '0' and '1' to represent present and absent values, respectively. The order of samples is preserved.
    
    pattern_cols_mv = np.zeros(tX_cols_mv.shape)
    pattern_cols_mv[np.where(tX_cols_mv == -999)] = 1
    
    pattern_cols_mv_num = np.dot(pattern_cols_mv,np.flip(np.power(10,np.arange(pattern_cols_mv.shape[1]),dtype=np.int64))) # Numerical (binary) representation of absence pattern in conserned columns.
    
    groups_mv_num = np.unique(pattern_cols_mv_num) # All observed patterns of missing values in the columns with gaps.
    
    ind_row = np.arange(tX.shape[0]) # Indices of tX's row
    
    tX_split = []
    ind_row_groups = []
    #bool_mask_col_mv_groups = np.array([tX[0,:] == -999])
    
    
    for group_mv_num in groups_mv_num:
        # calculating and stocking the indices of the rows of tX that belong to the group

        ind_row_group = ind_row[pattern_cols_mv_num == group_mv_num]
        ind_row_groups.append(ind_row_group)
        
        
        # extracting subset of tX faling into the group. Removing missing columns and stocking their positions.
        tX_group_rows = tX[ind_row_group,:]
        bool_mask_col_mv_group = [tX_group_rows[0,:] > -999]
        #bool_mask_col_mv_groups= np.vstack((bool_mask_col_MV_groups,bool_mask_col_MV_group))
        
        ind_col = np.arange(tX_group_rows.shape[1])
        
        tX_split.append(tX_group_rows[:,ind_col[bool_mask_col_mv_group]])
    
    #bool_mask_col_mv_groups = np.delete(bool_mask_col_MV_groups,[0],axis = 0)

    
    return tX_split, ind_row_groups, groups_mv_num 
    #bool_mask_col_mv_groups, (np.logical_not(bool_mask_col_mv_groups.flatten())).reshape(bool_mask_col_mv_groups.shape)

def split_y_according_to_pattern_of_missing_values(y, ind_row_groups):
    """
        This function splits y into a list of 1D arrays, y_split, according to the pattern found in the data matrix tX.
        The parameter 'ind_row_groups' gets calculated when splitting tX by means of 'split_data_accoring_to pattern_of_missing_values'.
    """
    y_split = []
    
    for ind_row_group in ind_row_groups:
        y_split.append(y[ind_row_group])
    
    return y_split

## Account for pattern of missing values combined with logistic regression

def standardize(x):

    centered_data = x - np.mean(x, axis=0)
    std_data = np.nan_to_num(centered_data / np.std(centered_data, axis=0))
    
    return std_data

def find_optimal_weights_pattern_mv(tX_split, ind_row_groups,y_split,len_y):
    ws_groups = []
    y_pred = np.zeros(len_y)
    
    for tX_group, y_group, ind_row_group in zip(tX_split,y_split,ind_row_groups):
        
        #Test
        print(ind_row_group)
        ind_col_non_zero = np.arange(len(tX_group[0,:]))[sum(tX_group**2)>0]
        #log_likelihoods, ws = find_optimal_w(rescale_y(y_group), standardize(tX_group[:,ind_col_non_zero]), np.random.rand(tX_group[:,ind_col_non_zero].shape[1])/1000000,10)
       #log_likelihoods, ws = find_optimal_w(rescale_y(y_group), tX_group[:,ind_col_non_zero], np.random.rand(tX_group[:,ind_col_non_zero].shape[1])/(100000*(np.std(tX_group[:,ind_col_non_zero], axis=0)+1)*np.mean(tX_group[:,ind_col_non_zero], axis=0)),15)
        log_likelihoods, ws = find_optimal_w(rescale_y(y_group), tX_group[:,ind_col_non_zero], np.random.rand(tX_group[:,ind_col_non_zero].shape[1])/(10000*tX_group[:,ind_col_non_zero].shape[1]*(np.std(tX_group[:,ind_col_non_zero], axis=0)+1)*np.mean(tX_group[:,ind_col_non_zero], axis=0)),10)
    
        w = np.zeros(len(tX_group[0,:]))
        w[ind_col_non_zero] = ws[np.argmax(log_likelihoods)]
        ws_groups.append(w)
        print(log_likelihoods,ws)
        y_pred[ind_row_group] = rescale_predictions(compute_p(w,tX_group))
    
    return ws_groups, y_pred

def predict_y_lr_weights_pattern_mv(ws_groups, tX_split_test, ind_row_groups_test, len_y_test):
    
    y_pred_test = np.zeros(len_y_test)

    for tX_group_test, w_group, ind_row_group_test in zip(tX_split_test, ws_groups, ind_row_groups_test) :

        y_pred_test[ind_row_group_test] = rescale_predictions(compute_p(w_group,tX_group_test))
    
    return y_pred_test

"""
def predict_y_lr_weights_pattern_mv(ws_groups, tX_split_test, ind_row_groups_test, len_y_test):
    
    y_pred_test = np.zeros(len_y_test)

    for tX_group_test, w_group, ind_row_group_test in zip(tX_split_test, ws_groups, ind_row_groups_test) :

        y_pred_test[ind_row_group_test] = rescale_predictions(compute_p(w_group,tX_group_test))
    
    return y_pred_test
"""

## Account for pattern of missing values and interaction terms

def add_exponential_terms_to_split_data(tX_split, degree = 2):
    tX_split_interaction_terms = []

    for tX_group in tX_split:
        tX_split_interaction_terms.append(add_higher_degree_terms(tX_group, degree))
    
    return tX_split_interaction_terms

def add_interaction_terms_to_split_data(tX_split):
    tX_split_interaction_terms = []

    for tX_group in tX_split:
        tX_split_interaction_terms.append(add_interaction_terms(tX_group))
    
    return tX_split_interaction_terms