import numpy as np
import matplotlib.pyplot as plt
from datapreprocessing import *
from functions_for_log_regression import *
from math import sqrt

# SPLIT DATA TRAIN + TEST

def split_data(x, y, ratio, seed=1):
    # split the data based on the given ratio
    """
    split the dataset based on the split ratio. If ratio is 0.8 
    you will have 80% of your data set dedicated to training 
    and the rest dedicated to testing
    """
    # set seed
    np.random.seed(seed)
    itrain=np.random.choice(y.shape[0],int(ratio*y.shape[0]),replace=False)
    # np.arange(size) donne les numéros de 0 à size
    itest = np.delete(np.arange(y.shape[0]),itrain)
    xtrain=x[itrain]
    ytrain=y[itrain]
    xtest=x[itest]
    ytest=y[itest]
    return xtrain,ytrain,xtest,ytest

def build_k_indices(y, k_fold, seed):
    """build k indices for k-fold."""
    num_row = y.shape[0]
    interval = int(num_row / k_fold)
    np.random.seed(seed)
    indices = np.random.permutation(num_row)
    k_indices = [indices[k * interval: (k + 1) * interval]
                 for k in range(k_fold)]
    return np.array(k_indices)

def cross_validation_logistic_regression_feature_engineering_ridge(tX, y, degree, k_indices, k, max_iters, threshold, lambdas, gamma = 1):
    """
    Auxiliary function for k-means cross validation.
    
    The function splits the data into a test and a training, adds polynomial terms of the explanatory variables up to 
    a given degree, and finds the optimal weights by performing a penalized logistic regression on the training set
    given a set of lambdas.
    
    It returns the RMSE as well as the absolute error (0 or 1) of the predictions for the test and the training sets.
    """
    
    # ***************************************************
    # split the data, and return train and test data
    # ***************************************************
    tX_test_split = tX[k_indices[k],:]
    tX_train_split = np.delete(tX,k_indices[k],axis=0)
    y_test_split = y[k_indices[k]]
    y_train_split = np.delete(y,k_indices[k])
    
    # ***************************************************
    # form train and test data with polynomial basis function:
    # ***************************************************
    tX_train_split_extended = add_higher_degree_terms(tX_train_split, degree)
    tX_test_split_extended = add_higher_degree_terms(tX_test_split, degree)
    
    # ***************************************************
    # calcualte most likely weights through logistic regression with ridge term
    # ***************************************************
    ind_col_non_const = np.arange(len(tX_train_split_extended[0,:]))[np.std(tX_train_split_extended,0)>0]
    
    tX_train_split_extended = adding_offset(tX_train_split_extended)
    tX_test_split_extended = adding_offset(tX_test_split_extended)
    ind_col_non_const += 1
    ind_col_non_const = np.insert(ind_col_non_const,0, 0)
    
    rmse_tr = np.zeros(len(lambdas))
    rmse_te = np.zeros(len(lambdas))
    
    abse_tr = np.zeros(len(lambdas))
    abse_te = np.zeros(len(lambdas))
    
    for ind_lambda, lambda_ in enumerate(lambdas):
        
        log_likelihoods, ws = penalized_logistic_regression(y_train_split, tX_train_split_extended[:,ind_col_non_const], max_iters, threshold,lambda_,gamma)
    
        ind_min = np.argmin(log_likelihoods)
        w_star = np.zeros(len(tX_train_split_extended[0,:]))
        w_star[ind_col_non_const] = ws[ind_min]

        # ***************************************************
        # calculate RMSE and ABSE for train and test data,and store them in rmse_tr, abse_tr, rmse_te and abse_te respectively
        # ***************************************************
        
        log_likelihoods_train = calculate_loss_logistic_regression(y_train_split, tX_train_split_extended, w_star)
        log_likelihoods_test = calculate_loss_logistic_regression(y_test_split, tX_test_split_extended, w_star)
    
        rmse_tr[ind_lambda] = np.linalg.norm((y_train_split - compute_p(w_star,tX_train_split_extended))) / sqrt(y_train_split.shape[0])
        rmse_te[ind_lambda] = np.linalg.norm((y_test_split - compute_p(w_star,tX_test_split_extended)))/ sqrt(y_test_split.shape[0])
        
        abse_tr[ind_lambda] = np.sum(abs(y_train_split - [compute_p(w_star,tX_train_split_extended) > 0.5]))
        abse_te[ind_lambda] = np.sum(abs(y_test_split - [compute_p(w_star,tX_test_split_extended) > 0.5]))
        
        print("degree={d}, k={k_}, lambda={l:10.3e}, Training RMSE={tr:.3f}, Testing RMSE={te:.3f}, Training loss={trl:.3f}, Testing loss={tel:.3f}, Training # Missclassification ={m_tr:.3f}, Testing # Missclassification={m_te:.3f}".format(
            d=degree, k_=k, l= lambda_, tr=rmse_tr[ind_lambda], te=rmse_te[ind_lambda], trl = log_likelihoods_train, tel = log_likelihoods_test, m_tr = abse_tr[ind_lambda], m_te = abse_te[ind_lambda]))
        
    return rmse_tr, rmse_te, abse_tr, abse_te

def train_test_split_logistic_regression_feature_engineering_ridge(tX, y, degree, ratio, seed, max_iters, threshold, lambdas, gamma):
    """
    The function splits the data into a test and a training, adds polynomial terms of the explanatory variables up to 
    a given degree, and finds the optimal weights by performing a penalized logistic regression on the training set
    given a set of lambdas.
    
    It returns the RMSE as well as the absolute error (0 or 1) of the predictions for the test and the training sets.
    """
    
    # ***************************************************
    # split the data, and return train and test data
    # ***************************************************
    tX_train_split,y_train_split, tX_test_split, y_test_split = split_data(tX, y, ratio, seed)
  
    # ***************************************************
    # form train and test data with polynomial basis function:
    # ***************************************************
    tX_train_split_extended = add_higher_degree_terms(tX_train_split, degree)
    tX_test_split_extended = add_higher_degree_terms(tX_test_split, degree)
    
    # ***************************************************
    # calcualte most likely weights through logistic regression with ridge term
    # ***************************************************
    ind_col_non_const = np.arange(len(tX_train_split_extended[0,:]))[np.std(tX_train_split_extended,0)>0]
    
    tX_train_split_extended = adding_offset(tX_train_split_extended)
    tX_test_split_extended = adding_offset(tX_test_split_extended)
    ind_col_non_const += 1
    ind_col_non_const = np.insert(ind_col_non_const,0, 0)
    
    rmse_tr = np.zeros(len(lambdas))
    rmse_te = np.zeros(len(lambdas))
    
    abse_tr = np.zeros(len(lambdas))
    abse_te = np.zeros(len(lambdas))
    
    for ind_lambda, lambda_ in enumerate(lambdas):
        
        log_likelihoods, ws = penalized_logistic_regression(y_train_split, tX_train_split_extended[:,ind_col_non_const], max_iters, threshold,lambda_,gamma)
    
        ind_min = np.argmin(log_likelihoods)
        w_star = np.zeros(len(tX_train_split_extended[0,:]))
        w_star[ind_col_non_const] = ws[ind_min]

        # ***************************************************
        # calculate RMSE and ABSE for train and test data, and store them in rmse_tr, abse_tr, rmse_te and abse_te respectively
        # ***************************************************
        
        log_likelihoods_train = calculate_loss_logistic_regression(y_train_split, tX_train_split_extended, w_star)
        log_likelihoods_test = calculate_loss_logistic_regression(y_test_split, tX_test_split_extended, w_star)
    
        rmse_tr[ind_lambda] = np.linalg.norm((y_train_split - compute_p(w_star,tX_train_split_extended))) /sqrt(y_train_split.shape[0])
        rmse_te[ind_lambda] = np.linalg.norm((y_test_split - compute_p(w_star,tX_test_split_extended))) /sqrt(y_test_split.shape[0])
        
        abse_tr[ind_lambda] = np.sum(abs(y_train_split - [compute_p(w_star,tX_train_split_extended) > 0.5]))
        abse_te[ind_lambda] = np.sum(abs(y_test_split - [compute_p(w_star,tX_test_split_extended) > 0.5]))
        
        print("ratio={r}, degree={d}, seed={s}, lambda={l:.3f}, Training RMSE={tr:.3f}, Testing RMSE={te:.3f}, Training loss={trl:.3f}, Testing loss={tel:.3f}, Training # Missclassification ={m_tr:.3f}, Testing # Missclassification={m_te:.3f}".format(
            r = ratio, d=degree, s=seed, l= lambda_, tr=rmse_tr[ind_lambda], te=rmse_te[ind_lambda], trl = log_likelihoods_train, tel = log_likelihoods_test, m_tr = abse_tr[ind_lambda], m_te = abse_te[ind_lambda]))
    
    return rmse_tr, rmse_te, abse_tr, abse_te
    
def train_test_split_logistic_regression_feature_engineering_ridge_demo(tX, y, degrees, split_ratios, seeds, max_iters, threshold, lambdas, gamma):
    rmse_tr = np.zeros([len(split_ratios),len(seeds),len(degrees),len(lambdas)])
    rmse_te = np.zeros([len(split_ratios),len(seeds),len(degrees),len(lambdas)])
    abse_tr = np.zeros([len(split_ratios),len(seeds),len(degrees),len(lambdas)])
    abse_te = np.zeros([len(split_ratios),len(seeds),len(degrees),len(lambdas)])
    
    for ind_split_ratio,split_ratio in enumerate(split_ratios):
        for ind_seed, seed in enumerate(seeds):
            for ind_degree, degree in enumerate(degrees):
                rmse_tr[ind_split_ratio,ind_seed,ind_degree,:],rmse_te[ind_split_ratio,ind_seed,ind_degree,:], \
                abse_tr[ind_split_ratio,ind_seed,ind_degree,:],abse_te[ind_split_ratio,ind_seed,ind_degree,:] = \
                train_test_split_logistic_regression_feature_engineering_ridge(tX, y,        \
                                                degree, split_ratio, seed, max_iters, threshold,lambdas,gamma)
    return     rmse_tr, rmse_te, abse_tr, abse_te

def train_test_split_logistic_regression_feature_engineering_ridge_groups_demo(tX, y, degrees, split_ratios, seeds, max_iters, threshold, lambdas, gamma):
    
    tX_split, ind_row_groups, groups_mv_num = split_data_according_to_pattern_of_missing_values(tX)
    y_split = split_y_according_to_pattern_of_missing_values(y, ind_row_groups)
    
    rmse_tr = np.zeros([len(ind_row_groups),len(split_ratios),len(seeds),len(degrees),len(lambdas)])
    rmse_te = np.zeros([len(ind_row_groups),len(split_ratios),len(seeds),len(degrees),len(lambdas)])
    abse_tr = np.zeros([len(ind_row_groups),len(split_ratios),len(seeds),len(degrees),len(lambdas)])
    abse_te = np.zeros([len(ind_row_groups),len(split_ratios),len(seeds),len(degrees),len(lambdas)])
    
    for ind_group, ind_row_group in enumerate(ind_row_groups):
        print('group (' + str(ind_group + 1) + '/' + str(len(groups_mv_num)) + ')')
        
        rmse_tr[ind_group], rmse_te[ind_group], abse_tr[ind_group], abse_te[ind_group] = \
            train_test_split_logistic_regression_feature_engineering_ridge_demo(tX[ind_row_group], rescale_y(y[ind_row_group]), degrees, split_ratios, seeds, max_iters, threshold, lambdas, gamma)

    return rmse_tr, rmse_te, abse_tr, abse_te
