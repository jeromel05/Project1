#!/usr/bin/env python
# coding: utf-8
"""some utility functions for project 1."""
import numpy as np
import matplotlib.pyplot as plt

from datapreprocessing import *

#LOSS
def compute_loss_linear(y, tx, w, method="mse"):
    """Calculate the loss.
    You can calculate the loss using mse or mae.
    """
    error = y - np.dot(tx,w)
    if method == "mae":
        return np.sum(np.abs(error)) / np.shape(y)[0] / 2
    elif method == "mse":
        return np.inner(error,error) / np.shape(y)[0] / 2 #for MSE
    else:
        raise Exception("Specified method unknown")
        
def rmse (lmse) :
    """ Calculate rmse from lmse"""
    return np.sqrt(2*lmse)

# SPLIT DATA TRAIN + TEST
# a utiliser pour les patterns ou on a des mauvaises approximations à chaque fois
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
    
        rmse_tr[ind_lambda] = np.linalg.norm((y_train_split - compute_p(w_star,tX_train_split_extended))/y_train_split.shape[0])
        rmse_te[ind_lambda] = np.linalg.norm((y_test_split - compute_p(w_star,tX_test_split_extended))/y_test_split.shape[0])
        
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
    
        rmse_tr[ind_lambda] = np.linalg.norm((y_train_split - compute_p(w_star,tX_train_split_extended))/y_train_split.shape[0])
        rmse_te[ind_lambda] = np.linalg.norm((y_test_split - compute_p(w_star,tX_test_split_extended))/y_test_split.shape[0])
        
        abse_tr[ind_lambda] = np.sum(abs(y_train_split - [compute_p(w_star,tX_train_split_extended) > 0.5]))
        abse_te[ind_lambda] = np.sum(abs(y_test_split - [compute_p(w_star,tX_test_split_extended) > 0.5]))
        
        #print("ratio={r}, degree={d}, seed={s}, lambda={l:.3f}, Training RMSE={tr:.3f}, Testing RMSE={te:.3f}, Training loss={trl:.3f}, Testing loss={tel:.3f}, Training # Missclassification ={m_tr:.3f}, Testing # Missclassification={m_te:.3f}".format(
           # r = ratio, d=degree, s=seed, l= lambda_, tr=rmse_tr[ind_lambda], te=rmse_te[ind_lambda], trl = log_likelihoods_train, tel = log_likelihoods_test, m_tr = abse_tr[ind_lambda], m_te = abse_te[ind_lambda]))
    
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
    
#FOR GRADIENT DESCENT   
def compute_gradient(y, tx, w):
    """Compute the gradient."""
    n = np.shape(tx)[0]
    error = y - np.dot(tx,w)
    return - np.dot(tx.T,error)/n          

def compute_subgradient(y, tx, w):
    """Compute the subgradient for MAE."""
    n = np.shape(tx)[0]
    error = (y - np.dot(tx,w))
    for e in error: #faire le vector set_h element wise!!!!
        if(e<0):
            e = -1
        if(e>0):
            e = 1 
        else:
          #print("non-diff point")
            e = uniform(-1.0,1.0)         
    return - np.dot(error, tx) / n


def batch_iter(y, tx, batch_size, num_batches=1, shuffle=True):
    """
    Generate a minibatch iterator for a dataset.
    Takes as input two iterables (here the output desired values 'y' and the input data 'tx')
    Outputs an iterator which gives mini-batches of `batch_size` matching elements from `y` and `tx`.
    Data can be randomly shuffled to avoid ordering in the original data messing with the randomness of the minibatches.
    Example of use :
    for minibatch_y, minibatch_tx in batch_iter(y, tx, 32):
        <DO-SOMETHING>
    """
    data_size = len(y)

    if shuffle:
        shuffle_indices = np.random.permutation(np.arange(data_size))
        shuffled_y = y[shuffle_indices]
        shuffled_tx = tx[shuffle_indices]
    else:
        shuffled_y = y
        shuffled_tx = tx
    for batch_num in range(num_batches):
        start_index = batch_num * batch_size
        end_index = min((batch_num + 1) * batch_size, data_size)
        if start_index != end_index:
            yield shuffled_y[start_index:end_index], shuffled_tx[start_index:end_index]


# VISUALIZATION
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


# FOR LOGISTIC REGRESSION
def sigmoid(t):
    """apply sigmoid function on t."""
    if(len(t) > 1):
        res = np.zeros(len(t))
        for ind,el in enumerate(t):
            if(el < -1e8):
                #if the ith element of the vector t is either very large or small, we return the value 1 or 0 respectively
                #it would serve no purpose to calculate this value exactly and it can potentially cause errors
                res[ind] = 0.0
            elif(el > 1e8):
                res[ind] = 1.0
            else:
                res[ind] = np.exp(el)/(1+np.exp(el))
        return res
    elif(t < -1e8 or t > 1e8):
        return np.nan_to_num(t, nan=0.0, posinf=1, neginf=0)
    else:
        return np.exp(t)/(1+np.exp(t))

#nb : different from compute_loss used for linear regression
def calculate_loss_logistic(y, tx, w):
    """compute the cost by negative log likelihood."""
    return np.sum(np.log(1+np.exp(np.dot(tx,w)))-y*(np.dot(tx,w)), axis=0)

def calculate_gradient_logistic(y, tx, w):
    """compute the gradient of loss."""
    return np.dot(tx.T, sigmoid(np.dot(tx,w)) - y)

def calculate_hessian(y, tx, w):
    """return the hessian of the loss function."""
    diagonal_values=sigmoid(np.dot(tx,w))*(1-sigmoid(np.dot(tx,w)))
    #Not possible to do such a large diagonal matrix -> we mutliply the matrix by these values directly
    return np.dot(tx.T*diagonal_values,tx)

# Methods Balz - To merge

def compute_p(w,tX): 
    """
        Computes probabilities of all observations in tX corresponding to -1 = 'b' or 1 = 's' based on weights w according to the logistc transformation.
        (p = 0 correponds to -1 resp. 'b' and p = 1 to 1 resp. 's' to simplify the computations)
    """
    odds = np.exp(np.dot(tX,w))
    #odds = np.nan_to_num(np.exp(np.dot(tX,w)))
    return odds/(1+odds)

"""
def logistic_regression(y, tX, max_iters, threshold, gamma = 1):
    #Finds the most likely weights using the iterative Newton-Raphson method.
    
    # Define parameters to store w and loss
    w = np.zeros(len(tX[0,:]))
    ws = [w]
    log_likelihoods = [compute_log_likelihood(w,tX, y)]

    
    for n_iter in range(max_iters):
        
        with np.errstate(all='raise'):
            try:
                gradient = compute_gradient_log_likelihood(w, tX, y)
                jacobean = compute_jacobean_log_likelihood(w,tX, y)
                w = w - gamma*np.linalg.solve(jacobean,gradient)
                log_likelihood = compute_log_likelihood(w,tX, y)

            except FloatingPointError as e:      
                print(e)
                break
            except np.linalg.LinAlgError as e:
                print(e)
                break
        
        # store w and loss
        ws.append(w)
        log_likelihoods.append(log_likelihood)     
                          
        if n_iter % 1 == 0:
            print("Current iteration={i}, the loss={l}".format(i=n_iter, l=log_likelihood))
            
        if len(log_likelihoods) > 1 and np.abs(log_likelihoods[-1] - log_likelihoods[-2]) < threshold:
            print('reached threshold')
            break
        

    return log_likelihoods, ws

def sigmoid(t):
 
    # Apply sigmoid function on t. Allows to compute probabilities of all observations in tX corresponding to -1 = 'b' or 1 ='s' based on weights w according to the logistc transformation. (p = 0 correponds to -1 resp. 'b' and p = 1 to 1 resp. 's' to simplify the computations)
    
    odds = np.exp(t)
    return np.exp(t)/(1+np.exp(t))
    
    with np.errstate(all='raise'):
        try:
            odds = np.exp(t)
            return np.exp(t)/(1+np.exp(t))
        except FloatingPointError as e:      
            print(e)
            odds = np.nan_to_num(np.exp(t))
    return odds/(1+odds)
"""

def calculate_loss_logistic_regression(y, tX, w):
    "Computes the negative log likelihood of observing the data tX with the given weights w."
    return sum((np.log(1+np.exp(np.dot(tX, w)))) - y*np.dot(tX, w) )
    #sigma = sigmoid(np.dot(tX, w))
    #return -sum(y*np.log(sigma)+(1-y)*np.log(1-sigma))

def calculate_loss_penalized_logistic_regression(y, tX, w, lambda_):
    return calculate_loss_logistic_regression(y, tX, w) + lambda_/2*np.sum(w**2)

def calculate_gradient_logistic_regression(y, tX, w):
    """compute the gradient of loss."""
    #return np.dot(tX.T, sigmoid(np.dot(tX, w))- y)
    return np.dot(tX.T, compute_p(w,tX)- y)
    
    #sigma = sigmoid(np.dot(tX, w))
    #return -sum(y*np.log(sigma)+(1-y)*np.log(1-sigma))

def calculate_gradient_penalized_logistic_regression(y, tX, w, lambda_):
    """compute the gradient of loss."""
    return calculate_gradient_logistic_regression(y, tX, w) + lambda_*w

def calculate_hessian_logistic_regression(tX, w):
    """return the hessian of the loss function."""

    #sigma = sigmoid(np.dot(tX, w))
    sigma = compute_p(w,tX)
    return np.dot((sigma*(1-sigma))*tX.T,tX)

def calculate_hessian_penalized_logistic_regression(tX, w, lambda_):
    return calculate_hessian_logistic_regression(tX, w) + lambda_*np.eye(len(w))

def penalized_logistic_regression(y, tX, max_iters, threshold, lambda_, gamma = 1):
    """Finds the most likely weights using the iterative Newton-Raphson method."""
    
    # Define parameters to store w and loss
    w = np.zeros(len(tX[0,:]))
    ws = [w]
    log_likelihoods = [calculate_loss_penalized_logistic_regression(y, tX, w, lambda_)]

    
    for n_iter in range(max_iters):
        
        #with np.errstate(divide='raise',invalid='raise'):
        with np.errstate(all='raise'):
            try:
                gradient = calculate_gradient_penalized_logistic_regression(y, tX, w, lambda_)
                hessian = calculate_hessian_penalized_logistic_regression(tX, w, lambda_)
                w = w - np.linalg.solve(hessian,gradient)
                log_likelihood = calculate_loss_penalized_logistic_regression(y, tX, w, lambda_)

            except FloatingPointError as e:      
                print(e)
                
                break
            except np.linalg.LinAlgError as e:
                print(e)
               
                break
        
        # store w and loss
        ws.append(w)
        log_likelihoods.append(log_likelihood)     
                          
        #if n_iter % 1 == 0:
         #   print("Current iteration={i}, the loss={l}".format(i=n_iter, l=log_likelihood))
            
        if len(log_likelihoods) > 1 and np.abs(log_likelihoods[-1] - log_likelihoods[-2]) < threshold:
            print('reached threshold')
            break
        

    return log_likelihoods, ws


#PLOT
def plot_implementation(errors, lambdas):
    """
    errors and lambas should be list (of the same size) the error for a given lambda,
    * lambda[0] = 1
    * errors[0] = RMSE of a ridge regression of set
    """
    plt.semilogx(lambdas,errors, color='b', marker='*', label="Train Error RMSE")
    plt.xlabel("lambda")
    plt.ylabel("RMSE")
    leg = plt.legend(loc=1, shadow=True)
    leg.draw_frame(False)


def plot_train_test(train_errors, test_errors, lambdas):
    """
    train_errors, test_errors and lambas should be list (of the same size) the respective train error and test error for a given lambda,
    * lambda[0] = 1
    * train_errors[0] = RMSE of a ridge regression on the train set
    * test_errors[0] = RMSE of the parameter found by ridge regression applied on the test set
    """
    plt.semilogx(lambdas, train_errors, color='b', marker='*', label="Train error")
    plt.semilogx(lambdas, test_errors, color='r', marker='*', label="Test error")
    plt.xlabel("lambda")
    plt.ylabel("RMSE")
    leg = plt.legend(loc=1, shadow=True)
    leg.draw_frame(False)


