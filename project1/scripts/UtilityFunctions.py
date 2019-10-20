#!/usr/bin/env python
# coding: utf-8
"""some utility functions for project 1."""
import numpy as np
import matplotlib.pyplot as plt


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


