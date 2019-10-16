#!/usr/bin/env python
# coding: utf-8
"""some utility functions for project 1."""
import numpy as np
import matplotlib.pyplot as plt


#LOSS
def compute_loss(y, tx, w, method="mse"):
    """Calculate the loss.

    You can calculate the loss using mse or mae.
    """
    error = y - np.dot(tx,w)
    if method == "mae":
        return np.sum(np.abs(error)) / np.shape(y)[0] / 2
    elif method == "mse":
        return np.inner(error,error) / np.shape(y)[0] / 2 #for MSE
    else:
        raise Exception("spam")
        
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
    # ***************************************************
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
    # **************************************************
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

def subgradient_descent(y, tx, initial_w, max_iters, gamma):
    """Subgradient descent algorithm using mae cost function."""
    # Define parameters to store w and loss
    ws = [initial_w]
    losses = []
    w = initial_w
    for n_iter in range(max_iters):
        # ***************************************************
        grad = compute_subgradient(y, tx, w)
        loss = compute_loss(y, tx, w, method="mae")
        w = w - gamma*grad
        # ***************************************************
        # store w and loss
        ws.append(w)
        losses.append(loss)
        print("Gradient Descent({bi}/{ti}): loss={l}, w0={w0}, w1={w1}".format(
              bi=n_iter, ti=max_iters - 1, l=loss, w0=w[0], w1=w[1]))

    return losses[-1], ws[-1]


def stochastic_subgradient_descent(
        y, tx, initial_w, batch_size, max_iters, gamma):
    """Stochastic gradient descent algorithm."""
    # ***************************************************
    ws = [initial_w]
    grad = 0
    losses = []
    w = initial_w
    for i in range(max_iters):
        for minibatch_y, minibatch_tx in batch_iter(y, tx, batch_size):
                grad = compute_subgradient(minibatch_y, minibatch_tx,w)
        grad = grad / batch_size    
        w = w - grad * gamma
        ws.append(w)
        loss=compute_loss(y, tx, w)
        losses.append(loss)
        print("GD: loss={l}, w0={w0}, w1={w1}".format(
             l=loss, w0=w[0], w1=w[1]))
    # ***************************************************
    return losses[-1], ws[-1]


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


