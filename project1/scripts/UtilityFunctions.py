#!/usr/bin/env python
# coding: utf-8
"""some utility functions for project 1."""
import numpy as np

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
    
def compute_gradient(y, tx, w):
    """Compute the gradient."""
    n = np.shape(tx)[0]
    error = y - np.dot(tx,w)
    return - np.dot(tx.T,error)/n


def leastsquaresGD(y, tx, initial_w, max_iters, gamma):
    """Gradient descent algorithm with mse."""
    # Define parameters to store w and loss
    loss = 0
    w = initial_w
    for n_iter in range(max_iters):
        # ***************************************************
        grad = compute_gradient(y, tx, w)
        loss = compute_loss(y, tx, w, method="mse")
        w = w - gamma*grad
        # ***************************************************
        if not(n_iter%30):
            print("Gradient Descent({bi}/{ti}): loss={l}, w0={w0}, w1={w1}".format(
              bi=n_iter, ti=max_iters - 1, l=loss, w0=w[0], w1=w[1]))

    return loss, w

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
            
def leastsquaresSGD(y, tx, initial_w, max_iters, gamma, batch_size=1):
    """Stochastic gradient descent algorithm."""
    # ***************************************************
    grad = 0
    loss = 0
    w = initial_w
    for n_iter in range(max_iters):
        for minibatch_y, minibatch_tx in batch_iter(y, tx, batch_size):
                grad = compute_gradient(minibatch_y, minibatch_tx,w)
        grad = grad / batch_size    
        w = w - grad * gamma
        loss = compute_loss(y, tx, w)
        if not(n_iter%10):
            print("Gradient Descent({bi}/{ti}): loss={l}, w0={w0}, w1={w1}".format(
                  bi=n_iter, ti=max_iters - 1, l=loss, w0=w[0], w1=w[1]))
    # ***************************************************
    return loss, w

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


# In[ ]:




