#!/usr/bin/env python
# coding: utf-8
"""regression functions for project 1."""


#NB : all functions should return (weight, loss)
#NB : we are only interested in the last one 
# il manque logistic regression using gradient descent (or SGD)  et reg logistic regression

import numpy as np
from UtilityFunctions import *

def least_squares_GD(y, tx, initial_w, max_iters, gamma):
    """Gradient descent algorithm with mse."""
    # Linear regression using gradient descent 
    # Define parameters to store w and loss
    loss = 0
    w = initial_w
    for n_iter in range(max_iters):
        # ***************************************************
        grad = compute_gradient(y, tx, w)
        loss = compute_loss(y, tx, w, method="mse")
        w = w - gamma*grad
        # ***************************************************
        #if not(n_iter%30):
    #print("Gradient Descent({bi}/{ti}): loss={l}, w0={w0}, w1={w1}".format(
    #bi=n_iter, ti=max_iters - 1, l=loss, w0=w[0], w1=w[1]))
    print("Gradient Descent({bi}/{ti}): loss={l}, w0={w0}, w1={w1}".format(
        bi=n_iter, ti=max_iters - 1, l=loss, w0=w[0], w1=w[1]))
    return w, loss

def least_squares_SGD(y, tx, initial_w, max_iters, gamma, batch_size=1):
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
        #if not(n_iter%10):
            #print("Gradient Descent({bi}/{ti}): loss={l}, w0={w0}, w1={w1}".format(
                  #bi=n_iter, ti=max_iters - 1, l=loss, w0=w[0], w1=w[1]))
    # ***************************************************
    print("Gradient Descent({bi}/{ti}): loss={l}, w0={w0}, w1={w1}".format(
         bi=n_iter, ti=max_iters - 1, l=loss, w0=w[0], w1=w[1]))
    return w, loss

def least_squares(y, tx):
    """calculate the least squares solution."""
    #least squares regression using normal equations
    # ***************************************************

    w=np.linalg.solve(np.dot(tx.T,tx),np.dot(tx.T,y))
    lmse= compute_loss(y, tx, w)
    print("Loss={l}, w0={w0}, w1={w1}".format(
          l=lmse, w0=w[0], w1=w[1]))
    return w, lmse
    # ***************************************************
    
def ridge_regression(y, tx, lambda_):
    #ridge regression using normal equations
    # ***************************************************
    w=np.linalg.solve((np.dot(tx.T,tx)+lambda_*2*y.shape[0]*np.eye(tx.shape[1])),np.dot(tx.T,y))
    lmse= compute_loss(y, tx, w)
    # ***************************************************
    return w,lmse


def logistic_regression(y, tX, w_initial, max_iters,gamma):
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
                return ws,log_likelihoods
        
        try:
            w = w - gamma*np.linalg.solve(jacobean,gradient)
        except np.linalg.LinAlgError as e:
            print(e)
            #print(log_likelihood,gradient,jacobean)
            print(tX, n_iter)
            
            return ws,log_likelihoods
            
        
        # store w and loss
        ws.append(w)
        log_likelihood = compute_log_likelihood(w,tX, y)
        log_likelihoods.append(log_likelihood)


    return  ws,log_likelihoods


def reg_logistic_regression(y, tx, lambda_, initial_w, max_iters, gamma)
    raise NotImplementedError
    return ws, log_lokelihoods
