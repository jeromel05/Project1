#!/usr/bin/env python
# coding: utf-8
"""regression functions for project 1."""


#NB : all functions should return (weight, loss)
#NB : we are only interested in the last one 

import numpy as np
from UtilityFunctions import *

def least_squares_GD(y, tx, initial_w, max_iters, gamma):
    """Gradient descent algorithm with mse."""
    # Linear regression using gradient descent 
    loss = 0
    w = initial_w
    for n_iter in range(max_iters):
        grad = compute_gradient(y, tx, w)
        loss = compute_loss_linear(y, tx, w, method="mse")
        w = w - gamma*grad
        
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
        loss = compute_loss_linear(y, tx, w)

    print("Gradient Descent({bi}/{ti}): loss={l}, w0={w0}, w1={w1}".format(
         bi=n_iter, ti=max_iters - 1, l=loss, w0=w[0], w1=w[1]))
    return w, loss

def least_squares(y, tx):
    """calculate the least squares solution."""
    #least squares regression using normal equations
    w=np.linalg.solve(np.dot(tx.T,tx),np.dot(tx.T,y))
    lmse= compute_loss_linear(y, tx, w)
    
    print("Loss={l}, w0={w0}, w1={w1}".format(
          l=lmse, w0=w[0], w1=w[1]))
    return w, lmse
    
def ridge_regression(y, tx, lambda_):
    #ridge regression using normal equations
    w=np.linalg.solve((np.dot(tx.T,tx)+lambda_*2*y.shape[0]*np.eye(tx.shape[1])),np.dot(tx.T,y))
    loss= compute_loss_linear(y, tx, w)
    #print("Loss={l}, λ={la}".format(l=loss, la=lambda_))
    return w,loss


def logistic_regression(y, tX, initial_w, max_iters,gamma):
    """logistic regression using gradient descent"""
    w=initial_w
    for iter in range(max_iters):
         with np.errstate(all='raise'):
            try:
                loss= calculate_loss_logistic(y,tX,w)
                gradient=calculate_gradient_logistic(y,tX,w)
                w=w+gamma*gradient  
                print("Loss={l}, w0={w0}, w1={w1}".format(l=loss, w0=w[0], w1=w[1]))
            except FloatingPointError as e:
                    print(e)
                    break
            except np.linalg.LinAlgError as e:
                    print(e)
                    break
                
    return w,loss

def logistic_regression_newton(y,tX,initial_w,max_iters,gamma) :
    """logistic regression using newton """
    w=initial_w
    for iter in range(max_iters):
        with np.errstate(all='raise'):
            try:
                loss=calculate_loss_logistic(y, tX, w)
                gradient=calculate_gradient_logistic(y, tX, w)
                h=calculate_hessian(y, tX, w)
         
                w=w-gamma*np.dot(np.linalg.inv(h),gradient)
            except FloatingPointError as e:
                    print(e)
                    break
            except np.linalg.LinAlgError as e:
                    print(e)
                    break
    print("Loss={l}, w0={w0}, w1={w1}".format(
                  l=loss, w0=w[0], w1=w[1]))
    return w,loss
    
def reg_logistic_regression(y, tX, lambda_, initial_w, max_iters, gamma):
    w=initial_w
    for iter in range(max_iters):
        with np.errstate(all='raise'):
            try:
                loss=calculate_loss_logistic(y,tX,w)+lambda_/2*np.dot(w.T,w)
                gradient=calculate_gradient_logistic(y,tX,w)
                hessian=calculate_hessian(y,tX,w)
                w=w-gamma*np.dot(np.linalg.inv(hessian),gradient)
            except FloatingPointError as e:
                    print(e)
                    break
            except np.linalg.LinAlgError as e:
                    print(e)
                    break
    print("Loss={l}, w0={w0}, w1={w1}".format(
          l=loss, w0=w[0], w1=w[1]))
    return w, loss
