#!/usr/bin/env python
# coding: utf-8
"""regression functions for project 1."""


#NB : all functions should return (weight, loss)
#NB : we are only interested in the last one 

import numpy as np
from loss_computations import *
from gradient_descent import *
from functions_for_log_regression import *

def least_squares_GD(y, tx, initial_w, max_iters, gamma):
    """Gradient descent algorithm with mse."""
    # Linear regression using gradient descent 
    loss_new = 0
    w = initial_w
    n_iter = 0
    while(True):
        loss_old = loss_new
        n_iter += 1 
        grad = compute_gradient(y, tx, w)
        loss_new = compute_loss_linear(y, tx, w, method="mse")
        w = w - gamma*grad
        if(np.abs(loss_new-loss_old) < 1e-6 or n_iter >= max_iters):
            break
        
    print("Gradient Descent({bi}/{ti}): loss={l}, w0={w0}, w1={w1}".format(
         bi=n_iter, ti=max_iters, l=loss_new, w0=w[0], w1=w[1]))
    return w, loss_new

def least_squares_SGD(y, tx, initial_w, max_iters, gamma, batch_size=1):
    """Stochastic gradient descent algorithm."""
    # ***************************************************
    loss_new = 0
    n_iter = 0
    w = initial_w
    while(True):
        n_iter += 1
        loss_old = loss_new
        for minibatch_y, minibatch_tx in batch_iter(y, tx, batch_size):
                grad = compute_gradient(minibatch_y, minibatch_tx,w)
        grad = grad / batch_size    
        w = w - grad * gamma
        loss_new = compute_loss_linear(y, tx, w)
        if(np.abs(loss_new-loss_old) < 1e-6 or n_iter >= max_iters):
            break

    print("Gradient Descent({bi}/{ti}): loss={l}, w0={w0}, w1={w1}".format(
         bi=n_iter, ti=max_iters, l=loss_new, w0=w[0], w1=w[1]))
    return w, loss_new

def least_squares(y, tx):
    """calculate the least squares solution."""
    #least squares regression using normal equations
    w=np.linalg.solve(np.dot(tx.T,tx),np.dot(tx.T,y))
    lmse= compute_loss_linear(y, tx, w)
    
    print("Loss={l}, w0={w0}, w1={w1}".format(
          l=lmse, w0=w[0], w1=w[1]))
    return w, lmse
    
def ridge_regression(y, tx, lambda_):
    """ridge regression using normal equations."""
    w=np.linalg.solve((np.dot(tx.T,tx)+lambda_*2*y.shape[0]*np.eye(tx.shape[1])),np.dot(tx.T,y))
    loss= compute_loss_linear(y, tx, w)
    #print("Loss={l}, λ={la}".format(l=loss, la=lambda_))
    return w,loss


def logistic_regression(y, tX, initial_w, max_iters,gamma):
    """logistic regression using gradient descent"""
    w = initial_w
    loss_new = 0
    n_iter = 0
    while(True):
        with np.errstate(all='raise'):
            try:
                n_iter += 1
                loss_old = loss_new
                loss_new = calculate_loss_logistic(y,tX,w)
                gradient = calculate_gradient_logistic(y,tX,w)
                w = w - gamma * gradient  
                #print("Loss={l}, w0={w0}, w1={w1}".format(l=loss_new, w0=w[0], w1=w[1]))
            except FloatingPointError as e:
                print(e)
                break
            except np.linalg.LinAlgError as e:
                print(e)
                break
        if(np.abs(loss_new-loss_old) < 1e1 or n_iter >= max_iters):
            break
                  
    return w,loss_new

def logistic_regression_newton(y,tX,initial_w,max_iters,gamma) :
    """logistic regression using newton """
    w = initial_w
    loss_new = 0
    n_iter = 0
    while(True):
        with np.errstate(all='raise'):
            try:
                n_iter += 1
                loss_old = loss_new
                loss_new=calculate_loss_logistic(y, tX, w)
                gradient=calculate_gradient_logistic(y, tX, w)
                h=calculate_hessian(y, tX, w)
                w=w-gamma*np.dot(np.linalg.inv(h),gradient)
                #print("Loss={l}, w0={w0}, w1={w1}".format(
                 # l=loss_new, w0=w[0], w1=w[1]))
            except FloatingPointError as e:
                    print(e)
                    break
            except np.linalg.LinAlgError as e:
                    print(e)
                    break
        if(np.abs(loss_new-loss_old) < 1e2 or n_iter >= max_iters):
                    break
    
    return w,loss_new
    
def reg_logistic_regression(y, tX, lambda_, initial_w, max_iters, gamma):
    """logistic regression using newton comprising ridge term"""
    w = initial_w
    loss_new = 0
    n_iter = 0
    while(True):
        with np.errstate(all='raise'):
            try:
                n_iter += 1
                loss_old = loss_new
                loss_new=calculate_loss_logistic(y,tX,w)+lambda_/2*np.dot(w.T,w)
                gradient=calculate_gradient_logistic(y,tX,w) + lambda_*w
                hessian=calculate_hessian(y,tX,w) + lambda_*np.eye(len(w))
                w=w-gamma*np.dot(np.linalg.inv(hessian),gradient)
            except FloatingPointError as e:
                    print(e)
                    break
            except np.linalg.LinAlgError as e:
                    print(e)
                    break
        if(np.abs(loss_new-loss_old) < 1e2 or n_iter >= max_iters):
                    break           
    #print("Loss={l}, λ={la}, w0={w0}, w1={w1}".format(
        #  l=loss_new, la=lambda_, w0=w[0], w1=w[1]))
    return w, loss_new
