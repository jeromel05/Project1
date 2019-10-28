#FOR LOGISTIC REGRESSION
import numpy as np
from math import sqrt

from patternsmissingvalues import *
from datapreprocessing import *
import matplotlib.pyplot as plt

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

def compute_p(w,tX): 
    """
        Computes probabilities of all observations in tX corresponding to -1 = 'b' or 1 = 's' based on weights w according to the logistc transformation.
        (p = 0 correponds to -1 resp. 'b' and p = 1 to 1 resp. 's' to simplify the computations)
    """
    #odds = np.exp(np.dot(tX,w))
    odds = np.nan_to_num(np.exp(np.dot(tX,w)))
    return odds/(1+odds)

def rmse_logistic(err,tX):
    return np.linalg.norm(err)/sqrt(tX.shape[0])

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

### Logistic regression on groups pattern missing values

def calculate_loss_logistic_regression(y, tX, w):
    "Computes the negative log likelihood of observing the data tX with the given weights w."
    return sum((np.log(1+np.exp(np.dot(tX, w)))) - y*np.dot(tX, w) )

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


def generate_predicitons_reg_logistic_regression_feature_engineering_groups(tX_test,tX,y,max_iters,threshold,lambda_star_groups,degrees_star_groups,gamma):
    "Generates predictions of the y values for a data set tX_test, plots predictions for the training set groupwise."
    tX_split_test, ind_row_groups_test, groups_mv_num_test = split_data_according_to_pattern_of_missing_values(tX_test)
    tX_split, ind_row_groups, groups_mv_num = split_data_according_to_pattern_of_missing_values(tX)
    
    assert(np.array_equal(groups_mv_num, groups_mv_num_test))
    
    w_star_groups = []
    y_pred_test = np.zeros(tX_test.shape[0])
    y_pred_train = np.zeros(tX.shape[0])

    num_col = 3
    num_row = 2
    f, axs = plt.subplots(num_row, num_col)

    for ind_group in range(len(groups_mv_num)):
        print('group (' + str(ind_group + 1) + '/' + str(len(groups_mv_num)) + ')')
    
        tX_group = tX[ind_row_groups[ind_group]]
        degree_group = degrees_star_groups[ind_group]
        tX_test_group = tX_test[ind_row_groups_test[ind_group]]
        y_group = rescale_y(y[ind_row_groups[ind_group]])
        lambda_group = lambda_star_groups[ind_group]
    
        tX_train_group_extended = add_higher_degree_terms(tX_group, degree_group)
        tX_test_group_extended = add_higher_degree_terms(tX_test_group, degree_group)
    
        # ***************************************************
        # calcualte most likely weights through logistic regression with ridge term
        # ***************************************************
        ind_col_non_const = np.arange(len(tX_train_group_extended[0,:]))[np.std(tX_train_group_extended,0)>0]
    
        tX_train_group_extended = adding_offset(tX_train_group_extended)
        tX_test_group_extended = adding_offset(tX_test_group_extended)
        ind_col_non_const += 1
        ind_col_non_const = np.insert(ind_col_non_const,0, 0)
    
        log_likelihoods, ws = penalized_logistic_regression(y_group, tX_train_group_extended[:,ind_col_non_const], max_iters, threshold,lambda_group,gamma)
    
        ind_min = np.argmin(log_likelihoods)
        w_star = np.zeros(len(tX_train_group_extended[0,:]))
        w_star[ind_col_non_const] = ws[ind_min]

        # ***************************************************
        # calculate RMSE and ABSE for train data,and store them in rmse_tr and abse_tr respectively
        # ***************************************************
        w_star_groups.append(w_star)
        y_pred_train[ind_row_groups[ind_group]] = compute_p(w_star,tX_train_group_extended)
        y_pred_test[ind_row_groups_test[ind_group]] = compute_p(w_star,tX_test_group_extended)
    
        log_likelihoods_train = calculate_loss_logistic_regression(y_group, tX_train_group_extended, w_star)
        rmse_tr = np.linalg.norm(y_group - compute_p(w_star,tX_train_group_extended))/sqrt(y_group.shape[0])
        abse_tr = np.sum(abs(y_group - [compute_p(w_star,tX_train_group_extended) > 0.5]))
    
        ax = axs[ind_group // num_col][ind_group % num_col]
        ax.scatter(y_pred_train[ind_row_groups[ind_group]][0:200],y_group[0:200],s=1)
        ax.set_xlabel("y pred train")
        ax.set_ylabel("y train")
        
        print("group={g}, degree={d} , lambda={l:10.3e}, Training RMSE={tr:.3f}, Training loss={trl:.3f}, Training # Missclassification ={m_tr:.3f}".format(
            g=ind_group,d=degree_group,  l= lambda_group, tr=rmse_tr, trl = log_likelihoods_train, m_tr = abse_tr))

    plt.tight_layout()
    plt.savefig("../plots/predictions_vs_y_value_training_groups_patter")
    plt.show()
    y_pred_test = rescale_predictions(y_pred_test)
    
    return y_pred_test, w_star_groups
