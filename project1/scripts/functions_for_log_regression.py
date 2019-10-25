#FOR LOGISTIC REGRESSION
import numpy as np

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
