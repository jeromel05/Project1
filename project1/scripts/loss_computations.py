import numpy as np

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