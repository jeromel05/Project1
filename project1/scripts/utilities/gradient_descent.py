#FOR GRADIENT DESCENT
import numpy as np

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
