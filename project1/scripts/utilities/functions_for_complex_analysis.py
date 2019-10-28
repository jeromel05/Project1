#COMPLEX ANALYSIS
import numpy as np
import matplotlib.pyplot as plt
#import warnings
from plots import *
from split_test_train import *
from patternsmissingvalues import *

from matplotlib.collections import LineCollection
from matplotlib import colors



def calculate_signal_fraction_features_binned(tX,y,nb_bins):
    """ Bins data of each feature after removing -999. For each bin the ratio of #signal/(#signal+#bg) is computed.
        The value of of a bin's ratio into which no observation falls is NaN. 
            Parameters : - the feature matrix, 'tX'
                         - the corresponding values (s/bg), 'y'
                         - the  number of bins, 'nb_bins'
            Returns :    - the matrix containing the ratios for each feature, 'pseudo_likelihoods_binned'
                         - the matrix defining the bins for each feature, 'shared_bins'
                         - a list of arrays with the columns of tX cleared of missing values, 'tX_cols_no_missing_val'
    """    
    
    pseudo_likelihoods_binned = np.zeros((tX.shape[1],nb_bins+3))
    shared_bins = np.zeros((tX.shape[1],nb_bins+1))
    tX_cols_no_missing_val = []

    for ind_col in range(tX.shape[1]):
        # Removes absent values (-999s)
        tX_col_cleared = tX[:,ind_col][tX[:,ind_col]>-999]
        y_cleared = y[tX[:,ind_col]>-999]
        tX_cols_no_missing_val.append(tX_col_cleared)
    
        # Generating bins to make sure we use the same ones for the colors as we do for the heights of the histogram bars.
        shared_bins_col = np.histogram_bin_edges(tX_col_cleared, bins = nb_bins)
        shared_bins[ind_col,:] = shared_bins_col
    
    
        # Counting the number of background (y = -1) and signal (y = 1) in and finally the fraction of signal in each bin
        ind_tX_col_cleared = np.digitize(tX_col_cleared,shared_bins_col,right = False)
        nb_bg_bin = np.array([np.count_nonzero(ind_tX_col_cleared[y_cleared < 1/2] == i) for i in range(nb_bins+2)])
        nb_s_bin = np.array([np.count_nonzero(ind_tX_col_cleared[y_cleared > 1/2] == i) for i in range(nb_bins+2)])
        #with warnings.catch_warnings():
        #    warnings.simplefilter("ignore")
        frac_s = nb_s_bin/(nb_bg_bin+nb_s_bin)
        frac_s[0] = frac_s[1]
    
        # Save fraction signal in bin
        pseudo_likelihoods_binned[ind_col,:] = np.append(frac_s,frac_s[-1])
    
    return pseudo_likelihoods_binned,shared_bins,tX_cols_no_missing_val

def plot_features_fraction_signal_binned(tX,y,nb_bins):
    """ Plots the the ratio #signal/(#signal+#bg) in each bin for each feature. The absence of a point indicates 
        that the feature never took a value within that bin.
            Parameters : - the feature matrix, 'tX'
                         - the corresponding values (s/bg), 'y'
                         - the  number of bins, 'nb_bins'
        
    """
    pseudo_likelihoods_binned,shared_bins,tX_cols_no_missing_val = calculate_signal_fraction_features_binned(tX,y,nb_bins)
    #pseudo_likelihoods_binned[np.isnan(pseudo_likelihoods_binned)] = 0.55
    
    #figure
    num_row = 5
    num_col = int(np.ceil(tX.shape[1]/num_row))

    fig,ax = plt.subplots(num_row, num_col)
    #fig.suptitle('Pseudo proability distribution boson ', fontsize=180)
    fig.set_figheight(150)
    fig.set_figwidth(150)

    for ind_col,frac_s in enumerate(pseudo_likelihoods_binned):
    
        #ax[int(ind_col/num_col),ind_col%num_col].scatter(shared_bins[ind_col], frac_s[1:], s=200)
        ax[int(ind_col/num_col),ind_col%num_col].scatter(shared_bins[ind_col], frac_s[1:-1], s=200)
        
        ax[int(ind_col/num_col),ind_col%num_col].tick_params(axis='both', which='major', labelsize=100)
        #ax[int(ind_col/num_col),ind_col%num_col].set_title('feature '+ str(ind_col+1),fontsize=100)
        ax[int(ind_col/num_col),ind_col%num_col].set_ylim(-0.01,1.01)
        ax[int(ind_col/num_col),ind_col%num_col].set_xlabel('value feature #'+ str(ind_col+1), fontsize=90)
        ax[int(ind_col/num_col),ind_col%num_col].set_ylabel('#s/(#s+#bg) / bin',fontsize=90)
    
    plt.tight_layout()  
    plt.savefig("../plots/fraction_signal_binned_by_features")
    plt.show()
    
    
def calculate_pseudo_likelihood_groups(pseudo_likelihoods_binned,shared_bins,tX_test):
    """ Generates a vector of matrixes, each containing the 'pseudo likelihoods' atrributed to every value of 
        every feature of a testing data set separated into groups by patterns of missing values.
        This attribution of values is based on the ratio of signal per bin observed in a training data set.
            Parameters : - 'pseudo_likelihoods_binned', a 2D array conatining the pseudo-likelihoods estimated based on the 
                           training data
                         - 'shared_bins', a 2D array specifying the bins used to cut up the training data
                         - 'tX_test', the matrix to classify
            Returns : - 'pseudo_likelihoods_groups_test', an array of matrices of the same dimension as tX_test_split,
                            which contains the 'pseudo likelihoods' by groups attributed to each value of each feature
                      - 'y_pseudo_likelihood_pred_groups_test', an vector of arrays containing the each sample's 
                            'pseudo log likelihood' grouped by pattern of missing value.
                        
    
    """
    
    # setting pseudo likelihood of bins without values arbitrarily to 0.5
    pseudo_likelihoods_binned[np.isnan(pseudo_likelihoods_binned)] = 0.5
    
    tX_split_test, ind_row_groups_test, groups_mv_num_test = split_data_according_to_pattern_of_missing_values(tX_test)   
    pseudo_likelihoods_groups_test = np.copy(tX_split_test)
    
    y_pseudo_likelihood_pred_groups_test = []

    for ind_group, ind_row_group_test in enumerate(ind_row_groups_test):
        
        likelihoods_bins_group_test = pseudo_likelihoods_binned[tX_test[ind_row_group_test[1]] > -999]
        shared_bins_group_test = shared_bins[tX_test[ind_row_group_test[1]] > -999]
        for ind_col in range(likelihoods_bins_group_test.shape[0]):
            ind_bin_tX_split_col_test = np.digitize(tX_split_test[ind_group][:,ind_col],shared_bins_group_test[ind_col,:],right = False)
            pseudo_likelihoods_groups_test[ind_group][:,ind_col] = likelihoods_bins_group_test[ind_col][ind_bin_tX_split_col_test]
        
        y_pseudo_likelihood_pred_groups_test.append(np.nan_to_num(np.sum(np.log(pseudo_likelihoods_groups_test[ind_group]),axis=1)))
              
    return pseudo_likelihoods_groups_test, y_pseudo_likelihood_pred_groups_test


def matthews_correlation_coefficient(nb_true_positives,nb_false_positives,nb_true_negatives,nb_false_negatives):
    """
        Calculates a vector of matthews correlation coefficients, a measure of the error from:
            Parameters: - nb_true_positives, an array containing the number of true positives
                        - nb_false_positives, an array containing the number of false positives
                        - nb_true_negatives, an array containing the number of true negatives
                        - nb_false_negatives, an array containing the number of false negatives
            Returns : an array of matthews correlation coefficients
    """
    denom_matthews_correlation_coefficient = np.sqrt((nb_true_positives+nb_false_positives)*(nb_true_positives+nb_false_negatives)*(nb_true_negatives+nb_false_positives)*(nb_true_negatives+nb_false_negatives))
    denom_matthews_correlation_coefficient[denom_matthews_correlation_coefficient == 0] = 1
    return (nb_true_positives*nb_true_negatives-nb_false_positives*nb_false_negatives)/denom_matthews_correlation_coefficient


def predicition_pseudo_likelihood_method_groups(tX_test,pseudo_likelihoods_binned,shared_bins,thresholds_likelihood_groups):
    pseudo_likelihoods_groups_test, y_pseudo_likelihood_pred_groups_test = calculate_pseudo_likelihood_groups(pseudo_likelihoods_binned,shared_bins,tX_test)
    tX_split_test, ind_row_groups_test, groups_mv_num_test = split_data_according_to_pattern_of_missing_values(tX_test)   
    
    y_pred_test = np.zeros(tX_test.shape[0])
    y_pred_test_binary = np.zeros(tX_test.shape[0])

    for ind_group, ind_row_group_test in enumerate(ind_row_groups_test):
        y_pred_group_test_binary = np.copy(y_pseudo_likelihood_pred_groups_test[ind_group])
        y_pred_group_test_binary[y_pred_group_test_binary > thresholds_likelihood_groups[ind_group]] = 1
        y_pred_group_test_binary[y_pred_group_test_binary < 1] = -1
    
        y_pred_test[ind_row_group_test] = y_pseudo_likelihood_pred_groups_test[ind_group]
        y_pred_test_binary[ind_row_group_test] = y_pred_group_test_binary
    
    return y_pred_test_binary,y_pred_test
