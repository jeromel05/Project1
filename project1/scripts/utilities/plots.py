#PLOT
import numpy as np
import matplotlib.pyplot as plt

from functions_for_complex_analysis import *

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


def plot_distribution_features_color_fraction_signal_binned(tX,y,nb_bins):
    """ Plots a histogram of each feature's distribution. The columns are colored as a function of the 
        ratio #signal/(#signal+#bg) in the corresponding bin. 'nb_bins' bins of identical width cover all values 
        between the minimal and the maximal value the feature takes.
            Parameters : - the feature matrix, 'tX'
                         - the corresponding values (s/bg), 'y'
                         - the  number of bins, 'nb_bins'
    """
    # Computing the ratio #signal/(#signal+#bg) for each bin.
    pseudo_likelihoods_binned,shared_bins,tX_cols_no_missing_val = calculate_signal_fraction_features_binned(tX,y,nb_bins)
    
    #figure
    fig,ax = plt.subplots(5, 6)
    #fig.suptitle('Distribution of the features values', fontsize=190)
    fig.set_figheight(150)
    fig.set_figwidth(150)


    # Normalize color
    norm = colors.Normalize(0, 1)
    for ind_col in range(tX.shape[1]):
   
        # Plot settings
        ax[int(ind_col/6),ind_col%6].tick_params(axis='both', which='major', labelsize=90)
        #ax[int(ind_col/6),ind_col%6].set_title('feature '+ str(ind_col+1),fontsize=90)
        ax[int(ind_col/6),ind_col%6].set_xlabel('value feature #'+ str(ind_col+1), fontsize=90)
        ax[int(ind_col/6),ind_col%6].set_ylabel('#obs/bin',fontsize=90)
    
        # Plot histogram for feature
        N, bins, patches = ax[int(ind_col/6),ind_col%6].hist(tX_cols_no_missing_val[ind_col], bins=shared_bins[ind_col], log=False)
    
        # Color histogram bars by fraction of signal in corresponding bin
        for frac, patch in zip(pseudo_likelihoods_binned[ind_col,1:-1], patches):
            color = plt.cm.viridis(norm(frac))
            patch.set_facecolor(color)
        
    # Create a continuous norm to map from data points to colors
    cmap = plt.cm.viridis
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm._A = []
    fig.subplots_adjust(right=0.95)
    cbar_ax = fig.add_axes([1.0, 0.3, 0.005, 0.3])
    cbar = plt.colorbar(sm, cax=cbar_ax)
    cbar.ax.set_ylabel('Proportion of y = s', rotation=90,fontsize=90)
    cbar.set_ticks([0.0, 0.25, 0.5, 0.75, 1.0])
    cbar.ax.set_yticklabels([0.0, 0.25, 0.5, 0.75, 1.0],fontsize=90)

    #plt.suptitle("Distribution of all the features", y = 1.0, fontsize=130)
    plt.tight_layout()
    plt.savefig("../plots/binned_distribution_of_features_colored_fraction_signal",)
    plt.show()
    

def find_threshold_pseudo_likelihood_method_groups(tX,y,nb_bins):
    """ Finds the optimal thresholds for classification according to 'pseudo log likelihoods' for each group with its
        particular pattern of missing values. It does so by applying the 'pseudo likelihood' method to the training set
        pased on which the 'pseudo likelihoods' are estimated in the first place. The threshold is chosen to maximize
        the Matthews correlation coefficient.
            Parameters: - tX, the training set based on which the 'pseudo likelihoods' are originally estimated
                        - y, the corresponding y values
                        - nb_bins, the number of bins into which the features separated
    """
    pseudo_likelihoods_binned,shared_bins,tX_cols_no_missing_val = calculate_signal_fraction_features_binned(tX,y,nb_bins)
    pseudo_likelihoods_groups, y_pseudo_likelihood_pred_groups = calculate_pseudo_likelihood_groups(pseudo_likelihoods_binned,shared_bins,tX)
    tX_split, ind_row_groups, groups_mv_num = split_data_according_to_pattern_of_missing_values(tX)
    y_split = split_y_according_to_pattern_of_missing_values(y, ind_row_groups)
    
    num_row = 2
    num_col = int(np.ceil(2*len(y_split)/num_row))
    fig,ax = plt.subplots(num_row, num_col)
    #fig.suptitle('Thresholds maximizing matthews correlation coefficient', fontsize=180)
    fig.set_figheight(75)
    fig.set_figwidth(150)
    
    assert(len(y_split)==len(y_pseudo_likelihood_pred_groups))
    
    thresholds_likelihood_groups = np.zeros(len(ind_row_groups))
    
    for ind_group, y_pred_group in enumerate(y_pseudo_likelihood_pred_groups):

        
        ind_sorted = np.argsort(y_pred_group)
        nb_true_positives = np.zeros(len(ind_sorted))
        nb_false_positives = np.zeros(len(ind_sorted))
        nb_true_negatives = np.zeros(len(ind_sorted))
        nb_false_negatives = np.zeros(len(ind_sorted))
        
        for i in range(len(ind_sorted)):
            nb_true_positives[i] = np.count_nonzero(y_split[ind_group][ind_sorted[i+1:]] == 1)
            nb_false_positives[i] = len(ind_sorted)-i-nb_true_positives[i]
            nb_true_negatives[i] = np.count_nonzero(y_split[ind_group][ind_sorted[0:i]] == -1)
            nb_false_negatives[i] = i-nb_true_negatives[i]
    
        mcc = matthews_correlation_coefficient(nb_true_positives,nb_false_positives,nb_true_negatives,nb_false_negatives)
  
        thresholds_likelihood_groups[ind_group] = y_pred_group[ind_sorted[np.argmax(mcc)]]
        
        
        ax[0,ind_group].scatter(y_pred_group,y_split[ind_group],s=200)
        ax[0,ind_group].scatter(thresholds_likelihood_groups[ind_group]*np.ones(50),np.linspace(-1, 1,50),s=200)
        ax[0,ind_group].tick_params(axis='both', which='major', labelsize=100)
        ax[0,ind_group].set_title('group '+ str(ind_group+1),fontsize=100)
        
        #ax[1,ind_group].scatter(np.arange(len(y_pred_group)),mcc,s=200)
        ax[1,ind_group].scatter(y_pred_group[ind_sorted],mcc,s=200)
        ax[1,ind_group].tick_params(axis='both', which='major', labelsize=100)
        ax[1,ind_group].set_xlabel("pseudo likelihood",fontsize=100)
        
    
    ax[0,0].set_ylabel("s/bg",fontsize=100)
    ax[1,0].set_ylabel("matthews correlation coefficient",fontsize=100)
    
    fig.align_ylabels(ax[:,:])
    plt.tight_layout()
    plt.savefig("../plots/proability_boson_explanatory_vars")
    plt.show()
    return thresholds_likelihood_groups