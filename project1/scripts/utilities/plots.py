#PLOT
import numpy as np
import matplotlib.pyplot as plt

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

    
import warnings
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
            #warnings.simplefilter("ignore")
        frac_s = nb_s_bin/(nb_bg_bin+nb_s_bin)
        frac_s[0] = frac_s[1]
    
        # Save fraction signal in bin
        pseudo_likelihoods_binned[ind_col,:] = np.append(frac_s,frac_s[-1])
    
    return pseudo_likelihoods_binned,shared_bins,tX_cols_no_missing_val


from matplotlib.collections import LineCollection
from matplotlib import colors

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