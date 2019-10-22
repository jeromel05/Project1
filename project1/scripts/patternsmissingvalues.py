import numpy as np

def set_missing_explanatory_vars_to_mean(tX):
    # replace -999 by average value
    # Missing values appear as -999 in the data. The function sets them to the mean of the successful measurements of the variable.
    tX_corr = tX.copy()
    for i in range(tX.shape[1]):
        mean_xi = np.mean(tX[:,i][np.not_equal(tX[:,i],-999*np.ones(len(tX[:,i])))])
        tX_corr[:,i][np.where(tX_corr[:,i] == -999)] = mean_xi
    return tX_corr


# Taking into account their pattern for the handling of missing values
def split_data_according_to_pattern_of_missing_values(tX):
    
    """
        This function separates the data tX into a list of of arrays containing only instances of the explanatory
        variable with the same values missing. It returns:
            - 'tX_split', a list of arrays of arrays with the same pattern of missing values
            - 'ind_row_groups', a list of 1D arrays containing the subgroup's rows indices (ids) they had
                                in the orginal data matrix tX.
            - 'groups_mv_num', an array containing a numerical, binary representation of each subgroup's pattern of missing
                               values in the columns of tX that lack any value. Can be used to verify if tX and tX_test
                               are divided into the same groups and in the same order.
            (- 'bool_mask_col_mv_groups', an array containing the same information as 'groups_mv_num' but in the form of a bolean matrix.)
    """
    
    # Extracting ensemble of columns that contain missing values.
    ind_col = np.arange(tX.shape[1])
    ind_col_mv = ind_col[np.where(sum(tX == -999) > 0)]

    tX_cols_mv = tX[:,ind_col_mv]
    
    # Simplifying by taking '0' and '1' to represent present and absent values, respectively. The order of samples is preserved.
    
    pattern_cols_mv = np.zeros(tX_cols_mv.shape)
    pattern_cols_mv[np.where(tX_cols_mv == -999)] = 1
    
    pattern_cols_mv_num = np.dot(pattern_cols_mv,np.flip(np.power(10,np.arange(pattern_cols_mv.shape[1]),dtype=np.int64))) # Numerical (binary) representation of absence pattern in conserned columns.
    
    groups_mv_num = np.unique(pattern_cols_mv_num) # All observed patterns of missing values in the columns with gaps.
    
    ind_row = np.arange(tX.shape[0]) # Indices of tX's row
    
    tX_split = []
    ind_row_groups = []
    #bool_mask_col_mv_groups = np.array([tX[0,:] == -999])
    
    
    for group_mv_num in groups_mv_num:
        # calculating and stocking the indices of the rows of tX that belong to the group

        ind_row_group = ind_row[pattern_cols_mv_num == group_mv_num]
        ind_row_groups.append(ind_row_group)
        
        
        # extracting subset of tX faling into the group. Removing missing columns and stocking their positions.
        tX_group_rows = tX[ind_row_group,:]
        bool_mask_col_mv_group = [tX_group_rows[0,:] > -999]
        #bool_mask_col_mv_groups= np.vstack((bool_mask_col_MV_groups,bool_mask_col_MV_group))
        
        ind_col = np.arange(tX_group_rows.shape[1])
        
        tX_split.append(tX_group_rows[:,ind_col[bool_mask_col_mv_group]])
    
    #bool_mask_col_mv_groups = np.delete(bool_mask_col_MV_groups,[0],axis = 0)

    
    return tX_split, ind_row_groups, groups_mv_num 
    #bool_mask_col_mv_groups, (np.logical_not(bool_mask_col_mv_groups.flatten())).reshape(bool_mask_col_mv_groups.shape)

def split_y_according_to_pattern_of_missing_values(y, ind_row_groups):
    """
        This function splits y into a list of 1D arrays, y_split, according to the pattern found in the data matrix tX.
        The parameter 'ind_row_groups' gets calculated when splitting tX by means of 'split_data_accoring_to pattern_of_missing_values'.
    """
    y_split = []
    
    for ind_row_group in ind_row_groups:
        y_split.append(y[ind_row_group])
    
    return y_split

## Account for pattern of missing values combined with logistic regression


def find_optimal_weights_pattern_mv(tX_split, ind_row_groups,y_split,len_y):
    ws_groups = []
    y_pred = np.zeros(len_y)
    
    for tX_group, y_group, ind_row_group in zip(tX_split,y_split,ind_row_groups):
        
        #Test
        print(ind_row_group)
        ind_col_non_zero = np.arange(len(tX_group[0,:]))[sum(tX_group**2)>0]
        #log_likelihoods, ws = find_optimal_w(rescale_y(y_group), standardize(tX_group[:,ind_col_non_zero]), np.random.rand(tX_group[:,ind_col_non_zero].shape[1])/1000000,10)
       #log_likelihoods, ws = find_optimal_w(rescale_y(y_group), tX_group[:,ind_col_non_zero], np.random.rand(tX_group[:,ind_col_non_zero].shape[1])/(100000*(np.std(tX_group[:,ind_col_non_zero], axis=0)+1)*np.mean(tX_group[:,ind_col_non_zero], axis=0)),15)
        log_likelihoods, ws = find_optimal_w(rescale_y(y_group), tX_group[:,ind_col_non_zero], np.random.rand(tX_group[:,ind_col_non_zero].shape[1])/(10000*tX_group[:,ind_col_non_zero].shape[1]*(np.std(tX_group[:,ind_col_non_zero], axis=0)+1)*np.mean(tX_group[:,ind_col_non_zero], axis=0)),10)
    
        w = np.zeros(len(tX_group[0,:]))
        w[ind_col_non_zero] = ws[np.argmax(log_likelihoods)]
        ws_groups.append(w)
        print(log_likelihoods,ws)
        y_pred[ind_row_group] = rescale_predictions(compute_p(w,tX_group))
    
    return ws_groups, y_pred

def predict_y_lr_weights_pattern_mv(ws_groups, tX_split_test, ind_row_groups_test, len_y_test):
    
    y_pred_test = np.zeros(len_y_test)

    for tX_group_test, w_group, ind_row_group_test in zip(tX_split_test, ws_groups, ind_row_groups_test) :

        y_pred_test[ind_row_group_test] = rescale_predictions(compute_p(w_group,tX_group_test))
    
    return y_pred_test

"""
def predict_y_lr_weights_pattern_mv(ws_groups, tX_split_test, ind_row_groups_test, len_y_test):
    
    y_pred_test = np.zeros(len_y_test)

    for tX_group_test, w_group, ind_row_group_test in zip(tX_split_test, ws_groups, ind_row_groups_test) :

        y_pred_test[ind_row_group_test] = rescale_predictions(compute_p(w_group,tX_group_test))
    
    return y_pred_test
"""

## Account for pattern of missing values and interaction terms

def add_exponential_terms_to_split_data(tX_split, degree = 2):
    tX_split_interaction_terms = []

    for tX_group in tX_split:
        tX_split_interaction_terms.append(add_higher_degree_terms(tX_group, degree))
    
    return tX_split_interaction_terms

def add_interaction_terms_to_split_data(tX_split):
    tX_split_interaction_terms = []

    for tX_group in tX_split:
        tX_split_interaction_terms.append(add_interaction_terms(tX_group))
    
    return tX_split_interaction_terms