import numpy as np
import matplotlib.pyplot as plt
from math import sqrt

import sys
sys.path.insert(1, '../utilities/')

from proj1_helpers import *
from datapreprocessing import *
from patternsmissingvalues import *
from split_test_train import *
from functions_for_log_regression import *
from loss_computations import *
from functions_for_complex_analysis import *

## include rescale_y ect and generate_predicitons_reg_logistic_regression_feature_engineering_groups
DATA_TRAIN_PATH = '../../data/train.csv'
y, tX, ids = load_csv_data(DATA_TRAIN_PATH)
DATA_TEST_PATH = "../../data/test.csv"
_, tX_test, ids_test = load_csv_data(DATA_TEST_PATH)

#Those hyperparameters were selected using other methods in the prject1 notebook
lambda_star_groups = np.array([1.3257113655901095e-04,6.8664884500429981e-04,8.6851137375135298e-04,6.2505519252739761e-01,2.4420530945486499e-01,3.3932217718953299e-04])
degrees_star_groups = np.array([9,3,3,4,4,3])

max_iters = 20
threshold = 10**(-8)
gamma = 1

y_pred_test, w_star_groups = generate_predicitons_reg_logistic_regression_feature_engineering_groups(tX_test,tX,y,max_iters,threshold,lambda_star_groups,degrees_star_groups,gamma)

OUTPUT_PATH = '../../data/predicted_final.csv'
y_pred_submission = np.copy(y_pred_test)
y_pred_submission[np.where(y_pred_test <= 0)] = -1
y_pred_submission[np.where(y_pred_test > 0)] = 1
create_csv_submission(ids_test, y_pred_submission, OUTPUT_PATH)
