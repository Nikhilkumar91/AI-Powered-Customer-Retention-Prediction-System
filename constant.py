import numpy as np
import pandas as pd
import sys
import logging
from log_code import setup_logging
logger = setup_logging('constant')
import warnings
warnings.filterwarnings('ignore')
from sklearn.feature_selection import VarianceThreshold
reg = VarianceThreshold(threshold=0)

def con(train_num,test_num):
    try:
        reg.fit(train_num)
        logger.info(f'Total columns : {train_num.shape[1]} -> without variance 0 : {sum(reg.get_support())} -> with Variance 0 : {sum(~reg.get_support())}')
        logger.info(f'Variance 0 : names : {train_num.columns[~reg.get_support()]}')

        logger.info(f'Train Column Names : {train_num.columns}')
        logger.info(f'Test Column Names : {test_num.columns}')
        return train_num,test_num

    except Exception as e:
        er_ty, er_msg, er_lin = sys.exc_info()
        logger.info(f'Issue is : {er_lin.tb_lineno} : due to : {er_msg}')