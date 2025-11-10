from seaborn import displot, kdeplot
from sklearn.model_selection import train_test_split
from sklearn.impute import KNNImputer
import matplotlib.pyplot as plt
from log_code import setup_logging
logger=setup_logging('trimming_log')
from feature_engine.transformation import LogCpTransformer,ArcsinTransformer,PowerTransformer
import seaborn as sns
import numpy as np
import pandas as pd
import sys
from sklearn.preprocessing import MinMaxScaler,QuantileTransformer,PowerTransformer
from scipy import stats
from feature_engine.outliers import Winsorizer

def trim_tech(X_train_num,X_test_num):
    try:
        winso=Winsorizer(capping_method='iqr',tail='both',fold=1.5,variables=['tenure_qt', 'MonthlyCharges_qt', 'TotalCharges_qt', 'charges_qt'])
        X_train_trim=winso.fit_transform(X_train_num)
        X_test_trim=winso.fit_transform(X_test_num)
        logger.info('Winzo Technique Successfully Applied')
        logger.info(f'After Trimming The X_train_num Columns : {X_train_trim.columns} ')
        # for i in X_train_trim:
        #     plt.title('Box Plot after Winsorizer Technique')
        #     sns.boxplot(x=X_train_trim[i],color='r',label=i)
        #     plt.show()

        return X_train_trim,X_test_trim

    except Exception as e:
        er_ty, er_msg, er_lin = sys.exc_info()
        logger.info(f'Issue is : {er_lin.tb_lineno} : due to : {er_msg}')



