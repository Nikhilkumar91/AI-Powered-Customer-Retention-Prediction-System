import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import scipy.stats as stats
import sys
import logging
from log_code import setup_logging
logger = setup_logging('feature_selection')
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OrdinalEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.feature_selection import chi2, SelectKBest
from sklearn.feature_selection import f_classif
def ordinal_encoding(train_cat,test_cat,y_train,y_test):
    try:
        # label encoding
        lb = LabelEncoder()
        lb.fit(y_train)
        y_train = lb.transform(y_train)
        y_test = lb.transform(y_test)
        # ordinal encoding
        contract_order = [['Month-to-month', 'One year', 'Two year']]
        ordinal_encoder = OrdinalEncoder(categories=contract_order)
        train_cat['Contract'] = ordinal_encoder.fit_transform(train_cat[['Contract']])
        test_cat['Contract'] = ordinal_encoder.transform(test_cat[['Contract']])

        # one hot encoding
        drop_cols = ['JoinDate', 'Churn','gateway','tax' ]
        train_cat =train_cat.drop(columns=drop_cols, errors='ignore')
        test_cat = test_cat.drop(columns=drop_cols, errors='ignore')
        ohe_cols = [
            'gender', 'SeniorCitizen', 'Partner', 'Dependents', 'PhoneService',
            'MultipleLines', 'InternetService', 'OnlineSecurity', 'OnlineBackup',
            'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies',
            'PaperlessBilling', 'PaymentMethod']
        oh = OneHotEncoder(handle_unknown='ignore', sparse_output=False, drop='first')
        x_train_encoded = oh.fit_transform(train_cat[ohe_cols])
        x_test_encoded = oh.transform(test_cat[ohe_cols])
        encoded_cols = oh.get_feature_names_out(ohe_cols)
        x_train_encoded_df = pd.DataFrame(x_train_encoded, columns=encoded_cols, index=train_cat.index)
        x_test_encoded_df = pd.DataFrame(x_test_encoded, columns=encoded_cols, index=test_cat.index)
        train_cat = pd.concat([train_cat.drop(columns=ohe_cols), x_train_encoded_df], axis=1)
        test_cat = pd.concat([test_cat.drop(columns=ohe_cols), x_test_encoded_df], axis=1)
        train_cat['sim'] = train_cat['sim'].map({'Airtel': 0, 'BSNL': 1, 'Jio': 2, 'Vi': 3}).astype(int)
        test_cat['sim'] = test_cat['sim'].map({'Airtel': 0, 'BSNL': 1, 'Jio': 2, 'Vi': 3}).astype(int)
        logger.info(f"Final categorical columns:\n{train_cat.columns}")

        logger.info("Object columns before chi-square:", train_cat.select_dtypes(include='object').columns.tolist())
        logger.info(train_cat.head(3))

        return train_cat,test_cat,y_train,y_test

    except Exception as e:
        er_ty, er_msg, er_lin = sys.exc_info()
        logger.error(f"Issue in ordinal_encoding: line {er_lin.tb_lineno} due to: {er_msg}")


def chi_square_test(train_cat,test_cat,y_train):
    try:
        chi2_selector = SelectKBest(score_func=chi2, k='all')
        chi2_selector.fit(train_cat, y_train)

        chi2_scores = pd.DataFrame({
            'Feature': train_cat.columns,
            'Chi2_Score': chi2_selector.scores_,
            'P_Value': chi2_selector.pvalues_
        }).sort_values(by='Chi2_Score', ascending=False)

        print("Chi-Square Test Results:")
        print(chi2_scores)
        remove_features = chi2_scores[chi2_scores['P_Value'] > 0.05]['Feature']
        train_cat.drop(columns=remove_features, inplace=True, errors='ignore')
        test_cat.drop(columns=remove_features, inplace=True, errors='ignore')
        logger.info(f'Removed features: {remove_features}')

        return train_cat,test_cat,y_train

    except Exception as e:
        er_ty, er_msg, er_lin = sys.exc_info()
        logger.error(f"Issue in chi_square_test: line {er_lin.tb_lineno} due to: {er_msg}")

    # def anova_feature_selection(self):
    #
    #     anova_selector = SelectKBest(score_func=f_classif, k='all')
    #     anova_selector.fit(self.x_train_num, self.y_train)
    #     anova_results = pd.DataFrame({
    #         'Feature': self.x_train_num.columns,
    #         'F_Score': anova_selector.scores_,
    #         'P_Value': anova_selector.pvalues_
    #     }).sort_values(by='F_Score', ascending=False)
    #     print("Annova Test Results:")
    #     print(anova_results)
    #     remove_features = anova_results[anova_results['P_Value'] > 0.05]['Feature']
    #     self.x_train_num = self.x_train_num.drop(columns=remove_features)
    #     self.x_test_num = self.x_test_num.drop(columns=remove_features)
    #     logger.info(f'Removed features: {remove_features}')
    #     logger.info(f'Numerical columns: {(self.x_train_num.columns)}')
    #     logger.info(f'Categorical columns: {(self.x_train_cat.columns)}')

    # def corelation(self):
    #     from sklearn.preprocessing import LabelEncoder
    #     from scipy.stats import pearsonr
    #     le = LabelEncoder()
    #     y = le.fit_transform(self.y_train)
    #     results = []
    #     for col in self.x_train_num.columns:
    #         corr, p = pearsonr(self.x_train_num[col], y)
    #         results.append([col, corr, p])
    #     corr_df = pd.DataFrame(results, columns=['Feature', 'Correlation', 'P_Value'])
    #     corr_df = corr_df.sort_values(by='Correlation', ascending=False)
    #     print("🔹 Correlation (Pearson) Results with y_train:")
    #     print(corr_df)
    #     logger.info(f'null checking{self.x_train_num.isnull().sum()}')
    #     logger.info(f'null checking{self.x_train_cat.isnull().sum()}')
    #     logger.info(f'{self.x_train}')