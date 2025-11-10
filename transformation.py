from seaborn import displot, kdeplot
from sklearn.model_selection import train_test_split
from sklearn.impute import KNNImputer
import matplotlib.pyplot as plt
from log_code import setup_logging
logger=setup_logging('transform_log')
from feature_engine.transformation import LogCpTransformer,ArcsinTransformer,PowerTransformer
import seaborn as sns
import numpy as np
import pandas as pd
import sys
from sklearn.preprocessing import MinMaxScaler,QuantileTransformer,PowerTransformer
from scipy import stats


class VAR:
    def transform(X_train_num,X_test_num):
        # Select only numeric columns
        #X_test_num=X_test_num.drop(['SeniorCitizen'],axis=1)
        #X_train_num = X_train_num.drop(['SeniorCitizen'], axis=1)
        numeric_cols = X_test_num.select_dtypes(include=[np.number]).columns
        #numeric_cols=numeric_cols.drop(['SeniorCitizen'],axis=1)

        # Copy dataframe
        df_transformed = X_test_num.copy()

        # Initialize transformers
        scaler = MinMaxScaler()
        qt = QuantileTransformer(output_distribution='normal', random_state=0)
        pt = PowerTransformer(method='yeo-johnson')

        # Loop through each numeric column
        for col in numeric_cols:
            # ---------------- LOG TRANSFORMATION ----------------
            df_transformed[f'{col}_logcp'] = np.log1p(df_transformed[col] - df_transformed[col].min() + 1)

            # ---------------- ARCSIN TRANSFORMATION ----------------
            scaled = scaler.fit_transform(df_transformed[[col]])
            df_transformed[f'{col}_arcsin'] = np.arcsin(np.sqrt(scaled)).flatten()

            # ---------------- BOXCOX TRANSFORMATION ----------------
            try:
                shifted = df_transformed[col] - df_transformed[col].min() + 1  # must be > 0
                df_transformed[f'{col}_boxcox'], _ = stats.boxcox(shifted)
            except Exception as e:
                print(f"⚠️ Box-Cox skipped for {col}: {e}")

            # ---------------- QUANTILE TRANSFORMATION ----------------
            df_transformed[f'{col}_quantile'] = qt.fit_transform(df_transformed[[col]]).flatten()

            # ---------------- YEO-JOHNSON TRANSFORMATION ----------------
            df_transformed[f'{col}_yeojohnson'] = pt.fit_transform(df_transformed[[col]]).flatten()

        # ✅ Check new columns added
        print("Transformed columns added:")
        print([col for col in df_transformed.columns if
               any(x in col for x in ['logcp', 'arcsin', 'boxcox', 'quantile', 'yeojohnson'])])

        logger.info(f'Transformation Columns added : {df_transformed.columns}')
        logger.info(f' X_train_num columns are : {X_train_num.columns}')

        # for col in numeric_cols:
        #     for suffix in ['logcp', 'arcsin', 'boxcox', 'quantile', 'yeojohnson']:
        #         transformed_col = f'{col}_{suffix}'
        #         if transformed_col in df_transformed.columns:
        #             plt.figure(figsize=(14, 4))
        #
        #             # Histogram + KDE
        #             plt.subplot(1, 3, 1)
        #             sns.kdeplot(df_transformed[transformed_col], color='blue', fill=True)
        #             plt.title(f'{transformed_col} - Distribution')
        #
        #             # Boxplot
        #             plt.subplot(1, 3, 2)
        #             sns.boxplot(x=df_transformed[transformed_col], color='lightgreen')
        #             plt.title(f'{transformed_col} - Boxplot')
        #
        #             # Probability Plot
        #             plt.subplot(1, 3, 3)
        #             stats.probplot(df_transformed[transformed_col], dist="norm", plot=plt)
        #             plt.title(f'{transformed_col} - Q–Q Plot')
        #
        #             plt.suptitle(f'Transformation: {suffix.upper()} | Column: {col}', fontsize=14, fontweight='bold')
        #             plt.tight_layout()
        #             plt.show()

        return X_train_num, X_test_num








