import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import sys
import logging
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score,classification_report,confusion_matrix
from log_code import setup_logging
logger=setup_logging('churn_pred')
from transformation import VAR
from trimming import trim_tech
from constant import con
from quasi_constant import con_
from feature_s import ordinal_encoding
from feature_s import chi_square_test
# from feature_s import anova_feature_selection
# from feature_s import corelation
from train_algo import common
from sklearn.model_selection import train_test_split
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.impute import KNNImputer
from sklearn.preprocessing import QuantileTransformer
import pickle
import warnings
warnings.filterwarnings('ignore')



class VIS:
  try:
      def __init__(self):
          pass
  except Exception as e:
      er_ty, er_msg, er_tb = sys.exc_info()
      line_number = er_tb.tb_lineno
      print("Error Type:", er_ty)
      print("Error Message:", er_msg)
      print("Error Line No:", line_number)


  def load_data(self):
      try:
          self.df=pd.read_csv('C:\\Users\\nikhi\\Downloads\\Data science practice\\Churn_Prediction\\Telco_Data_With_Tax_Gateway_Updated.csv')
          self.df=self.df.drop(['customerID'],axis=1)
          self.df['SeniorCitizen']=self.df['SeniorCitizen'].map({0:'No',1:'Yes'})
          print(self.df.sample(7))
          print(self.df.isnull().sum())
          print(self.df.info())
          print(self.df.columns)

          # converting TotalCharges column into numeric
          self.df['TotalCharges'] = pd.to_numeric(self.df['TotalCharges'], errors='coerce')

          self.X=self.df.drop('Churn',axis=1) #independent
          self.y=self.df['Churn']    #dependent
          print(self.X.shape)
          #print(self.X.sample(7))
          print(self.y.shape)
          #print(self.y.sample(7))


          self.X_train,self.X_test,self.y_train,self.y_test=train_test_split(self.X,self.y,test_size=0.2,random_state=42)
          logger.info(f'X train shape: {self.X_train.shape}---X test shape:{self.X_test.shape}----y_train shape:{self.y_train.shape}------y_test shape:  {self.y_test.shape}')
          # logger.info(f'Checking missing values : {self.df.isnull().sum()}')
          # logger.info({self.X_train.isnull().sum()})

      except Exception as e:
          er_ty, er_msg, er_lin = sys.exc_info()
          logger.info(f'Issue is : {er_lin.tb_lineno} : due to : {er_msg}')

  def missing_val(self):
      try:
          # Check missing values
          total_missing = self.df.isnull().sum().sum()
          logger.info(f"Total missing values before imputation: {total_missing}")

          # Initialize KNN imputer
          self.imputer1 = KNNImputer(n_neighbors=10, weights='uniform')
          self.imputer1.fit(self.X_train[['TotalCharges']])

          # Transform both train and test
          self.X_train['TotalCharges'] = self.imputer1.transform(self.X_train[['TotalCharges']])
          self.X_test['TotalCharges'] = self.imputer1.transform(self.X_test[['TotalCharges']])

          logger.info(f"Missing values after imputation (train): {self.X_train['TotalCharges'].isnull().sum()}")
          logger.info(f"Missing values after imputation (test): {self.X_test['TotalCharges'].isnull().sum()}")

          # dividing into categorical and numerical
          self.X_train_num = self.X_train.select_dtypes(exclude='object')
          self.X_train_cat = self.X_train.select_dtypes(include='object')
          self.X_test_num = self.X_test.select_dtypes(exclude='object')
          self.X_test_cat = self.X_test.select_dtypes(include='object')
          logger.info(f'Numerical column names: {self.X_train_num.columns}')
          logger.info(f'checking missing values:{self.X_train_num.isnull().sum()}')
          logger.info(f'Categorical Column Names: {self.X_train_cat.columns}')



      except Exception as e:
           er_ty, er_msg, er_lin = sys.exc_info()
           logger.info(f'Issue is : {er_lin.tb_lineno} : due to : {er_msg}')


  def var_tranform(self):
      try:
          qtr = QuantileTransformer(output_distribution='normal', random_state=0)
          for i in self.X_train_num:
              self.X_train_num[i+'_qt']=qtr.fit_transform(self.X_train_num[[i]]).flatten()
              self.X_test_num[i+'_qt']=qtr.fit_transform(self.X_test_num[[i]]).flatten()
          s=[]
          for i in self.X_train_num:
              if '_qt' not in i:
                  s.append(i)

          self.X_train_num=self.X_train_num.drop(s,axis=1)
          self.X_test_num=self.X_test_num.drop(s,axis=1)
          logger.info(f'After variable Transformation X_train_num columns are : {self.X_train_num.columns}')

      except Exception as e:
           er_ty, er_msg, er_lin = sys.exc_info()
           logger.info(f'Issue is : {er_lin.tb_lineno} : due to : {er_msg}')


  def handle_outliers(self):
      try:
          # lets convert Each numerical column into log transformation
          # 70 % of outliers we can remove
          # then we pass to trimming formula to remove all outliers in the data
          self.X_train_num, self.X_test_num = VAR.transform(self.X_train_num, self.X_test_num)
          self.X_train_num, self.X_test_num = trim_tech(self.X_train_num, self.X_test_num)
          logger.info(f'Train Data Features : {self.X_train_num.columns}')
          logger.info(f'Test Data Features : {self.X_test_num.columns}')

          return self.X_train_num,self.X_test_num

      except Exception as e:
          er_ty, er_msg, er_lin = sys.exc_info()
          logger.info(f'Issue is : {er_lin.tb_lineno} : due to : {er_msg}')


  def feature_selection(self):
      try:

          self.X_train_num, self.X_test_num = con(self.X_train_num, self.X_test_num)
          logger.info(f'Train Column Names : {self.X_train_num.columns}')
          logger.info(f'Test Column Names : {self.X_test_num.columns}')
          self.X_train_num, self.X_test_num = con_(self.X_train_num, self.X_test_num)
          logger.info(f'After constant and quasi contant Techniques')
          logger.info(f'Train Column Names : {self.X_train_num.columns}')
          logger.info(f'Test Column Names : {self.X_test_num.columns}')
          self.X_train_cat,self.X_test_cat,self.y_train,self.y_test=ordinal_encoding(self.X_train_cat,self.X_test_cat,self.y_train,self.y_test)
          self.X_train_cat, self.X_test_cat, self.y_train=chi_square_test(self.X_train_cat,self.X_test_cat,self.y_train)
          #self.X_train_num, self.X_test_num, self.y_train, self.y_test = anova_feature_selection(self.X_train_num, self.X_test_num, self.y_train, self.y_test)
          #self.X_train_num, self.X_test_num, self.y_train, self.y_test = corelation(self.X_train_num, self.X_test_num, self.y_train, self.y_test)

          return self.X_train_cat,self.X_test_cat,self.y_train,self.y_test

      except Exception as e:
          er_ty, er_msg, er_lin = sys.exc_info()
          logger.info(f'Issue is : {er_lin.tb_lineno} : due to : {er_msg}')

  def merge_data(self):
      try:
          # reset index so that we can concat data perfectlly
          self.X_train_num.reset_index(drop=True, inplace=True)
          self.X_train_cat.reset_index(drop=True, inplace=True)

          self.X_test_num.reset_index(drop=True, inplace=True)
          self.X_test_cat.reset_index(drop=True, inplace=True)

          self.training_data = pd.concat([self.X_train_num, self.X_train_cat], axis=1)
          self.testing_data = pd.concat([self.X_test_num, self.X_test_cat], axis=1)

          logger.info(f'Training_data shape : {self.training_data.shape} -> {self.training_data.columns}')
          logger.info(f'Testing_data shape : {self.testing_data.shape} -> {self.testing_data.columns}')
      except Exception as e:
          er_ty, er_msg, er_lin = sys.exc_info()
          logger.info(f'Issue is : {er_lin.tb_lineno} : due to : {er_msg}')

  def balanced_data(self):
      try:
          logger.info('----------------Before Balancing------------------------')
          logger.info(f'Churn Yes Count: {sum(self.y_train == 1)}')
          logger.info(f'Churn No  Count: {sum(self.y_train == 0)}')
          # 🔍 Verify no categorical columns remain
          obj_cols = self.training_data.select_dtypes(include='object').columns
          if len(obj_cols) > 0:
              #logger.warning(f"⚠️ Found unencoded columns before SMOTE: {list(obj_cols)}")
              for col in obj_cols:
                  self.training_data[col] = self.training_data[col].astype('category').cat.codes
                  logger.info(f"Encoded {col} to numeric codes.")
          #SMOTE
          sm = SMOTE(random_state=42)
          self.training_data_res, self.y_train_res = sm.fit_resample(self.training_data, self.y_train)
          logger.info('----------------After Balancing-------------------------')
          logger.info(f'Balanced Yes: {sum(self.y_train_res == 1)}')
          logger.info(f'Balanced No : {sum(self.y_train_res == 0)}')
          logger.info(f'Balanced Columns: {self.training_data_res.shape[1]}')

      except Exception as e:
              er_ty, er_msg, er_lin = sys.exc_info()
              logger.info(f'Issue is : {er_lin.tb_lineno} : due to : {er_msg}')

  def feature_scaling(self):
      try:
          logger.info('---------Before scaling-------')
          logger.info(f'before sccaling columns  :\n {self.training_data_res.columns}')
          logger.info(f'{self.training_data_res.head(4)}')
          sc = StandardScaler()
          sc.fit(self.training_data_res)
          self.training_data_res_t = sc.transform(self.training_data_res)
          self.testing_data_t = sc.transform(self.testing_data)
          with open('standard_scalar.pkl', 'wb') as t:
              pickle.dump(sc, t)
          logger.info('----------After scaling--------')
          logger.info(f'{self.training_data_res_t}')

          model_features = list(self.training_data_res.columns)

          #import pickle
          with open('model_features.pkl', 'wb') as f:
              pickle.dump(model_features, f)

          print("✅ Saved model_features.pkl with", len(model_features), "features")


      except Exception as e:
          er_ty, er_msg, er_lin = sys.exc_info()
          logger.info(f'Issue is : {er_lin.tb_lineno} : due to : {er_msg}')

  # def train_models(self):
  #     try:
  #         common(self.training_data_res_t,self.y_train_res,self.testing_data_t,self.y_test)
  #     except Exception as e:
  #         er_ty, er_msg, er_lin = sys.exc_info()
  #         logger.info(f'Issue is : {er_lin.tb_lineno} : due to : {er_msg}')

  def best_model(self):
      try:
          logger.info(f'============Finalied Model Logistic Regression===============')
          self.reg_log = LogisticRegression()
          self.reg_log.fit(self.training_data_res_t, self.y_train_res)
          logger.info(f'Final All coulumns of training data:\n {self.training_data_res_t[:10]}')
          logger.info(f'Model Test Accuracy : {accuracy_score(self.y_test, self.reg_log.predict(self.testing_data))}')
          logger.info(f'Confusion Matrix : {confusion_matrix(self.y_test, self.reg_log.predict(self.testing_data))}')
          logger.info(f'Classification Report : {classification_report(self.y_test, self.reg_log.predict(self.testing_data))}')
          logger.info(f'=====Model Saving======')
          with open('churn_prediction.pkl', 'wb') as f:
              pickle.dump(self.reg_log, f)
      except Exception as e:
          er_ty, er_msg, er_lin = sys.exc_info()
          logger.info(f'Issue is : {er_lin.tb_lineno} : due to : {er_msg}')


if __name__=='__main__':
    try:
        obj=VIS()
        obj.load_data()
        obj.missing_val()
        obj.var_tranform()
        obj.handle_outliers()
        obj.feature_selection()
        obj.merge_data()
        obj.balanced_data()
        obj.feature_scaling()
        #obj.train_models()
        obj.best_model()

    except Exception as e:
        er_ty, er_msg, er_tb = sys.exc_info()
        line_number = er_tb.tb_lineno
        print("Error Type:", er_ty)
        print("Error Message:", er_msg)
        print("Error Line No:", line_number)
