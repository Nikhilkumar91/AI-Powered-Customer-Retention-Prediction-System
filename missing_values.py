from cProfile import label

from seaborn import displot, kdeplot
from sklearn.model_selection import train_test_split
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer,SimpleImputer
from sklearn.impute import KNNImputer
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import sys

class MV:
    try:
        def __init__(self):
            self.df=pd.read_csv('C:\\Users\\nikhi\\Downloads\\Data science practice\\Churn_Prediction\\WA_Fn-UseC_-Telco-Customer-Churn.csv')
            self.df.sample(7)
            self.df=self.df.drop(['customerID'],axis=1)
            print(self.df.sample(5))
            print(self.df.info())
            self.df['TotalCharges']=pd.to_numeric(self.df['TotalCharges'],errors='coerce')
            print(self.df.info())
    except Exception as e:
        er_ty, er_msg, er_tb = sys.exc_info()
        line_number = er_tb.tb_lineno
        print("Error Type:", er_ty)
        print("Error Message:", er_msg)
        print("Error Line No:", line_number)

    def missing_val(self):
        try:

            #ForwardFill method
            self.df['TotalCharges-ff']=self.df['TotalCharges'].fillna(method='ffill')
            print(self.df.isnull().sum())


            #BackWard Filling method

            self.df['TotalCharges-bf'] = self.df['TotalCharges'].fillna(method='bfill')
            print(self.df.isnull().sum())


            #Simple Imputer Technique
            #print(f'Before Imputation : {self.df.isnull().sum()}')
            self.imputer_s = SimpleImputer(missing_values=np.nan, strategy='most_frequent')
            self.imputer_data_s = self.imputer_s.fit_transform(self.df[['TotalCharges']])
            self.df['TotalCharges-SI'] = self.imputer_data_s
            print(self.df['TotalCharges-SI'].sample(10))
            # print(self.df['TotalCharges'].isnull().sum())
            print(self.df.isnull().sum())


            #IterativeImputer method
            print(f'Before Imputation : {self.df.isnull().sum()}')
            self.imputer=IterativeImputer(max_iter=10,random_state=42)
            self.imputer_data=self.imputer.fit_transform(self.df[['TotalCharges']])
            self.df['TotalCharges-IM']=self.imputer_data
            print(self.df['TotalCharges-IM'].sample(10))
            #print(self.df['TotalCharges'].isnull().sum())
            print(self.df.isnull().sum())

            #knnImputer method
            self.imputer1=KNNImputer(n_neighbors=10,weights='uniform')
            self.imputer_data1=self.imputer1.fit_transform(self.df[['TotalCharges']])
            self.df['TotalCharges-KM']=self.imputer_data1
            print(self.df['TotalCharges-KM'].isnull().sum())
            print(f'After Imutations - {self.df.isnull().sum()}')


            #standard deviations
            print(f"STD of Simple Imputer is : {self.df['TotalCharges-SI'].std()}")
            print(f"STD of Original column is : {self.df['TotalCharges'].std()}")
            print(f"STD of IterativeImputer is : {self.df['TotalCharges-IM'].std()}")
            print(f"STD of KNNImputer is : {self.df['TotalCharges-KM'].std()}")
            print(f"STD of Forward filling is : {self.df['TotalCharges-ff'].std()}")
            print(f"STD of Backward filling is : {self.df['TotalCharges-bf'].std()}")
            print('\n')



            #visualize
            plt.figure(figsize=(8,6))
            imputaion_techqs=['Original','SI','FF','BF','IM',"KNNM"]
            std_values=[self.df['TotalCharges'].std(),self.df['TotalCharges-SI'].std(),self.df['TotalCharges-ff'].std(),self.df['TotalCharges-bf'].std(),self.df['TotalCharges-IM'].std(),self.df['TotalCharges-KM'].std()]
            plt.bar(imputaion_techqs,std_values)
            plt.xlabel("Imputation Techniques")
            plt.ylabel("Standard Deviation")
            plt.title("Comparison of Standard Deviations for Different Imputation Techniques")
            for i, val in enumerate(std_values):
                plt.text(i, val + 5, f'{val:.3f}', ha='center', va='bottom', fontsize=10, fontweight='bold')
            plt.show()

            print("from the visualization the standard deviation of KNN imputer and iterative imputer is same. So, Iam going with the KNN imputer technique")

        except Exception as e:
            er_ty, er_msg, er_tb = sys.exc_info()
            line_number = er_tb.tb_lineno
            print("Error Type:", er_ty)
            print("Error Message:", er_msg)
            print("Error Line No:", line_number)

if __name__=='__main__':
    try:
        object=MV()
        object.missing_val()
    except Exception as e:
        er_ty, er_msg, er_tb = sys.exc_info()
        line_number = er_tb.tb_lineno
        print("Error Type:", er_ty)
        print("Error Message:", er_msg)
        print("Error Line No:", line_number)

