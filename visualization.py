"""
TELCO CUSTOMER CHURN ANALYSIS & VISUALIZATION
---------------------------------------------
This script performs detailed exploratory data analysis (EDA)
on the Telco Customer Churn dataset.

Author: V Nikhil Kumar
"""

# ===============================
# 1️⃣ Importing Libraries
# ===============================
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

# ===============================
# 2️⃣ Loading the Dataset
# ===============================
df = pd.read_csv('C:\\Users\\nikhi\\Downloads\\Data science practice\\Churn_Prediction\\Telco_Data_With_Tax_Gateway_Updated.csv')

print("\n📘 Dataset Info:")
print(df.info())

print("\n📊 Random Sample:")
print(df.sample(10))

print("\n🔍 Missing Values:")
print(df.isnull().sum())

print("\n📑 Columns in DataFrame:")
print(df.columns)

# ===============================
# 3️⃣ Target Variable: Churn
# ===============================
print("\n📈 Churn Value Counts:")
print(df['Churn'].value_counts())

plt.figure(figsize=(4,3))
plt.title('Number of Customers Leaving')
plt.bar(x=df['Churn'].value_counts().index,
        height=df['Churn'].value_counts().values)
plt.show()

# ===============================
# 4️⃣ Gender Analysis
# ===============================
print(df['gender'].value_counts())

plt.figure(figsize=(10,3))
plt.subplot(1,2,1)
plt.title('Gender Distribution')
plt.bar(x=df['gender'].value_counts().index,
        height=df['gender'].value_counts().values)

plt.subplot(1,2,2)
sns.barplot(x=['Female-No','Female-Yes','Male-No','Male-Yes'],
            y=df.groupby('gender')['Churn'].value_counts(),
            color='g')
plt.title('Churn by Gender')
plt.show()

# ===============================
# 5️⃣ Dependents Analysis
# ===============================
plt.figure(figsize=(10,3))
plt.subplot(1,2,1)
plt.title('Dependents Count')
plt.bar(x=df['Dependents'].value_counts().index,
        height=df['Dependents'].value_counts().values)

plt.subplot(1,2,2)
sns.barplot(x=['ND-NC','ND-WC','D-NC','D-WC'],
            y=df.groupby('Dependents')['Churn'].value_counts(),
            color='r')
plt.title('Churn with Dependents')
plt.show()

# ===============================
# 6️⃣ Senior Citizen Analysis
# ===============================
plt.figure(figsize=(10,3))
plt.subplot(1,2,1)
plt.title('SeniorCitizen Distribution')
plt.pie(df['SeniorCitizen'].value_counts(),
        labels=['Not SeniorCitizen','SeniorCitizen'],
        autopct='%.2f', shadow=True)

plt.subplot(1,2,2)
plt.title('SeniorCitizen vs Churn')
plt.pie(df.groupby('SeniorCitizen')['Churn'].value_counts(normalize=True)*100,
        labels=['WSC-NC','WSC-WC','SC-NC','SC-WC'],
        autopct='%.2f', shadow=True)
plt.show()

# ===============================
# 7️⃣ Charges Overview
# ===============================
print("\nAverage Monthly Charges:", np.mean(df['MonthlyCharges']))
print("\nSample Total Charges:", df['TotalCharges'].sample(7))

# ===============================
# 8️⃣ Internet Service Analysis
# ===============================
plt.figure(figsize=(9,4))
plt.subplot(1,2,1)
plt.title('Internet Connectivity')
labels = ['No','Fiber optic','DSL']
sizes = [df['InternetService'].value_counts()[x] for x in labels]
plt.pie(sizes, labels=labels, autopct='%.2f', shadow=True)

plt.subplot(1,2,2)
plt.title('Internet Service vs Churn')
sns.countplot(x='InternetService',hue='Churn',data=df,palette='magma')
plt.show()

# ===============================
# 9️⃣ Payment Method Analysis
# ===============================
plt.figure(figsize=(20,5))
plt.subplot(1,2,1)
plt.title('Payment Methods')
sns.barplot(x=['Electronic Check','Mailed Check','Bank Transfer','Credit Card'],
            y=df['PaymentMethod'].value_counts(), color='g')

plt.subplot(1,2,2)
sns.countplot(x='PaymentMethod',hue='Churn',data=df,width=0.9)
plt.title('Payment Method vs Churn')
plt.show()

# ===============================
# 🔟 Dependents Histogram
# ===============================
sns.displot(df['Dependents'],kind='hist',color='r')
plt.title('Number of Dependents')
plt.show()

# ===============================
# 11️⃣ Boxplot: Monthly Charges
# ===============================
sns.boxplot(df['MonthlyCharges'])
plt.title('Monthly Charges Outliers')
plt.show()

# ===============================
# 12️⃣ Churn by Dependents Pie Chart
# ===============================
plt.figure(figsize=(5,4))
plt.title('Churn w.r.t Dependents')
Dependents = ['Without Dependents-No Churn','Without Dependents-Churn',
              'With Dependents-No Churn','With Dependents-Churn']
plt.pie(df.groupby('Churn')['Dependents'].value_counts(),
        labels=Dependents, autopct='%.2f',shadow=True)
plt.show()

# ===============================
# 13️⃣ Churn by Gender (again)
# ===============================
sns.barplot(x=['Female-No','Female-Yes','Male-No','Male-Yes'],
            y=df.groupby('gender')['Churn'].value_counts(),color='g')
plt.title('Churn with Gender')
plt.show()

# ===============================
# 14️⃣ Senior Citizen Normalized %
# ===============================
print(df.groupby('SeniorCitizen')['Churn'].value_counts(normalize=True)*100)

# ===============================
# 15️⃣ Distribution Plots
# ===============================
sns.displot(df['TotalCharges'])
plt.title('Total Charges Distribution')
plt.show()

sns.boxplot(data=df, x='Churn', y='TotalCharges', palette='coolwarm')
plt.title('Total Charges by Churn')
plt.show()

sns.displot(df['MonthlyCharges'],kde=True,color='r')
plt.title('Monthly Charges Distribution')
plt.show()

# ===============================
# 16️⃣ Gender vs SeniorCitizen
# ===============================
plt.figure(figsize=(5,4))
plt.title('Gender vs Senior Citizen')
GS=['Female-Without SC','Female-SC','Male-Without SC','Male-SC']
plt.pie(df.groupby('gender')['SeniorCitizen'].value_counts(normalize=True)*100,
        labels=GS,autopct='%.2f',shadow=True)
plt.show()


# 17️⃣ Streaming Services

sns.barplot(df['StreamingMovies'].value_counts())
plt.title('Streaming Movies Distribution')
plt.show()


# 18️⃣ Pairplot (Multivariate Overview)

sns.pairplot(df,hue='Churn')
plt.show()


# 19️⃣ Partner vs Churn

sns.countplot(x='Partner',hue='Churn',data=df)
plt.title('Partner vs Churn')
plt.show()


# 20️⃣ Internet Service vs Churn %

print(df.groupby('InternetService')['Churn'].value_counts(normalize=True)*100)

sns.barplot(x=['DSL-NC','DSL-WC','FO-NC','FO-WC','NS-NC','NS-WC'],
            y=df.groupby('InternetService')['Churn'].value_counts(normalize=True)*100,
            color='black')
plt.title('Churn w.r.t Internet Service')
plt.show()

# 21️⃣ Contract vs Churn

plt.figure(figsize=(12,5))
plt.subplot(1,2,1)
ax = sns.countplot(x='Contract', data=df)
ax.bar_label(ax.containers[0], fmt='N=%d')
plt.title('Contract Distribution')

plt.subplot(1,2,2)
ax2 = sns.countplot(x='Contract', hue='Churn', data=df)
[ax2.bar_label(c, fmt='%d') for c in ax2.containers]
plt.title('Churn by Contract Type')
plt.tight_layout()
plt.show()


# 22️⃣ Correlation Heatmap

plt.figure(figsize=(10,6))
sns.heatmap(df.corr(numeric_only=True), annot=True, cmap='coolwarm', linewidths=0.5)
plt.title('Correlation Heatmap of Numerical Features')
plt.show()


# ✅ End of Script

print("\n✅ EDA & Visualization Completed Successfully!")
