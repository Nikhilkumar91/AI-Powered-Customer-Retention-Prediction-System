# AI-Powered Customer Retention Prediction System
---
🧠 Goal

The goal of this project is to develop a Machine Learning–based Churn Prediction System that predicts whether a telecom customer will leave (Churn) or stay (Retain) based on their service usage, billing details, and demographic patterns.
This helps telecom companies take proactive actions like customer retention campaigns, discount offers, and service improvements.
------
📂 Dataset Overview
Source: Telco_Data_With_Tax_Gateway_Updated.csv
Rows: 7043
Columns: 21 features + Target (Churn)
Target Variable: Churn (Yes = Customer left, No = Customer stayed)

## 🔑 Key Features

| **Feature** | **Description** |
|--------------|-----------------|
| Gender | Male / Female |
| Partner / Dependents | Customer relationship info |
| InternetService | DSL / Fiber optic / None |
| PaymentMethod | Electronic check, Credit card, etc. |
| Contract | Month-to-month / One year / Two year |
| Tenure, MonthlyCharges, TotalCharges | Numeric variables |
| Churn | Target variable (Yes/No) |

-----
## 📊 End-to-End Project Workflow — AI-Powered Customer Retention Prediction System

---

### 1️⃣ Data Uncleaned → Data Cleaning Process

**🎯 Goal:**  
Prepare raw telecom dataset for machine learning.

**🛠️ Actions Performed:**
- Removed irrelevant column: `customerID`
- Converted invalid strings in `TotalCharges` to numeric
- Handled missing and blank values in key features
- Checked for datatype mismatches, duplicates, and whitespace issues

**✅ Result:**  
Clean and structured dataset ready for preprocessing.

---
### 2️⃣ Feature Engineering / Missing Value Imputation

**🎯 Goal:**  
Replace missing values effectively while preserving statistical integrity.

**⚙️ Methods Tried:**

| **Method** | **Description** | **Observation** |
|-------------|-----------------|-----------------|
| Forward Fill | Replaced missing values using previous non-null entries. | Worked for time-series style data but not random missingness. |
| Backward Fill | Used next valid value to fill NaNs. | Similar limitations as forward fill. |
| Simple Imputer | Replaced with mean/median. | Too basic, lost variance information. |
| Iterative Imputer | Modeled missing values using other variables iteratively. | Gave stable results but slightly slower. |
| KNN Imputer | Used K-nearest neighbors to estimate missing values. | ✅ Most accurate for continuous & correlated features. |

**✅ Finalized:**  
`KNNImputer()`

**📍 Reason:**  
It leverages feature similarity to fill gaps — ideal for this structured telecom dataset with correlated numeric features like `MonthlyCharges`, `TotalCharges`, and `tenure`.

---
### 3️⃣ Variable Transformation

**🎯 Goal:**  
Transform non-normal features to approximate a Gaussian distribution and improve model stability.

**⚙️ Methods Tried:**

| **Transformation** | **Description** | **Observation** |
|---------------------|-----------------|-----------------|
| Log Transform | For right-skewed variables | Not suitable for zero/negative values |
| Arcsin | Works for proportion data | Not applicable here |
| Box-Cox | Strong but only positive data | Limited usability |
| Yeo-Johnson | Handles negative/zero | Good results, slight skew |
| Quantile Transformer | Maps data to uniform/normal distribution | ✅ Excellent normalization across all numeric features |

**✅ Finalized:**  
`QuantileTransformer(output_distribution='normal')`

**📍 Reason:**  
Provides smooth Gaussian-like data distribution, preserving outlier structure while improving model convergence.

---
## 4️⃣ Handling Outliers

**🎯 Goal:**  
Reduce outlier influence to stabilize model training.

| **Method** | **Description** | **Observation** |
|-------------|-----------------|-----------------|
| Power Transformer | Normalized variance but distorted relationships |  |
| Quantile Transform | Reduced extreme values but overly smooth |  |
| Winsorizer | Caps extreme values using IQR range | ✅ Balanced trimming and preserved shape |

**✅ Finalized:**  
Winsorizer (IQR-based)

**📍 Reason:**  
Winsorizing effectively capped extreme billing outliers without data loss — especially in `MonthlyCharges` and `TotalCharges`.

---

## 5️⃣ Feature Selection

**🎯 Goal:**  
Select the most relevant features and remove low-variance or redundant ones.

**🧮 Filter Methods Used:**
- **Constant Method:** Removed features with a single unique value.  
- **Quasi-Constant Method:** Removed features with very low variance (<1%).

**✅ Result:**  
Improved feature set with only meaningful variation retained.

---

## 6️⃣ Categorical → Numerical Encoding

**🎯 Goal:**  
Convert categorical data into machine-understandable numerical format.

| **Encoding Type** | **Columns** | **Reason** |
|--------------------|-------------|-------------|
| Ordinal Encoding | `Contract` (Month-to-month → 0, One year → 1, Two year → 2) | Natural order hierarchy |
| Label Encoding | `Churn (Yes=1, No=0)` | Binary target variable |
| One-Hot Encoding | `gender`, `InternetService`, `PaymentMethod`, `OnlineSecurity`, `OnlineBackup`, `DeviceProtection`, `TechSupport`, `StreamingTV`, `StreamingMovies`, `PaperlessBilling`, `MultipleLines` | Non-ordinal multi-class categorical variables |

**✅ Result:**  
Transformed categorical columns into a structured numeric feature space for modeling.

---

## 7️⃣ Hypothesis Testing

**🎯 Goal:**  
Statistically verify feature significance with respect to churn.

| **Method** | **Suitable For** | **Purpose** |
|-------------|------------------|--------------|
| Chi-Square Test | Categorical vs Target | Test dependence between churn and categorical features |
| ANOVA | Continuous vs Target | Compare means across churn groups |
| Correlation Matrix | Numeric-Numeric | Measure linear relationships |

**✅ Finalized:**  
`Chi-Square Test`

**📍 Reason:**  
Ideal for categorical telecom data; effectively identified high-impact features like `Contract`, `InternetService`, and `TechSupport`.

---

## 8️⃣ Merging Data

**🎯 Goal:**  
Combine numeric and encoded categorical data into a single training dataset.

**✅ Action:**  
`pd.concat([train_num, train_cat], axis=1)` after encoding and scaling.

**✅ Result:**  
Unified dataset for model training with consistent indexing.

---

## 9️⃣ Balancing Data (SMOTE)

**🎯 Goal:**  
Handle target class imbalance since “Churn = Yes” cases were underrepresented.

**✅ Used:**  
`SMOTE (Synthetic Minority Oversampling Technique)`

**📍 Reason:**  
Creates synthetic samples for the minority class, improving recall and reducing bias.

**✅ Result:**  
Balanced target distribution — improved model fairness and generalization.

---

## 🔟 Train All Machine Learning Models

**🧠 Models Trained & Compared:**

| **Model** | **Type** | **Performance** |
|------------|-----------|----------------|
| Logistic Regression | Linear Classifier | ⭐ Excellent interpretability |
| Decision Tree | Tree-based | Overfit slightly |
| Random Forest | Ensemble | Stable but slower |
| K-Nearest Neighbors | Distance-based | Moderate accuracy |
| Naïve Bayes | Probabilistic | Poor fit for mixed data |

---

## 1️⃣1️⃣ Model Selection using ROC-AUC

**📈 Metric Used:**  
*AUC-ROC Curve (Area Under the Receiver Operating Characteristic)*

| **Model** | **AUC-ROC** | **Result** |
|------------|-------------|-------------|
| Logistic Regression | 0.77 ✅ | Best |
| Random Forest | 0.72 | Good |
| Decision Tree | 0.67 | Acceptable |
| KNN | 0.71 | Lower accuracy |

**✅ Finalized Model:**  
`Logistic Regression`

**📍 Reason:**  
Highest AUC, interpretable coefficients, consistent probability outputs, minimal overfitting.

---

## 1️⃣2️⃣ Train on Best Model

**⚙️ Steps:**
- Re-trained Logistic Regression on the **full balanced dataset**
- Used scaled numeric features
- Applied optimized hyperparameters: `C=1.0`, `solver='liblinear'`
- Final feature set after Chi-square filtering

**✅ Output:**  
Saved final performance metrics and model artifacts.

---

## 1️⃣3️⃣ Save Model Artifacts

**🧾 Pickled for Deployment:**

| **File** | **Description** |
|-----------|-----------------|
| `churn_prediction.pkl` | Trained Logistic Regression model |
| `standard_scalar.pkl` | StandardScaler used for numeric features |
| `model_features.pkl` | Feature column order used in model |

**✅ Benefit:**  
Ensures consistent real-time predictions in the Flask app.
---

## 1️⃣5️⃣ Prediction Output

**🎯 Goal:**  
Predict if the customer will churn or stay.

**📊 Example Output:**
🟥 Customer will CHURN
Probability: 78.3%

✅ Customer will STAY
Probability: 21.7%

**✅ Business Use:**  
Telecom teams can focus on **high-risk customers** with churn probability > 70%,  
enhancing customer retention and reducing revenue loss.

---
## 🧰 Tools & Libraries

**🧮 Core Languages & Frameworks**
- Python 3.9+
- Flask (for web app deployment)
- Render (for cloud hosting)

**📦 Libraries Used**
- NumPy  
- Pandas  
- Matplotlib  
- Seaborn  
- scikit-learn  
- feature-engine  
- imblearn  
- xgboost  

---

## 📊 Key Insights

- Customers with **month-to-month contracts** churn more frequently.  
- **Fiber optic** users have higher churn rates.  
- Customers **without dependents or partners** are more likely to leave.  
- **Electronic check** payment method strongly correlates with higher churn.  
- **Tenure** and **contract length** are strong predictors of customer retention.  

---
---

## 👨‍💻 Developer 
-----
   Nikhil Kumar

**💼 Machine Learning Engineer | AI & ML Enthusiast | Data Science Enthusiast**

---

### 🧾 Background

Hi! I’m **V. Nikhil Kumar**, a passionate **Machine Learning Engineer** with a strong interest in data-driven solutions.  
I specialize in building **predictive models**, **automating ML pipelines**, and developing **end-to-end machine learning web applications** that solve real-world problems.

---

### 💪 Skills

| **Category** | **Technologies / Tools** |
|---------------|---------------------------|
| Programming | Python |
| Machine Learning | scikit-learn, XGBoost, feature-engine, imblearn |
| Deep Learning | TensorFlow / Keras (basics) |
| Web Development | Flask |
| Data Visualization | Matplotlib, Seaborn |
| Databases | SQL |
| Version Control | Git & GitHub |

---

### 💼 Previous Works

| **Project** | **Description** | **Live Link** |
|--------------|-----------------|---------------|
| 📊 **Credit Card Customer Analysis** | Data-driven insights into customer credit usage patterns. | 🔗 *Coming Soon* |
| 💰 **Salary and Profit Predictor** | ML regression project for profit prediction. | 🌐 [**View Project ↗**](https://simple-and-multiple-regression-project.onrender.com/) |

---

### 📞 Contact

- **LinkedIn:** [linkedin.com/in/nikhilkumar91](https://linkedin.com/in/nikhilkumar91)  
- **Email:** [nikhilkumarchary30@gmail.com](mailto:nikhilkumarchary30@gmail.com)  
- **GitHub:** [github.com/Nikhilkumar91](https://github.com/Nikhilkumar91)  
- **Mobile:** +91 9133164879  

---

> 🌟 *Built with passion for AI, Data, and Real-world Impact.*

---

