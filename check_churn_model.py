import pickle

# === Change path to your churn_model.pkl ===
model_path = "C:\\Users\\nikhi\\Downloads\\Data science practice\\Churn_Prediction\\churn_prediction.pkl"

with open(model_path, "rb") as f:
    model = pickle.load(f)

print("✅ Object Type:", type(model))

# For normal sklearn models (like LogisticRegression)
if hasattr(model, "coef_"):
    print("➡️ Logistic Regression detected.")
    print("Number of features expected:", model.coef_.shape[1])

# For Pipeline objects
elif hasattr(model, "named_steps"):
    print("➡️ Pipeline detected! Steps:", model.named_steps.keys())
    if "classifier" in model.named_steps:
        clf = model.named_steps["classifier"]
        print("Classifier type:", type(clf))
        if hasattr(clf, "coef_"):
            print("Classifier expects features:", clf.coef_.shape[1])
    if hasattr(model, "feature_names_in_"):
        print("Feature names available:", model.feature_names_in_)

# For models with feature_names_in_ (Scikit-learn 1.0+)
elif hasattr(model, "feature_names_in_"):
    print("➡️ Model trained with feature names.")
    print("Feature count:", len(model.feature_names_in_))
    print("Column names:", model.feature_names_in_)

else:
    print("⚠️ Could not detect feature information. It might not store column names directly.")
