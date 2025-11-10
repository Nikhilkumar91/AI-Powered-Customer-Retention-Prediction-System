from flask import Flask, render_template, request
import pandas as pd
import pickle
import traceback

app = Flask(__name__)

# ---------------------- Load model, scaler, and features ----------------------
with open('churn_prediction.pkl', 'rb') as f:
    model = pickle.load(f)

with open('standard_scalar.pkl', 'rb') as t:
    scaler = pickle.load(t)

with open('model_features.pkl', 'rb') as f:
    model_features = pickle.load(f)

# Numeric columns used in model training (if present)
numeric_cols = ['tenure', 'MonthlyCharges', 'TotalCharges']

# ---------------------- Home Page ----------------------
@app.route('/')
def home():
    return render_template('index.html')


# ---------------------- Prediction Route ----------------------
@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Step 1️⃣ — Collect form inputs
        data = {
            'gender': request.form.get('gender'),
            'tenure': float(request.form.get('tenure')),
            'MonthlyCharges': float(request.form.get('MonthlyCharges')),
            'TotalCharges': float(request.form.get('TotalCharges')),
            'SeniorCitizen': request.form.get('SeniorCitizen'),
            'Partner': request.form.get('Partner'),
            'Dependents': request.form.get('Dependents'),
            'PhoneService': request.form.get('PhoneService'),
            'MultipleLines': request.form.get('MultipleLines'),
            'PaperlessBilling': request.form.get('PaperlessBilling'),
            'PaymentMethod': request.form.get('PaymentMethod'),
            'InternetService': request.form.get('InternetService'),
            'OnlineSecurity': request.form.get('OnlineSecurity'),
            'OnlineBackup': request.form.get('OnlineBackup'),
            'DeviceProtection': request.form.get('DeviceProtection'),
            'TechSupport': request.form.get('TechSupport'),
            'StreamingTV': request.form.get('StreamingTV'),
            'StreamingMovies': request.form.get('StreamingMovies'),
            'Contract': request.form.get('Contract'),
            'sim': request.form.get('sim')
        }

        # Step 2️⃣ — Create a DataFrame
        df = pd.DataFrame([data])

        # Step 3️⃣ — Encode input like training
        encoded = {}

        # Numeric columns
        encoded['tenure'] = df['tenure'][0]
        encoded['MonthlyCharges'] = df['MonthlyCharges'][0]
        encoded['TotalCharges'] = df['TotalCharges'][0]

        # Mapping categorical values
        sim_map = {'Airtel': 0, 'BSNL': 1, 'Jio': 2, 'Vi': 3}
        contract_map = {'Month-to-month': 0, 'One year': 1, 'Two year': 2}
        encoded['sim'] = sim_map.get(df['sim'][0], 0)
        encoded['Contract'] = contract_map.get(df['Contract'][0], 0)

        # Binary columns
        encoded['SeniorCitizen_1'] = 1 if df['SeniorCitizen'][0] == '1' else 0
        encoded['Partner_Yes'] = 1 if df['Partner'][0] == 'Yes' else 0
        encoded['Dependents_Yes'] = 1 if df['Dependents'][0] == 'Yes' else 0
        encoded['MultipleLines_Yes'] = 1 if df['MultipleLines'][0] == 'Yes' else 0
        encoded['PaperlessBilling_Yes'] = 1 if df['PaperlessBilling'][0] == 'Yes' else 0

        # Internet Service encoding
        encoded['InternetService_Fiber optic'] = 1 if df['InternetService'][0] == 'Fiber optic' else 0
        encoded['InternetService_No'] = 1 if df['InternetService'][0] == 'No' else 0

        # Yes/No + "No internet service" pattern
        yes_no_features = [
            'OnlineSecurity', 'OnlineBackup', 'DeviceProtection',
            'TechSupport', 'StreamingTV', 'StreamingMovies'
        ]
        for feat in yes_no_features:
            val = df[feat][0]
            encoded[f'{feat}_No internet service'] = 1 if val == 'No internet service' else 0
            encoded[f'{feat}_Yes'] = 1 if val == 'Yes' else 0

        # Payment method encoding
        encoded['PaymentMethod_Credit card (automatic)'] = 1 if df['PaymentMethod'][0] == 'Credit card (automatic)' else 0
        encoded['PaymentMethod_Electronic check'] = 1 if df['PaymentMethod'][0] == 'Electronic check' else 0
        encoded['PaymentMethod_Mailed check'] = 1 if df['PaymentMethod'][0] == 'Mailed check' else 0

        # Create DataFrame from encoded dictionary
        input_df = pd.DataFrame([encoded])

        # Step 4️⃣ — Align input_df with model features
        for col in model_features:
            if col not in input_df.columns:
                input_df[col] = 0
        input_df = input_df.reindex(columns=model_features)

        # Step 5️⃣ — Scale numeric features only if they exist
        for col in numeric_cols:
            if col in input_df.columns:
                input_df[[col]] = scaler.transform(input_df[[col]])

        # Step 6️⃣ — Predict churn
        prediction = model.predict(input_df)[0]
        probabilities = model.predict_proba(input_df)[0]
        churn_prob = round(probabilities[1] * 100, 2)
        stay_prob = round(probabilities[0] * 100, 2)

        # Pass all details to template
        return render_template(
            'result.html',
            prediction=prediction,
            churn_prob=churn_prob,
            stay_prob=stay_prob
        )

        result = "Customer is likely to CHURN ❌ " if prediction == 1 else "Customer is likely to STAY ✅ "

        # Step 7️⃣ — Return result to frontend
        return render_template('result.html', prediction_text=result)

    except Exception as e:
        print("Prediction Error:", traceback.format_exc())
        return render_template('index.html', prediction_text=f"Error: {e}")


# ---------------------- Run Flask App ----------------------
if __name__ == '__main__':
    app.run(debug=True)
