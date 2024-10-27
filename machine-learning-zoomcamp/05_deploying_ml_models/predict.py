import pickle

C = 1.0
output_file = f"model_C={C}.bin"

# Load and use the model
with open(output_file, "rb") as f_in:
    dv, model = pickle.load(f_in)

customer = {
    "customerid": "7590-vhveg",
    "gender": "female",
    "seniorcitizen": 0,
    "partner": "yes",
    "dependents": "no",
    "tenure": 1,
    "phoneservice": "no",
    "multiplelines": "no_phone_service",
    "internetservice": "dsl",
    "onlinesecurity": "no",
    "onlinebackup": "yes",
    "deviceprotection": "no",
    "techsupport": "no",
    "streamingtv": "no",
    "streamingmovies": "no",
    "contract": "month-to-month",
    "paperlessbilling": "yes",
    "paymentmethod": "electronic_check",
    "monthlycharges": 29.85,
    "totalcharges": 29.85,
    "churn": 0,
}

X = dv.transform([customer])
y_pred = model.predict_proba(X)[0, 1]

print(f"Input: {customer}")
print(f"Churn probability: {y_pred:.3f}")
