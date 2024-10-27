import requests

url = "http://127.0.0.1:9696/predict"

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

response = requests.post(url, json=customer).json()


if response["churn"]:
    print(response)
    print(f"Sending promo email to {customer['customerid']}")
