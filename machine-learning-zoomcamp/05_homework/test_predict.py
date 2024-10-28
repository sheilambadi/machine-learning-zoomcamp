import requests

url = "http://127.0.0.1:9696/predict"

client = {"job": "student", "duration": 280, "poutcome": "failure"}

response = requests.post(url, json=client).json()

print(response)
