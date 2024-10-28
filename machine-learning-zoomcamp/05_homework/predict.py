import pickle

model_file = "model1.bin"
vectorizer_file = "dv.bin"

with open(model_file, "rb") as model_in:
    model = pickle.load(model_in)

with open(vectorizer_file, "rb") as dv_in:
    dv = pickle.load(dv_in)

client = {"job": "management", "duration": 400, "poutcome": "success"}

X = dv.transform([client])
y_pred = model.predict_proba(X)

print(y_pred)
