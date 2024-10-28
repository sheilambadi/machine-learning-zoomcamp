import pickle

from flask import Flask, jsonify, request

model_file = "model2.bin"
vectorizer_file = "dv.bin"

with open(model_file, "rb") as model_in:
    model = pickle.load(model_in)

with open(vectorizer_file, "rb") as dv_in:
    dv = pickle.load(dv_in)

app = Flask("subscription")


@app.route("/predict", methods=["POST"])
def predict():
    client = request.get_json()

    X = dv.transform([client])
    y_pred = model.predict_proba(X)[0, 1]
    subscribe = y_pred >= 0.5

    result = {
        "subscription_probability": round(y_pred, 3),
        "subscription": bool(subscribe),
    }

    return jsonify(result)


# this won't run if we use gunicorn -> gunicorn turns Flask into a WSGI app (production ready)
if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=9696)
