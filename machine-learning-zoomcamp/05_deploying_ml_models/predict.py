import pickle

from flask import Flask, jsonify, request

C = 1.0
output_file = f"model_C={C}.bin"

# Load and use the model
with open(output_file, "rb") as f_in:
    dv, model = pickle.load(f_in)

app = Flask("churn")


@app.route("/predict", methods=["POST"])
def predict():
    customer = request.get_json()

    X = dv.transform([customer])
    y_pred = model.predict_proba(X)[0, 1]
    churn = y_pred >= 0.5

    result = {"churn_probability": y_pred, "churn": bool(churn)}

    return jsonify(result)


# this won't run if we use gunicorn -> gunicorn turns Flask into a WSGI app (production ready)
if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=9696)
