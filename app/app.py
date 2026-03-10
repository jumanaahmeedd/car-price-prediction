from flask import Flask, request, jsonify, render_template
import pandas as pd
import joblib

app = Flask(__name__)

# Load trained pipeline (model + preprocessing)
model = joblib.load("car_price_model.joblib")


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():

    try:
        # get form data
        data = request.form.to_dict()

        # convert numeric fields
        numeric_cols = [
            "Levy",
            "Mileage",
            "Engine volume",
            "Prod. year",
            "Cylinders",
            "Airbags"
        ]

        for col in numeric_cols:
            if col in data:
                data[col] = float(data[col])

        # convert to dataframe
        df = pd.DataFrame([data])

        # prediction
        prediction = model.predict(df)[0]

        return render_template(
            "index.html",
            prediction_text=f"Estimated Car Price: ${round(prediction,2)}"
        )

    except Exception as e:
        return jsonify({"error": str(e)})


@app.route("/predict", methods=["POST"])
def predict_api():

    try:
        data = request.get_json(force=True)

        df = pd.DataFrame([data])

        prediction = model.predict(df)[0]

        return jsonify({
            "predicted_price": float(prediction)
        })

    except Exception as e:
        return jsonify({"error": str(e)})


if __name__ == "__main__":
    app.run(debug=True)