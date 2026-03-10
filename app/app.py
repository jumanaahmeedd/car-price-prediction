from flask import Flask, request, jsonify, render_template
import pandas as pd
import joblib

app = Flask(__name__)

# Load trained model
model = joblib.load(r"C:\Users\juman\OneDrive\Desktop\car-price-prediction\model\carpriceprediction.joblib")


# Home page
@app.route("/")
def home():
    return render_template("index.html")


# Prediction from HTML form
from flask import Flask, request, render_template, jsonify
import pandas as pd
import joblib

app = Flask(__name__)

# Load the trained model
def load_model():
    with open(r'C:\Users\juman\OneDrive\Desktop\car-price-prediction\model\carpriceprediction.joblib', 'rb') as f:
        return joblib.load(f)

model = load_model()

# List of numeric and categorical columns
numeric_cols = ["Levy", "Engine volume", "Mileage", "Prod. year"]
categorical_cols = ["Manufacturer", "Category", "Leather interior", "Fuel type", "Gear box type", "Drive wheels"]

# Prepare data safely
def prepare_data(data):
    prepared = {}
    # Convert numeric fields
    for col in numeric_cols:
        try:
            prepared[col] = [float(data.get(col, 0))]
        except ValueError:
            prepared[col] = [0]  # default if conversion fails

    # Convert categorical fields
    for col in categorical_cols:
        prepared[col] = [data.get(col, "")]  # empty string if missing

    return pd.DataFrame(prepared)

# Home page with HTML form
@app.route('/', methods=['GET'])
def home():
    return render_template('index.html')

# Prediction route for HTML form
@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.form
        df = prepare_data(data)
        prediction = model.predict(df)[0]
        return render_template('index.html', prediction_text=f"Estimated Price: ${round(prediction, 2)}")
    except Exception as e:
        return render_template('index.html', prediction_text=f"Error: {str(e)}")

# Prediction API route for Postman (form-data only)
@app.route('/predict_api', methods=['POST'])
def predict_api():
    try:
        data = request.form
        df = prepare_data(data)
        prediction = model.predict(df)[0]
        return jsonify({"predicted_price": float(prediction)})
    except Exception as e:
        return jsonify({"error": str(e)})

if __name__ == "__main__":
    app.run(debug=True)