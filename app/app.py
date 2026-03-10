from flask import Flask, request, render_template, jsonify
import pandas as pd
import joblib

app = Flask(__name__)

# Load your trained model
def load_model():
    with open(r'C:\Users\juman\OneDrive\Desktop\car-price-prediction\model\carpriceprediction.joblib', 'rb') as f:
        return joblib.load(f)

model = load_model()

# Updated columns (Cylinders and Airbags removed)
numeric_cols = ["Levy", "Engine volume", "Mileage", "Prod. year"]
categorical_cols = ["Manufacturer", "Category", "Leather interior", "Fuel type", "Gear box type", "Drive wheels"]

# Safe data preparation
def prepare_data(data):
    prepared = {}
    
    # Convert numeric fields
    for col in numeric_cols:
        try:
            prepared[col] = [float(data.get(col, 0))]
        except (ValueError, TypeError):
            prepared[col] = [0]  # fallback if conversion fails

    # Convert categorical fields
    for col in categorical_cols:
        prepared[col] = [data.get(col, "")]  # fallback if missing

    return pd.DataFrame(prepared)

# HTML form route
@app.route('/', methods=['GET'])
def home():
    return render_template('index.html')

# HTML form prediction
@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.form
        df = prepare_data(data)
        prediction = model.predict(df)[0]
        return render_template('index.html', prediction_text=f"${round(prediction, 2)}")
    except Exception as e:
        return render_template('index.html', prediction_text=f"Error: {str(e)}")

# API endpoint for Postman
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