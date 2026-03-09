from flask import Flask, jsonify, request, render_template
import joblib
import pandas as pd

app = Flask(__name__)

def load_model():
    model_file = open(r'C:\Users\juman\OneDrive\Desktop\car-price-prediction\model\carpriceprediction.joblib', 'rb')
    model = joblib.load(model_file)
    return model

model = load_model()

def prepare_data(data):
    Levy = data.get('Levy')
    Manufacturer = data.get('Manufacturer')
    Prod_year = data.get('Prod. year')
    Category = data.get('Category')
    Leather_interior = data.get('Leather interior')
    Fuel_type = data.get('Fuel type')
    Engine_volume = data.get('Engine volume')
    Mileage = data.get('Mileage')
    Gear_box_type = data.get('Gear box type')
    Drive_wheels = data.get('Drive wheels')

    prepared = pd.DataFrame({
        "Levy": [float(Levy)],
        "Manufacturer": [Manufacturer],
        "Prod. year": [int(Prod_year)],
        "Category": [Category],
        "Leather interior": [Leather_interior],
        "Fuel type": [Fuel_type],
        "Engine volume": [float(Engine_volume)],
        "Mileage": [int(Mileage)],
        "Gear box type": [Gear_box_type],
        "Drive wheels": [Drive_wheels]
    })

    return prepared


@app.route('/predict', methods=['POST'])
def predict():
    data = request.form
    prepared_data = prepare_data(data)
    prediction = model.predict(prepared_data)[0]

    return render_template('index.html', prediction=round(prediction, 2))

@app.route('/')
def home():
    return render_template('index.html')
app.run()