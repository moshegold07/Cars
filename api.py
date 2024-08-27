import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle
import pandas as pd
import os
from car_data_prep import prepare_data  # Make sure this file exists and contains the prepare_data function

app = Flask(__name__)

# Load the trained model
rf_model = pickle.load(open('trained_model.pkl', 'rb'))


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    features = {
        "manufactor": request.form.get('manufactor'),
        'Year': int(request.form.get('Year')),
        'model': request.form.get('model'),
        'Hand': int(request.form.get('Hand')),
        'Gear': request.form.get('Gear'),
        'capacity_Engine': int(request.form.get('capacity_Engine')),
        'Engine_type': request.form.get('Engine_type'),
        'Km': int(request.form.get('Km')),
        'Prev_ownership': request.form.get('Prev_ownership'),
        'Curr_ownership': request.form.get('Curr_ownership'),
        'Color': request.form.get('Color'),
        'Test': request.form.get('Test'),
        'Price': 0  # Placeholder for Price
    }

    input_df = pd.DataFrame([features])

    # Prepare the data in the same way as during training
    input_df = prepare_data(input_df)
    input_df = input_df.drop(columns=['Price'])
    # Ensure the column order matches the model
    input_df = input_df[rf_model.feature_names_in_]

    # Make prediction
    prediction = rf_model.predict(input_df)[0]
    if prediction < 3000:
        prediction = 3000

    print(prediction)
    return render_template('index.html', prediction_price=f"The prediction price is: ש''ח {int(prediction)}")

if __name__ == "__main__":
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=True)