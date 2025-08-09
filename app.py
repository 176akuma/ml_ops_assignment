# app.py
from flask import Flask, request, jsonify
import joblib
import numpy as np
import mlflow.pyfunc
import pandas as pd
import sqlite3
import logging
from datetime import datetime


app = Flask(__name__)

# Load the trained model
try:
    model = joblib.load('linear_regression.joblib')
except FileNotFoundError:
    print("Error: model.joblib not found. Please ensure your trained model is in the project directory.")
    exit()

# Set up logging
logging.basicConfig(filename='api.log', level=logging.INFO)

# Set up SQLite
conn = sqlite3.connect('logs.db', check_same_thread=False)
cursor = conn.cursor()
cursor.execute('''
    CREATE TABLE IF NOT EXISTS logs (
        timestamp TEXT,
        input TEXT,
        prediction TEXT
    )
''')
conn.commit()



feature_names = [
    'longitude',
    'latitude',
    'housing_median_age',
    'total_rooms',
    'total_bedrooms',
    'population',
    'households',
    'median_income',
    'median_house_value',
    'ocean_proximity'
]

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json(force=True)
        # Assuming input data is a dictionary matching model's expected features
        # Example: {"feature1": 10, "feature2": 20, "feature3": 30}
        
        # Convert input to a format suitable for your model (e.g., numpy array)
        # This part will depend on your specific model's input requirements
        # if 'features' not in data:
        #     return jsonify({"error" :f"Features not present"}), 400
        features = data["features"]
        print("Features :", features)
        # if not isinstance(features, list) or len(features) != 8:
        #     return jsonify({"error     :" :f"Expected 8 feature"}), 400
        
        input_array = np.array(features).reshape(1,-1)
        prediction = model.predict(input_array) # Assuming a single prediction
        prediction_value = prediction.tolist()

        # Log to DB and file
        log_entry = (str(datetime.utcnow()), str(input_array), str(prediction_value))
        cursor.execute("INSERT INTO logs (timestamp, input, prediction) VALUES (?, ?, ?)", log_entry)
        conn.commit()




        
        # prediction = model.predict(features)[0] # Assuming a single prediction
        logging.info(f"Input: {input_array} | Prediction: {prediction}")
        return jsonify({'prediction': prediction.item()}) # .item() for single numpy value
    except Exception as e:
        return jsonify({'error': str(e)}), 400
    
@app.route("/metrics", methods=["GET"])
def metrics():
    cursor.execute("SELECT COUNT(*) FROM logs")
    count = cursor.fetchone()[0]
    return jsonify({"total_predictions": count})


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)






    model.fit(X_train, Y_train)