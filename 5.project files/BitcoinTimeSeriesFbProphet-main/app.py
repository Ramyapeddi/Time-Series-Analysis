import numpy as np
import pandas as pd
from flask import Flask, request, jsonify, render_template
import pickle
from datetime import datetime

# Initialize Flask app
app = Flask(__name__)

# Load the saved Prophet model
with open('prophet_model.pkl', 'rb') as f:
    m = pickle.load(f)

# Generate future dates and forecast (run once when the app starts)
future = m.make_future_dataframe(periods=365)
forecast = m.predict(future)
# Ensure 'ds' in forecast is in string format for easier matching
forecast['ds'] = forecast['ds'].dt.strftime('%Y-%m-%d')

# Route for the homepage
@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')

# Route for the prediction page
@app.route('/Bitcoin', methods=['POST', 'GET'])
def prediction():
    return render_template('predict.html')

# Route for handling prediction form submission
@app.route('/predict', methods=['POST'])
def y_predict():
    if request.method == "POST":
        # Get the date input from the form
        ds = request.form["Date"]
        print(f"Selected date: {ds}")

        # Use the input date as the prediction target
        next_day = ds
        print(f"Prediction date: {next_day}")

        try:
            # Find the prediction in the precomputed forecast
            prediction = forecast[forecast['ds'] == next_day]['yhat'].item()
            prediction = round(prediction, 2)
            print(f"Predicted value: {prediction}")

            # Render the prediction on the predict.html page
            return render_template('predict.html', 
                                 prediction_text=f"Bitcoin Price on selected date is ${prediction:,.2f}")
        except ValueError:
            # Handle case where the date is not in the forecast
            return render_template('predict.html', 
                                 prediction_text=f"No prediction available for {next_day}. Please select a date within the forecast range.")
    
    # If not POST, render the predict page without a prediction
    return render_template("predict.html")

# Run the Flask app
if __name__ == "__main__":
    app.run(debug=False)