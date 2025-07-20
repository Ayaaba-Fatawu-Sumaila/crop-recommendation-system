from flask import Flask, request, render_template
import numpy as np
import pandas as pd
import pickle

# Initialize Flask application
app = Flask(__name__)

# Load the crop recommendation model
with open("crop_recommendation.pkl", "rb") as file:
    model = pickle.load(file)

# Define the feature names in the exact order used during training
FEATURE_NAMES = ['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall']

@app.route("/")
def index():
    """Render the main page with the form"""
    return render_template("index.html", prediction_text=None)

@app.route("/predict", methods=["POST"])
def predict_crop():
    """Handle form submission and return prediction"""
    try:
        # Get form data
        form_data = {
            'N': float(request.form["N"]),
            'P': float(request.form["P"]),
            'K': float(request.form["K"]),
            'temperature': float(request.form["temperature"]),
            'humidity': float(request.form["humidity"]),
            'ph': float(request.form["ph"]),
            'rainfall': float(request.form["rainfall"])
        }
        
        # Create a DataFrame with the input data (preserving column order)
        input_df = pd.DataFrame([form_data], columns=FEATURE_NAMES)
        
        # Make prediction
        prediction = model.predict(input_df)
        crop_name = prediction[0]
        
    except Exception as e:
        # Handle errors gracefully
        error_message = f"Prediction error: {str(e)}"
        return render_template("index.html", 
                               prediction_text=error_message,
                               form_data=request.form)

    # Crop information database
    crop_info = {
        "rice": {"temp": "22-30°C", "ph": "5-6.5", "rain": "150-300mm"},
        "wheat": {"temp": "12-25°C", "ph": "6-7.5", "rain": "50-100mm"},
        "maize": {"temp": "18-27°C", "ph": "5.5-7.5", "rain": "60-110mm"},
        "cotton": {"temp": "21-30°C", "ph": "5.5-8.5", "rain": "50-100mm"},
        "jute": {"temp": "24-37°C", "ph": "6-7.5", "rain": "150-250mm"},
        "sugarcane": {"temp": "21-27°C", "ph": "6-7.5", "rain": "1100-1500mm"},
        "coconut": {"temp": "20-30°C", "ph": "5-8", "rain": "1000-2000mm"},
        "apple": {"temp": "21-24°C", "ph": "5.5-6.5", "rain": "100-125mm"},
        "mango": {"temp": "24-27°C", "ph": "5.5-7.5", "rain": "89-158mm"},
        "banana": {"temp": "26-30°C", "ph": "6-7.5", "rain": "200-250mm"},
        "grapes": {"temp": "15-40°C", "ph": "5.5-6.5", "rain": "50-70mm"},
        "watermelon": {"temp": "21-29°C", "ph": "6-6.8", "rain": "50-75mm"},
        "orange": {"temp": "13-37°C", "ph": "6-7", "rain": "100-200mm"},
        "papaya": {"temp": "21-33°C", "ph": "6-6.5", "rain": "150-200mm"},
        "muskmelon": {"temp": "18-30°C", "ph": "6-6.8", "rain": "50-75mm"},
        "pomegranate": {"temp": "25-35°C", "ph": "5.5-7", "rain": "50-75mm"},
        "lentil": {"temp": "18-30°C", "ph": "5.5-7", "rain": "80-100mm"},
        "blackgram": {"temp": "25-35°C", "ph": "6.5-7.8", "rain": "60-75mm"},
        "mungbean": {"temp": "27-30°C", "ph": "6.2-7.5", "rain": "60-70mm"},
        "coffee": {"temp": "15-24°C", "ph": "6-6.5", "rain": "150-250mm"},
        # Add more crops as needed
    }
    
    # Get crop info (case-insensitive match)
    crop_key = crop_name.lower()
    optimal = crop_info.get(crop_key, {"temp": "N/A", "ph": "N/A", "rain": "N/A"})
    
    # Pass data to template
    return render_template("index.html",
                           prediction_text=crop_name,
                           optimal_temp=optimal["temp"],
                           optimal_ph=optimal["ph"],
                           optimal_rain=optimal["rain"],
                           form_data=form_data)



if __name__ == "__main__":
    app.run(debug=True, port=5000, host='localhost')

