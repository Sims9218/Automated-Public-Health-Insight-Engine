import os
import requests
import pandas as pd
from datetime import datetime

# 1. Calculate HRI (Simplified Weighted Index)
def calculate_hri(data):
    # Weights based on health impact (PM2.5 is most dangerous)
    weights = {'pm2_5': 0.4, 'pm10': 0.2, 'no2': 0.15, 'o3': 0.15, 'co': 0.1}
    # Standardizing against common safety limits (example values)
    limits = {'pm2_5': 25, 'pm10': 50, 'no2': 40, 'o3': 100, 'co': 10}
    
    hri_score = 0
    for pollutant, weight in weights.items():
        hri_score += (data[pollutant] / limits[pollutant]) * weight
    return round(hri_score * 100, 2)

# 2. Precaution Logic
def get_precautions(hri):
    if hri < 50: return "Air quality is good. Enjoy outdoor activities."
    elif hri < 100: return "Moderate risk. Sensitive individuals should reduce prolonged exertion."
    elif hri < 150: return "Unhealthy. Wear a mask (N95) and avoid heavy outdoor exercise."
    else: return "Hazardous! Stay indoors and use air purifiers if possible."

# 3. Fetch Data (Example using OpenWeatherMap)
def fetch_pollution_data():
    api_key = os.getenv("API_KEY")
    lat, lon = "19.0760", "72.8777" # Example: Mumbai
    url = f"http://api.openweathermap.org/data/2.5/air_pollution?lat={lat}&lon={lon}&appid={api_key}"
    response = requests.get(url).json()
    
    components = response['list'][0]['components']
    return components

# MAIN ENGINE EXECUTION
raw_data = fetch_pollution_data()
hri_value = calculate_hri(raw_data)
precaution = get_precautions(hri_value)

print(f"Current HRI: {hri_value}")
print(f"Precaution: {precaution}")

# Logic for "Self-Training" check would go here:
# Compare yesterday's prediction with today's actual raw_data.
# If error > threshold, trigger model.fit() and save new model.pkl
