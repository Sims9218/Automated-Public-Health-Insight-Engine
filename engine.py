import os
import requests
import pandas as pd
import joblib
import xgboost as xgb
from datetime import datetime, timedelta

# --- CONFIGURATION ---
DATA_PATH = 'data/pollution_history.csv'
MODEL_PATH = 'models/hri_model.pkl'
LOG_PATH = 'data/performance_log.csv'
RETRAIN_THRESHOLD = 15.0  # MAE points

# 1. HRI Calculation Logic
def calculate_hri(data):
    weights = {'pm2_5': 0.4, 'pm10': 0.2, 'no2': 0.15, 'o3': 0.15, 'co': 0.1}
    limits = {'pm2_5': 25, 'pm10': 50, 'no2': 40, 'o3': 100, 'co': 10}
    hri_score = sum((data.get(k, 0) / limits[k]) * weights[k] for k in weights)
    return round(hri_score * 100, 2)

def get_precautions(hri):
    if hri < 50:
        return "Air quality is good. Safe for outdoor activities."
    elif hri < 100:
        return "Moderate risk. Sensitive groups should limit outdoor exertion."
    elif hri < 150:
        return "Unhealthy. Wear an N95 mask and avoid heavy outdoor exercise."
    else:
        return "Hazardous! Stay indoors and use air purifiers."

# 2. Automated Retraining Logic
def retrain_model():
    if not os.path.exists(DATA_PATH): return
    df = pd.read_csv(DATA_PATH)
    if len(df) < 50: return # Need minimum data points to train
    
    X = df[['pm2_5', 'pm10', 'no2', 'o3', 'co']]
    y = df['hri_actual']
    
    model = xgb.XGBRegressor(n_estimators=100, learning_rate=0.05)
    model.fit(X, y)
    joblib.dump(model, MODEL_PATH)
    print("🤖 Model retrained due to high error.")

# 3. Main Execution Loop
def run_engine():
    # --- A. DATA INGESTION ---
    api_key = os.getenv("API_KEY")
    # Coordinates for your target location
    url = f"http://api.openweathermap.org/data/2.5/air_pollution?lat=19.07&lon=72.87&appid={api_key}"
    
    try:
        response = requests.get(url).json()
        raw_data = response['list'][0]['components']
    except Exception as e:
        print(f"Error fetching API data: {e}")
        return

    current_hri = calculate_hri(raw_data)
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M')

    # --- B. SELF-CORRECTION & PERFORMANCE LOGGING ---
    # We look at the LAST prediction made to compare with TODAY'S actual HRI
    if os.path.exists(LOG_PATH):
        logs = pd.read_csv(LOG_PATH)
        if not logs.empty:
            # Get the prediction from the previous run
            last_pred = logs['predicted_hri'].iloc[-1]
            error = abs(last_pred - current_hri)
            
            # Update the last row with the actual HRI and the calculated error
            logs.loc[logs.index[-1], 'actual_hri'] = current_hri
            logs.loc[logs.index[-1], 'error'] = error
            logs.to_csv(LOG_PATH, index=False)
            
            # Trigger Retraining if error exceeds threshold
            if error > RETRAIN_THRESHOLD:
                print(f"Model drift detected (Error: {error}). Retraining...")
                retrain_model()
    else:
        # Initialize the file if it doesn't exist
        empty_logs = pd.DataFrame(columns=['timestamp', 'predicted_hri', 'actual_hri', 'error'])
        empty_logs.to_csv(LOG_PATH, index=False)
        print("Initialized performance_log.csv")

    # --- C. ML PREDICTION FOR THE NEXT CYCLE ---
    if os.path.exists(MODEL_PATH):
        model = joblib.load(MODEL_PATH)
        # Prepare features for XGBoost
        features = pd.DataFrame([raw_data])[['pm2_5', 'pm10', 'no2', 'o3', 'co']]
        prediction = model.predict(features)[0]
    else:
        # If no model exists, our 'prediction' is just a baseline 
        prediction = current_hri 
        retrain_model() # Initial training to create the .pkl file

    # --- D. LOG THE NEW PREDICTION ---
    # We log the prediction NOW so that the NEXT run can compare it to reality
    new_row = pd.DataFrame([[timestamp, round(prediction, 2), None, None]], 
                           columns=['timestamp', 'predicted_hri', 'actual_hri', 'error'])
    new_row.to_csv(LOG_PATH, mode='a', header=not os.path.exists(LOG_PATH), index=False)

    # --- E. ARCHIVE RAW DATA FOR TRAINING ---
    raw_data['hri_actual'] = current_hri
    raw_data['timestamp'] = timestamp
    pd.DataFrame([raw_data]).to_csv(DATA_PATH, mode='a', header=not os.path.exists(DATA_PATH), index=False)

    # --- F. NOTIFICATIONS ---
    precaution = get_precautions(current_hri)
    print(f"Done! Current HRI: {current_hri} | Forecasted: {round(prediction, 2)}")


if __name__ == "__main__":
    run_engine()
