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
    # A. Fetch Live Data
    api_key = os.getenv("API_KEY")
    url = f"http://api.openweathermap.org/data/2.5/air_pollution?lat=19.07&lon=72.87&appid={api_key}"
    raw_data = requests.get(url).json()['list'][0]['components']
    current_hri = calculate_hri(raw_data)
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M')

    # B. Self-Correction Check (Compare past prediction with current reality)
    if os.path.exists(LOG_PATH):
        logs = pd.read_csv(LOG_PATH)
        # Find if we had a prediction for 'now' made 24h ago
        # (Simplified: comparing latest prediction to current actual)
        last_pred = logs['predicted_hri'].iloc[-1]
        error = abs(last_pred - current_hri)
        
        # Log the error for tracking
        new_log = pd.DataFrame([[timestamp, last_pred, current_hri, error]], 
                               columns=['timestamp', 'predicted_hri', 'actual_hri', 'error'])
        new_log.to_csv(LOG_PATH, mode='a', header=False, index=False)
        
        if error > RETRAIN_THRESHOLD:
            retrain_model()

    # C. Make New Prediction for Tomorrow
    if os.path.exists(MODEL_PATH):
        model = joblib.load(MODEL_PATH)
        features = pd.DataFrame([raw_data])[['pm2_5', 'pm10', 'no2', 'o3', 'co']]
        prediction = model.predict(features)[0]
    else:
        prediction = current_hri # Fallback if no model exists yet
        retrain_model() # Initial training

    # D. Save Current Actual Data for future training
    raw_data['hri_actual'] = current_hri
    raw_data['timestamp'] = timestamp
    pd.DataFrame([raw_data]).to_csv(DATA_PATH, mode='a', header=not os.path.exists(DATA_PATH), index=False)

    print(f"Analysis Complete: Current HRI is {current_hri}. Forecasted HRI: {round(prediction, 2)}")

if __name__ == "__main__":
    run_engine()
