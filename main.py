import requests
import pandas as pd
import os
from datetime import datetime

API_KEY = os.getenv("f8a67246c69ba428919ace2c52c82cc7")
LAT, LON = 19.0330, 73.0297  # Navi Mumbai

def run_pipeline():
    # 1. Fetch Data
    url = f"http://api.openweathermap.org/data/2.5/air_pollution?lat={LAT}&lon={LON}&appid={API_KEY}"
    res = requests.get(url).json()
    
    # 2. Extract & Validate (Data Quality)
    data = res['list'][0]['components']
    aqi = res['list'][0]['main']['aqi']
    
    # DQ Check: PM2.5 shouldn't be zero or missing in Navi Mumbai
    if data['pm2_5'] <= 0:
        print("Data Quality Error: PM2.5 reading is impossible.")
        return

    # 3. Save to "Database" (CSV)
    new_entry = pd.DataFrame([{
        "date": datetime.now().strftime("%Y-%m-%d"),
        "aqi": aqi,
        "pm2_5": data['pm2_5'],
        "no2": data['no2']
    }])
    
    # Append to existing history.csv
    new_entry.to_csv("data/history.csv", mode='a', header=not os.path.exists("data/history.csv"), index=False)
    print(f"Success! Logged AQI {aqi} for today.")

if __name__ == "__main__":
    run_pipeline()
