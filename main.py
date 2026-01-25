import requests
import pandas as pd
import os
from datetime import datetime

# Fetch API Key from GitHub Secrets
API_KEY = os.getenv("OPENWEATHER_API_KEY")
LAT, LON = 19.0330, 73.0297  # Navi Mumbai

def run_pipeline():
    url = f"http://api.openweathermap.org/data/2.5/air_pollution?lat={LAT}&lon={LON}&appid={API_KEY}"
    response = requests.get(url)
    res = response.json()
    
    # 1. Check if the API returned an error code
    if response.status_code != 200:
        print(f"❌ API Error: {res.get('message', 'Unknown error')}")
        return

    # 2. Check if 'list' exists before accessing it
    if 'list' not in res:
        print(f"❌ Data Error: 'list' key missing. Full response: {res}")
        return

    # 3. Only now do we proceed with extraction
    data = res['list'][0]['components']
    aqi = res['list'][0]['main']['aqi']
        
        # Data Quality Check
        if data['pm2_5'] <= 0:
            print("⚠️ Data Quality Alert: PM2.5 reading is invalid.")
            return

        # 4. Save to CSV
        os.makedirs('data', exist_ok=True)
        file_path = "data/history.csv"
        
        new_entry = pd.DataFrame([{
            "date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "aqi": aqi,
            "pm2_5": data['pm2_5'],
            "no2": data['no2']
        }])
        
        new_entry.to_csv(file_path, mode='a', header=not os.path.exists(file_path), index=False)
        print(f"✅ Success! Logged AQI {aqi}")

    except Exception as e:
        print(f"❌ Pipeline Failed: {str(e)}")

if __name__ == "__main__":
    run_pipeline()
