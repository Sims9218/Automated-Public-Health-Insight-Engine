import xgboost as xgb
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split

def train_and_save_model(data_path='data/pollution_history.csv'):
    df = pd.read_csv(data_path)
    
    # Feature Engineering: Use previous hours as 'lags' to predict next 24h
    # For a simple version, we'll use current values to predict next-day HRI
    X = df[['pm2_5', 'pm10', 'no2', 'o3', 'co']]
    y = df['hri_actual']  # The value we want to predict
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    
    # Model configuration
    model = xgb.XGBRegressor(
        n_estimators=100,
        learning_rate=0.05,
        max_depth=5,
        objective='reg:squarederror'
    )
    
    model.fit(X_train, y_train)
    
    # Save the model
    joblib.dump(model, 'models/hri_model.pkl')
    print("Model updated and saved.")

if __name__ == "__main__":
    train_and_save_model()
