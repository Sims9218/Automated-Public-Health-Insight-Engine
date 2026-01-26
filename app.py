import streamlit as st
import pandas as pd
import plotly.express as px

# Replace these with your actual GitHub username and repo name
GITHUB_USER = "sims9218"
REPO_NAME = "Automated-Public-Health-Insight-Engine"
RAW_URL_BASE = f"https://raw.githubusercontent.com/{GITHUB_USER}/{REPO_NAME}/main"

st.set_page_config(page_title="Health Risk Index Engine", layout="wide")

st.title("🌍 Automated Public Health Insight Engine")
st.markdown("Real-time air quality monitoring and 24h health risk predictions.")

# Function to load data from GitHub
@st.cache_data(ttl=3600) # Refresh every hour
def load_github_data(filename):
    url = f"{RAW_URL_BASE}/data/{filename}"
    return pd.read_csv(url)

try:
    # Load History and Logs
    history_df = load_github_data("pollution_history.csv")
    log_df = load_github_data("performance_log.csv")

    # --- TOP METRICS ---
    latest_data = history_df.iloc[-1]
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Current HRI", f"{latest_data['hri_actual']}")
    with col2:
        st.metric("PM2.5", f"{latest_data['pm2_5']} µg/m³")
    with col3:
        st.metric("O3 (Ozone)", f"{latest_data['o3']} µg/m³")
    with col4:
        st.metric("Latest Error", f"{round(log_df['error'].iloc[-1], 2)}")

    # --- PRECAUTION BOX ---
    hri = latest_data['hri_actual']
    if hri < 50:
        st.success(f"### ✅ Level: Good (HRI: {hri})\nPrecautions: Air quality is satisfactory. Outdoor activities are safe.")
    elif hri < 100:
        st.warning(f"### ⚠️ Level: Moderate (HRI: {hri})\nPrecautions: Sensitive groups should reduce heavy outdoor exertion.")
    else:
        st.error(f"### 🚨 Level: High Risk (HRI: {hri})\nPrecautions: Avoid outdoor activities. Wear N95 masks if commuting.")

    # --- CHARTS ---
    st.divider()
    c1, c2 = st.columns(2)
    
    with c1:
        st.subheader("HRI Trend Over Time")
        fig_hri = px.line(history_df, x='timestamp', y='hri_actual', title="Actual HRI (Past 30 Days)")
        st.plotly_chart(fig_hri, use_container_width=True)
        
    with c2:
        st.subheader("Model Prediction vs Actual")
        # Comparing the last 10 entries of the log
        fig_perf = px.bar(log_df.tail(10), x='timestamp', y=['predicted_hri', 'actual_hri'], 
                          barmode='group', title="Forecast Accuracy")
        st.plotly_chart(fig_perf, use_container_width=True)

except Exception as e:
    st.error(f"Waiting for data... Ensure your GitHub Action has run at least once. Error: {e}")
