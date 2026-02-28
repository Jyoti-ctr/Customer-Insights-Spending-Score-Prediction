import streamlit as st
import pandas as pd
import joblib
import plotly.express as px
import os

# Set Page Config
st.set_page_config(page_title="Let's Intelligence make predictions", layout="wide")

# Load Models & Data
@st.cache_resource
def load_resources():
    kmeans = joblib.load('models/kmeans_model.pkl')
    xgb_model = joblib.load('models/xgboost_model.pkl')
    df = pd.read_csv('data/processed/clustered_data.csv')
    return kmeans, xgb_model, df

try:
    kmeans, xgb_model, df = load_resources()
except:
    st.error("Models not found! Please run 'main_run.py' first.")
    st.stop()

# --- SIDEBAR: Navigation ---
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Dashboard", "Score Predictor"])

# --- PAGE 1: Dashboard ---
if page == "Dashboard":
    st.title("🛍️ Customer Segmentation Dashboard")
    st.markdown("View natural groupings of customers based on their spending behavior.")

    # Interactive 3D Plot
    fig = px.scatter_3d(df, x='AnnualIncome', y='SpendingScore', z='Age',
                  color='Cluster', title="Customer Segments (3D)",
                  labels={'AnnualIncome': 'Income (k$)', 'SpendingScore': 'Score'},
                  color_continuous_scale='Viridis')
    st.plotly_chart(fig, use_container_width=True)

    # Statistics
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Cluster Distribution")
        st.bar_chart(df['Cluster'].value_counts())
    with col2:
        st.subheader("Average Income by Cluster")
        avg_income = df.groupby('Cluster')['AnnualIncome'].mean()
        st.line_chart(avg_income)

# --- PAGE 2: Predictor ---
else:
    st.title("🔮 Spending Score Predictor")
    st.write("Enter customer details below to predict their spending behavior.")

    col1, col2 = st.columns(2)
    with col1:
        age = st.slider("Customer Age", 18, 100, 30)
        gender = st.selectbox("Gender", ["Female", "Male"])
    with col2:
        income = st.number_input("Annual Income (k$)", 10, 200, 50)
        
    if st.button("Predict Spending Score"):
        gender_val = 1 if gender == "Male" else 0
        
        # Determine Cluster
        cluster = kmeans.predict([[income, 50]])[0] 
        
        # Predict Score
        features = pd.DataFrame([[gender_val, age, income, cluster]], 
                                columns=['Gender', 'Age', 'AnnualIncome', 'Cluster'])
        prediction = xgb_model.predict(features)[0]
        
        # Results Display
        st.success(f"### Predicted Spending Score: {prediction:.2f}")
        st.info(f"This customer belongs to **Segment {cluster}**")
        
        # Custom Advice
        if cluster == 2:
            st.warning("🎯 Recommendation: Target with high-end luxury products.")
        elif cluster == 3:
            st.warning("🏷️ Recommendation: Target with discounts and flash sales.")