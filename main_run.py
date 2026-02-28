import os
import pandas as pd
from src.preprocess import load_and_clean
from src.cluster import perform_clustering
from src.predict import train_prediction_model

# Ensure folders exist
for folder in ['data/processed', 'models']:
    os.makedirs(folder, exist_ok=True)

print("--- Step 1: Preprocessing ---")
# Adjust path to where you saved the Kaggle file
raw_data_path = 'data/raw/Mall_Customers.csv'
df_clean = load_and_clean(raw_data_path)
df_clean.to_csv('data/processed/cleaned_data.csv', index=False)

print("--- Step 2: Clustering (Customer Personas) ---")
df_clustered = perform_clustering(df_clean, n_clusters=5)
df_clustered.to_csv('data/processed/clustered_data.csv', index=False)

print("--- Step 3: Training Prediction Model (XGBoost) ---")
train_prediction_model(df_clustered)

print("\n✅ Project executed successfully! Check the 'models/' folder for your saved AI.")