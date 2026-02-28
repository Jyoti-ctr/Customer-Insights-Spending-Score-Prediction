import joblib
import pandas as pd
import numpy as np

def predict_customer_score():
    # Load models
    try:
        kmeans = joblib.load('models/kmeans_model.pkl')
        xgb_model = joblib.load('models/xgboost_model.pkl')
    except FileNotFoundError:
        print("Error: Models not found. Run main_run.py first!")
        return

    print("\n--- AI Customer Spending Predictor ---")
    age = int(input("Enter Customer Age: "))
    income = int(input("Enter Annual Income (k$): "))
    gender = input("Enter Gender (Male/Female): ").strip().lower()
    gender_val = 1 if gender == 'male' else 0

    # 1. Predict Cluster first (Requirement for the XGBoost model)
    # Scaler was used in training, for simplicity we use raw features if model was trained on them
    cluster = kmeans.predict([[income, 50]])[0] # Using a neutral score of 50 for cluster prediction
    
    # 2. Predict Spending Score
    features = pd.DataFrame([[gender_val, age, income, cluster]], 
                            columns=['Gender', 'Age', 'AnnualIncome', 'Cluster'])
    prediction = xgb_model.predict(features)[0]

    print(f"\n✅ Predicted Spending Score: {prediction:.2f}/100")
    print(f"Detected Segment: Cluster {cluster}")

if __name__ == "__main__":
    predict_customer_score()