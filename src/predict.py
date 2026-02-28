import pandas as pd
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import joblib

def train_prediction_model(df):
    # Features: Gender, Age, AnnualIncome, Cluster
    # Target: SpendingScore
    X = df[['Gender', 'Age', 'AnnualIncome', 'Cluster']]
    y = df['SpendingScore']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Hyperparameters for XGBoost
    model = XGBRegressor(
        n_estimators=200,
        learning_rate=0.05,
        max_depth=4,
        random_state=42
    )
    
    model.fit(X_train, y_train)
    
    # Evaluate
    preds = model.predict(X_test)
    print(f"Model R2 Score: {r2_score(y_test, preds):.4f}")
    
    # Save the model
    joblib.dump(model, 'models/xgboost_model.pkl')

if __name__ == "__main__":
    df = pd.read_csv('data/processed/clustered_data.csv')
    train_prediction_model(df)
    print("XGBoost training complete. Model saved in models/")