import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
import joblib

def load_and_clean(filepath):
    df = pd.read_csv(filepath)
    # Rename columns for easier coding
    df.columns = ['CustomerID', 'Gender', 'Age', 'AnnualIncome', 'SpendingScore']
    
    # Encode Gender
    le = LabelEncoder()
    df['Gender'] = le.fit_transform(df['Gender']) # Female=0, Male=1 usually
    
    return df

def scale_data(df, features):
    scaler = StandardScaler()
    df_scaled = scaler.fit_transform(df[features])
    return df_scaled, scaler

if __name__ == "__main__":
    data = load_and_clean('data/raw/Mall_Customers.csv')
    data.to_csv('data/processed/cleaned_data.csv', index=False)
    print("Data cleaned and saved to processed folder.")