import pandas as pd
from sklearn.cluster import KMeans
import joblib

def perform_clustering(df, n_clusters=5):
    # We cluster based on Income and Spending Score
    features = ['AnnualIncome', 'SpendingScore']
    X = df[features]
    
    kmeans = KMeans(n_clusters=n_clusters, init='k-means++', random_state=42)
    df['Cluster'] = kmeans.fit_predict(X)
    
    # Save the model
    joblib.dump(kmeans, 'models/kmeans_model.pkl')
    return df

if __name__ == "__main__":
    df = pd.read_csv('data/processed/cleaned_data.csv')
    df_clustered = perform_clustering(df)
    df_clustered.to_csv('data/processed/clustered_data.csv', index=False)
    print("Clustering complete. Model saved in models/")